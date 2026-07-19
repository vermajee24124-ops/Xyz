#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

# Remove invisible formatting marks that can break API auth headers or prompts.
_INVISIBLE_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\uFEFF]")
def strip_invisible(value: str) -> str:
    return _INVISIBLE_RE.sub("", value).strip()


# Public GitHub safe workflow:
# - code stays in GitHub
# - cleaned transcript + analysis go directly to Hugging Face
# - nothing sensitive is committed to GitHub

HF_REPO = "Kumarverma11/PocketFM_Audio"
HF_TYPE = "dataset"

SOURCE_FOLDER = "Transcripts_Episode_0001_to_0200"
OUTPUT_FOLDER = "Veda_Training_Ready_FINAL_0001_to_0200"

TRACK_A_FOLDER = f"{OUTPUT_FOLDER}/TRACK_A_CLEAN_EPISODES"
TRACK_B_FOLDER = f"{OUTPUT_FOLDER}/TRACK_B_STORY_INTELLIGENCE"
DATASETS_FOLDER = f"{OUTPUT_FOLDER}/TRAINING_DATASETS"
STATE_FOLDER = f"{OUTPUT_FOLDER}/STATE"

BATCH_SIZE = 20
REQUEST_DELAY_SECONDS = 1.0
MAX_RETRIES = 4

# Verified NVIDIA Build free-endpoint models.
CLEAN_MODEL = "deepseek-ai/deepseek-v4-pro"
ANALYSIS_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

MERGED_RANGES = (
    (111, 120),
    (121, 130),
    (133, 135),
    (138, 139),
    (140, 143),
)

WORK = Path("/tmp/veda_public_safe_final")
RAW_DIR = WORK / "raw"
CLEAN_DIR = WORK / "clean"
INTEL_DIR = WORK / "intel"
STATE_DIR = WORK / "state"
for p in (RAW_DIR, CLEAN_DIR, INTEL_DIR, STATE_DIR):
    p.mkdir(parents=True, exist_ok=True)


def secret(name: str) -> str:
    value = strip_invisible(os.getenv(name, ""))
    if not value:
        raise RuntimeError(f"Missing GitHub secret: {name}")
    return value


HF_TOKEN = secret("HF_TOKEN")
NVIDIA_API_KEY = secret("NVIDIA_API_KEY")

api = HfApi(token=HF_TOKEN)
http = requests.Session()


def normalize_text(text: str) -> str:
    text = strip_invisible(unicodedata.normalize("NFC", text).replace("\ufeff", ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([।!?])\1{1,}", r"\1", text)
    return text.strip()


def parse_episode_number(path: str) -> Optional[int]:
    m = re.search(r"episode[_\s-]*0*(\d{1,4})", Path(path).name, re.I)
    return int(m.group(1)) if m else None


def in_merged_range(n: int) -> bool:
    return any(start <= n <= end for start, end in MERGED_RANGES)


def list_source_paths() -> Dict[int, str]:
    prefix = f"{SOURCE_FOLDER}/"
    result: Dict[int, str] = {}

    for item in api.list_repo_tree(HF_REPO, repo_type=HF_TYPE, recursive=True):
        path = getattr(item, "path", "")
        if not (path.startswith(prefix) and path.lower().endswith(".txt")):
            continue
        ep = parse_episode_number(path)
        if ep and 1 <= ep <= 200:
            if ep in result:
                raise RuntimeError(f"Duplicate source episode {ep}: {result[ep]} AND {path}")
            result[ep] = path

    missing = [n for n in range(1, 201) if n not in result]
    if missing:
        raise RuntimeError(f"Missing source episodes: {missing}")

    return result


def remote_completed() -> set[int]:
    completed: set[int] = set()
    prefix = f"{TRACK_B_FOLDER}/"
    try:
        for item in api.list_repo_tree(HF_REPO, repo_type=HF_TYPE, recursive=True):
            path = getattr(item, "path", "")
            if not (path.startswith(prefix) and path.endswith(".json")):
                continue
            ep = parse_episode_number(path)
            if ep:
                completed.add(ep)
    except Exception:
        pass
    return completed


def download_episode(repo_path: str) -> str:
    local = hf_hub_download(
        repo_id=HF_REPO,
        repo_type=HF_TYPE,
        filename=repo_path,
        token=HF_TOKEN,
        local_dir=str(RAW_DIR),
    )
    return Path(local).read_text(encoding="utf-8", errors="replace")


def parse_wait(value: Optional[str], default: int = 60) -> float:
    if not value:
        return float(default)
    try:
        return max(1.0, float(value))
    except Exception:
        pass
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)\s*(ms|s|m|h)", value.lower()):
        x = float(num)
        if unit == "ms":
            total += x / 1000.0
        elif unit == "s":
            total += x
        elif unit == "m":
            total += x * 60.0
        elif unit == "h":
            total += x * 3600.0
    return max(1.0, total or float(default))


def call_nvidia(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float = 0.1,
    enable_thinking: bool = False,
    reasoning_budget: Optional[int] = None,
    retries: int = MAX_RETRIES,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": strip_invisible(model),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if enable_thinking:
        extra_body: Dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": True}}
        if reasoning_budget is not None:
            extra_body["reasoning_budget"] = reasoning_budget
        payload["extra_body"] = extra_body

    last_error = ""

    for attempt in range(1, retries + 1):
        try:
            response = http.post(NVIDIA_BASE_URL, headers=headers, json=payload, timeout=(30, 900))
            if response.status_code in (429, 500, 502, 503, 504):
                wait = parse_wait(
                    response.headers.get("retry-after")
                    or response.headers.get("x-ratelimit-reset-requests")
                    or response.headers.get("x-ratelimit-reset-tokens"),
                    default=60,
                )
                wait += min(4.0, attempt)
                print(f"{model}: HTTP {response.status_code}, retry in {wait:.1f}s")
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()

        except Exception as exc:
            last_error = str(exc)
            if attempt < retries:
                wait = min(90.0, 8.0 * (2 ** (attempt - 1)))
                print(f"{model}: error on attempt {attempt}/{retries}: {exc}")
                print(f"{model}: retry in {wait:.1f}s")
                time.sleep(wait)
            else:
                break

    raise RuntimeError(f"{model} failed after retries: {last_error}")


def extract_message_text(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}

    pieces: List[str] = []
    for key in ("content", "reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            pieces.append(value.strip())
        elif isinstance(value, list):
            for part in value:
                if isinstance(part, str):
                    pieces.append(part)
                elif isinstance(part, dict):
                    txt = part.get("text") or part.get("content")
                    if isinstance(txt, str) and txt.strip():
                        pieces.append(txt.strip())

    return "\n".join(pieces).strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    decoder = json.JSONDecoder()
    found: List[Dict[str, Any]] = []
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    found.append(obj)
            except Exception:
                pass

    if found:
        return found[-1]

    raise ValueError("No valid JSON object found")


def safe_text(prompt: str, model: str, max_tokens: int, thinking: bool = False) -> str:
    data = call_nvidia(
        model=model,
        messages=[
            {
                "role": "system",
                "content": strip_invisible(
                    "You are a careful transcript editor for a fictional Hindi drama. "
                    "Preserve the original story, do not invent new events, and output only the requested text."
                ),
            },
            {"role": "user", "content": strip_invisible(prompt)},
        ],
        max_tokens=max_tokens,
        temperature=0.1,
        enable_thinking=thinking,
        reasoning_budget=max_tokens if thinking else None,
        retries=MAX_RETRIES,
    )
    text = extract_message_text(data)
    if not text:
        raise RuntimeError(f"{model} returned empty text")
    return text.strip()


def safe_json(prompt: str, model: str, max_tokens: int, thinking: bool = True) -> Dict[str, Any]:
    data = call_nvidia(
        model=model,
        messages=[
            {
                "role": "system",
                "content": strip_invisible(
                    "You are a continuity analyst for a fictional Hindi drama. "
                    "Use only transcript-supported facts. Return valid JSON only."
                ),
            },
            {"role": "user", "content": strip_invisible(prompt)},
        ],
        max_tokens=max_tokens,
        temperature=0.1,
        enable_thinking=thinking,
        reasoning_budget=max_tokens if thinking else None,
        retries=MAX_RETRIES,
    )
    text = extract_message_text(data)
    if not text:
        raise RuntimeError(f"{model} returned empty JSON text")
    return extract_json_object(text)


def clean_prompt(ep: int, current: str, prev_text: str, next_text: str) -> str:
    if in_merged_range(ep):
        boundary = (
            f"\nMerged-range repair mode is active for episode {ep}.\n"
            "Check only for overlap, duplicated lines, and sentences cut at the boundary.\n"
            "Do not rewrite the story, do not summarize, and do not add new dialogue.\n"
            f"PREVIOUS EPISODE END:\n{prev_text[-1800:]}\n"
            f"NEXT EPISODE START:\n{next_text[:1800]}\n"
        )
    else:
        boundary = "\nNormal clean mode. Do not change episode boundaries.\n"

    return (
        "This is a fictional Hindi drama transcript.\n"
        "Fix only ASR mistakes, Hindi grammar, spelling, punctuation, spacing, and obvious character-name errors.\n"
        "Keep the story, tone, order of events, and dialogue meaning unchanged.\n"
        "Return only the corrected transcript text.\n"
        f"{boundary}\n"
        f"CURRENT EPISODE {ep} RAW TEXT:\n{current}"
    )


def boundary_repair(ep: int, prev_clean: str, current_clean: str, next_clean: str) -> Tuple[str, str]:
    prompt = (
        f"Episode {ep-1} and Episode {ep} are from a split merged audio file.\n"
        "Repair only the boundary overlap or cut sentence.\n"
        "Return valid JSON only with keys left_tail, right_head, changed.\n"
        "Do not add new content.\n"
        f"LEFT EPISODE TAIL:\n{prev_clean[-1400:]}\n"
        f"RIGHT EPISODE HEAD:\n{current_clean[:1400]}"
    )

    obj = safe_json(prompt, CLEAN_MODEL, max_tokens=2200, thinking=False)
    left_tail = str(obj.get("left_tail", "")).strip()
    right_head = str(obj.get("right_head", "")).strip()

    if not left_tail or not right_head:
        raise RuntimeError("Boundary repair JSON missing left_tail/right_head")

    new_prev = prev_clean[:-min(len(prev_clean), 1400)] + left_tail if prev_clean else left_tail
    new_cur = right_head + current_clean[min(len(current_clean), 1400):] if current_clean else right_head
    return new_prev, new_cur


TRACK_B_SCHEMA = {
    "episode": 1,
    "story_summary": "",
    "opening_state": {
        "situation": "",
        "active_problem": "",
        "immediate_goal": "",
    },
    "character_states": [
        {
            "name": "",
            "role": "",
            "current_goal": "",
            "emotion": "",
            "knowledge": [],
            "relationships": [],
            "change_in_episode": "",
        }
    ],
    "active_plot_threads": [
        {
            "thread": "",
            "status": "opened",
            "evidence": "",
            "next_pressure": "",
        }
    ],
    "conflicts": [
        {
            "type": "",
            "characters": [],
            "cause": "",
            "development": "",
            "result": "",
        }
    ],
    "turning_points": [
        {
            "event": "",
            "before": "",
            "after": "",
            "why_it_matters": "",
        }
    ],
    "setups": [
        {
            "setup": "",
            "possible_payoff": "",
            "status": "",
        }
    ],
    "payoffs": [
        {
            "payoff": "",
            "setup_reference": "",
            "effect": "",
        }
    ],
    "continuity_constraints": [
        {
            "fact": "",
            "must_remain_true_until_changed": "",
            "risk_if_ignored": "",
        }
    ],
    "reveals_and_knowledge": [
        {
            "fact": "",
            "known_by": [],
            "unknown_to": [],
            "effect": "",
        }
    ],
    "cliffhanger": {
        "type": "",
        "question_created": "",
        "ending_event": "",
        "promised_next_pressure": "",
    },
    "next_episode_logic": {
        "must_continue": [],
        "likely_immediate_actions": [],
        "unresolved_questions": [],
        "do_not_do": [],
    },
    "timeline_delta": "",
    "locations": [],
    "objects_or_resources": [],
    "continuity_memory_update": [],
    "evidence": [],
}


def track_b_prompt(ep: int, clean_text: str, memory: Dict[str, Any]) -> str:
    return (
        f"Episode {ep} is a fictional Hindi drama.\n"
        "Extract ONLY facts supported by the transcript. Do not create the next episode, do not continue the story, and do not invent future canon.\n"
        "Return valid JSON that matches this schema exactly:\n"
        f"{json.dumps(TRACK_B_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
        f"PRIOR MEMORY:\n{json.dumps(memory, ensure_ascii=False)}\n\n"
        f"CLEAN EPISODE TEXT:\n{clean_text}\n\n"
        "Rules:\n"
        "- Use transcript-supported facts only.\n"
        "- If something is uncertain, keep it cautious or leave it empty.\n"
        "- Evidence should be short text snippets taken from the episode.\n"
    )


def ensure_track_b_shape(obj: Dict[str, Any], ep: int) -> Dict[str, Any]:
    obj["episode"] = ep
    for key in (
        "story_summary",
        "opening_state",
        "character_states",
        "active_plot_threads",
        "conflicts",
        "turning_points",
        "setups",
        "payoffs",
        "continuity_constraints",
        "reveals_and_knowledge",
        "cliffhanger",
        "next_episode_logic",
        "timeline_delta",
        "locations",
        "objects_or_resources",
        "continuity_memory_update",
        "evidence",
    ):
        obj.setdefault(key, TRACK_B_SCHEMA[key])

    return obj


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def rebuild_jsonl() -> Tuple[Path, Path]:
    track_a_jsonl = WORK / "track_a.jsonl"
    track_b_jsonl = WORK / "track_b.jsonl"

    with track_a_jsonl.open("w", encoding="utf-8") as fa, track_b_jsonl.open("w", encoding="utf-8") as fb:
        for ep in range(1, 201):
            a = CLEAN_DIR / f"Episode_{ep:04d}.txt"
            b = INTEL_DIR / f"Episode_{ep:04d}.json"
            if a.exists():
                fa.write(json.dumps({"episode": ep, "text": a.read_text(encoding="utf-8").strip()}, ensure_ascii=False) + "\n")
            if b.exists():
                fb.write(json.dumps(json.loads(b.read_text(encoding="utf-8")), ensure_ascii=False) + "\n")

    return track_a_jsonl, track_b_jsonl


def upload_batch(batch_episodes: List[int], memory: Dict[str, Any]) -> None:
    track_a_jsonl, track_b_jsonl = rebuild_jsonl()
    state_json = WORK / "story_memory.json"
    save_json(state_json, memory)

    files: Dict[str, Path] = {}
    for ep in batch_episodes:
        files[f"{TRACK_A_FOLDER}/Episode_{ep:04d}.txt"] = CLEAN_DIR / f"Episode_{ep:04d}.txt"
        files[f"{TRACK_B_FOLDER}/Episode_{ep:04d}.json"] = INTEL_DIR / f"Episode_{ep:04d}.json"

    files[f"{DATASETS_FOLDER}/track_a.jsonl"] = track_a_jsonl
    files[f"{DATASETS_FOLDER}/track_b.jsonl"] = track_b_jsonl
    files[f"{STATE_FOLDER}/story_memory.json"] = state_json

    ops = [
        CommitOperationAdd(path_in_repo=remote_path, path_or_fileobj=str(local_path))
        for remote_path, local_path in sorted(files.items())
    ]

    api.create_commit(
        repo_id=HF_REPO,
        repo_type=HF_TYPE,
        operations=ops,
        commit_message=f"Veda episodes {batch_episodes[0]:04d}-{batch_episodes[-1]:04d}",
        token=HF_TOKEN,
    )
    print(f"Uploaded batch {batch_episodes[0]:04d}-{batch_episodes[-1]:04d}")


def update_memory(memory: Dict[str, Any], intel: Dict[str, Any], ep: int) -> Dict[str, Any]:
    previous_threads = memory.get("active_plot_threads", [])
    previous_constraints = memory.get("continuity_constraints", [])
    previous_facts = memory.get("important_facts", [])

    new_threads = [x.get("thread", "") for x in intel.get("active_plot_threads", []) if isinstance(x, dict)]
    new_constraints = [x.get("fact", "") for x in intel.get("continuity_constraints", []) if isinstance(x, dict)]
    new_facts = [x.get("fact", "") for x in intel.get("reveals_and_knowledge", []) if isinstance(x, dict)]

    def unique_tail(items: List[str], limit: int) -> List[str]:
        return list(dict.fromkeys([i for i in items if i]))[-limit:]

    return {
        "last_completed_episode": ep,
        "story_summary": intel.get("story_summary", ""),
        "opening_state": intel.get("opening_state", {}),
        "active_plot_threads": unique_tail(previous_threads + new_threads, 50),
        "continuity_constraints": unique_tail(previous_constraints + new_constraints, 100),
        "important_facts": unique_tail(previous_facts + new_facts, 100),
        "latest_cliffhanger": intel.get("cliffhanger", {}),
    }


def main() -> None:
    source_paths = list_source_paths()
    completed = remote_completed()

    print(f"PASS: source episodes 1-200 found in {SOURCE_FOLDER}")
    print(f"Remote already complete: {len(completed)}/200")
    print(f"Cleaning model: {CLEAN_MODEL}")
    print(f"Analysis model: {ANALYSIS_MODEL}")

    memory: Dict[str, Any] = {}
    state_file = STATE_DIR / "story_memory.json"
    if state_file.exists():
        try:
            memory = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            memory = {}

    batch: List[int] = []
    prev_clean: Optional[str] = None

    for ep in range(1, 201):
        if ep in completed:
            print(f"[{ep:03d}/200] already uploaded, skipping")
            continue

        repo_path = source_paths[ep]
        raw_text = normalize_text(download_episode(repo_path))

        current_clean = raw_text
        if in_merged_range(ep):
            prev_text = prev_clean or ""
            next_text = ""
            if ep < 200 and (ep + 1) not in completed:
                next_text = normalize_text(download_episode(source_paths[ep + 1]))
            prompt = clean_prompt(ep, raw_text, prev_text, next_text)
            current_clean = safe_text(prompt, CLEAN_MODEL, max_tokens=5500, thinking=False)

        current_clean = normalize_text(current_clean)
        save_text(CLEAN_DIR / f"Episode_{ep:04d}.txt", current_clean)

        intel_prompt = track_b_prompt(ep, current_clean, memory)
        intel = safe_json(intel_prompt, ANALYSIS_MODEL, max_tokens=7000, thinking=True)
        intel = ensure_track_b_shape(intel, ep)
        save_json(INTEL_DIR / f"Episode_{ep:04d}.json", intel)

        memory = update_memory(memory, intel, ep)
        save_json(state_file, memory)

        batch.append(ep)
        prev_clean = current_clean

        print(f"DONE: Episode {ep:04d}")

        if len(batch) >= BATCH_SIZE:
            upload_batch(batch, memory)
            completed.update(batch)
            batch = []

        time.sleep(REQUEST_DELAY_SECONDS)

    if batch:
        upload_batch(batch, memory)

    print("ALL 200 EPISODES COMPLETE")


if __name__ == "__main__":
    main()
