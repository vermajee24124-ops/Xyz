#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

HF_REPO = "Kumarverma11/PocketFM_Audio"
HF_TYPE = "dataset"
SOURCE_FOLDER = "Transcripts_Episode_0001_to_0200"
EXPORT_FOLDER = "Veda_Final_Training_Export_0001_to_0200"

TRACK_A_FOLDER = f"{EXPORT_FOLDER}/TRACK_A_CLEAN_EPISODES"
TRACK_B_FOLDER = f"{EXPORT_FOLDER}/TRACK_B_STORY_INTELLIGENCE"
DATASETS_FOLDER = f"{EXPORT_FOLDER}/TRAINING_DATASETS"
STATE_FOLDER = f"{EXPORT_FOLDER}/STATE"

BATCH_SIZE = 20
REQUEST_DELAY_SECONDS = 1.5
MAX_RETRIES = 4

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODELS_URL = "https://integrate.api.nvidia.com/v1/models"

PREFERRED_CLEAN_MODELS = [
    "deepseek-ai/deepseek-v4-pro",
    "deepseek-ai/deepseek-v4-flash",
]
PREFERRED_ANALYSIS_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "deepseek-ai/deepseek-v4-pro",
    "deepseek-ai/deepseek-v4-flash",
]

MERGED_RANGES = (
    (111, 120),
    (121, 130),
    (133, 135),
    (138, 139),
    (140, 143),
)

WORK = Path("/tmp/veda_final_reference")
RAW_DIR = WORK / "raw"
CLEAN_DIR = WORK / "clean"
INTEL_DIR = WORK / "intel"
STATE_DIR = WORK / "state"
for p in (RAW_DIR, CLEAN_DIR, INTEL_DIR, STATE_DIR):
    p.mkdir(parents=True, exist_ok=True)


def _strip_invisible(s: str) -> str:
    s = s.replace("\ufeff", "")
    s = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", s)
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C").strip()


def secret(name: str) -> str:
    value = _strip_invisible(os.getenv(name, ""))
    if not value:
        raise RuntimeError(f"Missing GitHub secret: {name}")
    return value


HF_TOKEN = secret("HF_TOKEN")
NVIDIA_API_KEY = secret("NVIDIA_API_KEY")

api = HfApi(token=HF_TOKEN)
http = requests.Session()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([।!?])\1{1,}", r"\1", text)
    return text.strip()


def episode_no(path: str) -> Optional[int]:
    m = re.search(r"episode[_\s-]*0*(\d{1,4})", Path(path).name, re.I)
    return int(m.group(1)) if m else None


def in_merged_range(n: int) -> bool:
    return any(a <= n <= b for a, b in MERGED_RANGES)


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
        total += x / 1000.0 if unit == "ms" else x if unit == "s" else x * 60.0 if unit == "m" else x * 3600.0
    return max(1.0, total or float(default))


def list_source_paths() -> Dict[int, str]:
    prefix = f"{SOURCE_FOLDER}/"
    out: Dict[int, str] = {}
    for item in api.list_repo_tree(HF_REPO, repo_type=HF_TYPE, recursive=True):
        path = getattr(item, "path", "")
        if not (path.startswith(prefix) and path.lower().endswith(".txt")):
            continue
        ep = episode_no(path)
        if ep and 1 <= ep <= 200:
            if ep in out:
                raise RuntimeError(f"Duplicate source episode {ep}: {out[ep]} AND {path}")
            out[ep] = path
    missing = [n for n in range(1, 201) if n not in out]
    if missing:
        raise RuntimeError(f"Missing source episodes: {missing}")
    return out


def remote_completed() -> set[int]:
    done: set[int] = set()
    prefix = f"{TRACK_B_FOLDER}/"
    try:
        for item in api.list_repo_tree(HF_REPO, repo_type=HF_TYPE, recursive=True):
            path = getattr(item, "path", "")
            if not (path.startswith(prefix) and path.endswith(".json")):
                continue
            ep = episode_no(path)
            if ep:
                done.add(ep)
    except Exception:
        pass
    return done


def download_text(repo_path: str) -> str:
    local = hf_hub_download(HF_REPO, filename=repo_path, repo_type=HF_TYPE, token=HF_TOKEN, local_dir=str(RAW_DIR))
    return Path(local).read_text(encoding="utf-8", errors="replace")


def model_list() -> List[str]:
    r = http.get(NVIDIA_MODELS_URL, headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"}, timeout=(20, 60))
    r.raise_for_status()
    data = r.json()
    items = data.get("data", data if isinstance(data, list) else [])
    models = []
    for item in items:
        if isinstance(item, dict):
            mid = item.get("id") or item.get("name") or item.get("model")
            if isinstance(mid, str) and mid.strip():
                models.append(mid.strip())
        elif isinstance(item, str):
            models.append(item.strip())
    return sorted(set(models))


def preferred_available(preferred: List[str], available: List[str]) -> List[str]:
    amap = {m.lower(): m for m in available}
    chosen = [amap[p.lower()] for p in preferred if p.lower() in amap]
    if chosen:
        return list(dict.fromkeys(chosen))
    for p in preferred:
        ps = p.lower().split("/")[-1]
        for a in available:
            if a.lower().split("/")[-1] == ps:
                chosen.append(a)
    return list(dict.fromkeys(chosen))


def call_nvidia(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float = 0.1,
    thinking: bool = False,
    retries: int = MAX_RETRIES,
) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}

    def payload_variant(use_thinking: bool) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        ml = model.lower()
        if "nemotron-3-ultra" in ml and use_thinking:
            payload["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_budget": min(4096, max_tokens),
            }
        elif "deepseek-v4" in ml:
            payload["extra_body"] = {"chat_template_kwargs": {"thinking": bool(use_thinking)}}
        return payload

    variants = [payload_variant(thinking)]
    if thinking:
        variants.append(payload_variant(False))  # fallback if thinking params are not accepted

    last_error = ""
    for payload in variants:
        for attempt in range(1, retries + 1):
            try:
                r = http.post(NVIDIA_BASE_URL, headers=headers, json=payload, timeout=(30, 900))
                if r.status_code in (401, 403, 404):
                    raise PermissionError(f"HTTP {r.status_code}: {r.text[:1000]}")
                if r.status_code in (429, 500, 502, 503, 504):
                    wait = parse_wait(
                        r.headers.get("retry-after")
                        or r.headers.get("x-ratelimit-reset-requests")
                        or r.headers.get("x-ratelimit-reset-tokens"),
                        default=60,
                    )
                    wait += min(4.0, attempt)
                    print(f"{model}: HTTP {r.status_code}, retry in {wait:.1f}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
            except PermissionError:
                raise
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
        val = message.get(key)
        if isinstance(val, str) and val.strip():
            pieces.append(val.strip())
        elif isinstance(val, list):
            for part in val:
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
    dec = json.JSONDecoder()
    found = []
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[i:])
                if isinstance(obj, dict):
                    found.append(obj)
            except Exception:
                pass
    if found:
        return found[-1]
    raise ValueError("No valid JSON object found")


def call_chain(
    models: List[str],
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float = 0.1,
    thinking: bool = False,
    as_json: bool = False,
) -> str | Dict[str, Any]:
    last_error = ""
    for model in models:
        try:
            data = call_nvidia(model, messages, max_tokens=max_tokens, temperature=temperature, thinking=thinking)
            text = extract_message_text(data)
            if not text:
                raise RuntimeError(f"{model} returned empty response")
            return extract_json_object(text) if as_json else text.strip()
        except PermissionError as exc:
            last_error = str(exc)
            print(f"{model}: forbidden, switching")
        except Exception as exc:
            last_error = str(exc)
            print(f"{model}: failed: {exc}")
    raise RuntimeError(f"All models failed: {last_error}")


def clean_prompt(ep: int, current: str, prev_text: str, next_text: str) -> str:
    if in_merged_range(ep):
        boundary = (
            f"\nMerged-range repair mode is active for episode {ep}.\n"
            "Check only for overlap, duplicated lines, and sentences cut at the boundary.\n"
            "Do not rewrite the story, do not summarize, and do not add new dialogue.\n"
            f"PREVIOUS EPISODE END:\n{prev_text[-1500:]}\n"
            f"NEXT EPISODE START:\n{next_text[:1500]}\n"
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


def track_b_schema() -> Dict[str, Any]:
    return {
        "episode": 1,
        "story_summary": "",
        "opening_state": {"situation": "", "active_problem": "", "immediate_goal": ""},
        "character_states": [{"name": "", "role": "", "current_goal": "", "emotion": "", "knowledge": [], "relationships": [], "change_in_episode": ""}],
        "active_plot_threads": [{"thread": "", "status": "opened", "evidence": "", "next_pressure": ""}],
        "conflicts": [{"type": "", "characters": [], "cause": "", "development": "", "result": ""}],
        "turning_points": [{"event": "", "before": "", "after": "", "why_it_matters": ""}],
        "setups": [{"setup": "", "possible_payoff": "", "status": ""}],
        "payoffs": [{"payoff": "", "setup_reference": "", "effect": ""}],
        "continuity_constraints": [{"fact": "", "must_remain_true_until_changed": "", "risk_if_ignored": ""}],
        "reveals_and_knowledge": [{"fact": "", "known_by": [], "unknown_to": [], "effect": ""}],
        "cliffhanger": {"type": "", "question_created": "", "ending_event": "", "promised_next_pressure": ""},
        "next_episode_logic": {"must_continue": [], "likely_immediate_actions": [], "unresolved_questions": [], "do_not_do": []},
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
        f"{json.dumps(track_b_schema(), ensure_ascii=False, indent=2)}\n\n"
        f"PRIOR MEMORY:\n{json.dumps(memory, ensure_ascii=False)}\n\n"
        f"CLEAN EPISODE TEXT:\n{clean_text}\n\n"
        "Rules:\n"
        "- Use transcript-supported facts only.\n"
        "- If something is uncertain, keep it cautious or leave it empty.\n"
        "- Evidence should be short text snippets taken from the episode.\n"
        "- Do not write the next story.\n"
    )


def ensure_track_b_shape(obj: Dict[str, Any], ep: int) -> Dict[str, Any]:
    obj["episode"] = ep
    for key, default in track_b_schema().items():
        obj.setdefault(key, default)
    return obj


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def rebuild_jsonl() -> Tuple[Path, Path]:
    a = WORK / "track_a.jsonl"
    b = WORK / "track_b.jsonl"
    with a.open("w", encoding="utf-8") as fa, b.open("w", encoding="utf-8") as fb:
        for ep in range(1, 201):
            ta = CLEAN_DIR / f"Episode_{ep:04d}.txt"
            tb = INTEL_DIR / f"Episode_{ep:04d}.json"
            if ta.exists():
                fa.write(json.dumps({"episode": ep, "text": ta.read_text(encoding="utf-8").strip()}, ensure_ascii=False) + "\n")
            if tb.exists():
                fb.write(json.dumps(json.loads(tb.read_text(encoding="utf-8")), ensure_ascii=False) + "\n")
    return a, b


def upload_batch(batch_episodes: List[int], memory: Dict[str, Any]) -> None:
    a_jsonl, b_jsonl = rebuild_jsonl()
    state_json = WORK / "story_memory.json"
    save_json(state_json, memory)

    files: Dict[str, Path] = {}
    for ep in batch_episodes:
        files[f"{TRACK_A_FOLDER}/Episode_{ep:04d}.txt"] = CLEAN_DIR / f"Episode_{ep:04d}.txt"
        files[f"{TRACK_B_FOLDER}/Episode_{ep:04d}.json"] = INTEL_DIR / f"Episode_{ep:04d}.json"
    files[f"{DATASETS_FOLDER}/track_a.jsonl"] = a_jsonl
    files[f"{DATASETS_FOLDER}/track_b.jsonl"] = b_jsonl
    files[f"{STATE_FOLDER}/story_memory.json"] = state_json

    ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(l)) for r, l in sorted(files.items())]
    api.create_commit(
        repo_id=HF_REPO,
        repo_type=HF_TYPE,
        operations=ops,
        commit_message=f"Veda episodes {batch_episodes[0]:04d}-{batch_episodes[-1]:04d}",
        token=HF_TOKEN,
    )
    print(f"Uploaded batch {batch_episodes[0]:04d}-{batch_episodes[-1]:04d}")


def update_memory(memory: Dict[str, Any], intel: Dict[str, Any], ep: int) -> Dict[str, Any]:
    def uniq(items: List[str], limit: int) -> List[str]:
        return list(dict.fromkeys([x for x in items if x]))[-limit:]

    prev_threads = memory.get("active_plot_threads", [])
    prev_constraints = memory.get("continuity_constraints", [])
    prev_facts = memory.get("important_facts", [])

    new_threads = [x.get("thread", "") for x in intel.get("active_plot_threads", []) if isinstance(x, dict)]
    new_constraints = [x.get("fact", "") for x in intel.get("continuity_constraints", []) if isinstance(x, dict)]
    new_facts = [x.get("fact", "") for x in intel.get("reveals_and_knowledge", []) if isinstance(x, dict)]

    return {
        "last_completed_episode": ep,
        "story_summary": intel.get("story_summary", ""),
        "opening_state": intel.get("opening_state", {}),
        "active_plot_threads": uniq(prev_threads + new_threads, 50),
        "continuity_constraints": uniq(prev_constraints + new_constraints, 100),
        "important_facts": uniq(prev_facts + new_facts, 100),
        "latest_cliffhanger": intel.get("cliffhanger", {}),
    }


def main() -> None:
    source_paths = list_source_paths()
    completed = remote_completed()

    available = model_list()
    clean_models = preferred_available(PREFERRED_CLEAN_MODELS, available) or PREFERRED_CLEAN_MODELS[:]
    analysis_models = preferred_available(PREFERRED_ANALYSIS_MODELS, available) or PREFERRED_ANALYSIS_MODELS[:]

    print(f"PASS: source episodes 1-200 found in {SOURCE_FOLDER}")
    print(f"Remote already complete: {len(completed)}/200")
    print(f"Available NVIDIA models: {len(available)}")
    print(f"Clean models order: {clean_models}")
    print(f"Analysis models order: {analysis_models}")

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
        raw_text = normalize_text(download_text(repo_path))
        current_clean = raw_text

        if in_merged_range(ep):
            prev_text = prev_clean or ""
            next_text = ""
            if ep < 200 and (ep + 1) not in completed:
                next_text = normalize_text(download_text(source_paths[ep + 1]))
            current_clean = call_chain(
                clean_models,
                [
                    {"role": "system", "content": "You are a careful transcript editor for a fictional Hindi drama. Do not invent new events. Return only corrected transcript text."},
                    {"role": "user", "content": clean_prompt(ep, raw_text, prev_text, next_text)},
                ],
                max_tokens=5500,
                temperature=0.05,
                thinking=False,
                as_json=False,
            )
        else:
            current_clean = normalize_text(current_clean)

        current_clean = normalize_text(str(current_clean))
        save_text(CLEAN_DIR / f"Episode_{ep:04d}.txt", current_clean)

        intel = call_chain(
            analysis_models,
            [
                {"role": "system", "content": "You are a continuity analyst for a fictional Hindi drama. Use only transcript-supported facts. Return valid JSON only."},
                {"role": "user", "content": track_b_prompt(ep, current_clean, memory)},
            ],
            max_tokens=7000,
            temperature=0.1,
            thinking=True,
            as_json=True,
        )
        assert isinstance(intel, dict)
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
