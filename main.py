"""
Veda pipeline: Hindi drama transcript cleaning + story-intelligence extraction.

Downloads source transcripts from Hugging Face, cleans them (Track A),
extracts structured story-intelligence JSON (Track B), and uploads the
results back to Hugging Face in batches of 20 episodes. Never writes
transcript or JSON data into the GitHub repo itself.

Required GitHub Secrets:
    HF_TOKEN         - Hugging Face access token (write access to output repo)
    NVIDIA_API_KEY   - NVIDIA build.nvidia.com API key

Optional environment variables (set in workflow.yml, safe to keep as plain
env vars since they are not secrets):
    HF_SOURCE_REPO_ID       (default: Kumarverma11/PocketFM_Audio)
    HF_SOURCE_REPO_TYPE     (default: dataset)
    HF_SOURCE_FOLDER        (default: Transcripts_Episode_0001_to_0200)
    HF_OUTPUT_REPO_ID       (default: same as HF_SOURCE_REPO_ID)
    HF_OUTPUT_REPO_TYPE     (default: dataset)
    HF_EXPORT_FOLDER        (default: Veda_Final_Training_Export_0001_to_0200)
    TOTAL_EPISODES          (default: 200)
    BATCH_SIZE              (default: 20)
    MAX_EPISODES_THIS_RUN   (default: 40) - safety cap so one GitHub Actions
                            job never runs long enough to hit the runner
                            timeout; the next scheduled/dispatched run picks
                            up where this one left off.
"""

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_download

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SOURCE_REPO_ID = os.environ.get("HF_SOURCE_REPO_ID", "Kumarverma11/PocketFM_Audio")
SOURCE_REPO_TYPE = os.environ.get("HF_SOURCE_REPO_TYPE", "dataset")
SOURCE_FOLDER = os.environ.get("HF_SOURCE_FOLDER", "Transcripts_Episode_0001_to_0200")

OUTPUT_REPO_ID = os.environ.get("HF_OUTPUT_REPO_ID", SOURCE_REPO_ID)
OUTPUT_REPO_TYPE = os.environ.get("HF_OUTPUT_REPO_TYPE", "dataset")
EXPORT_FOLDER = os.environ.get("HF_EXPORT_FOLDER", "Veda_Final_Training_Export_0001_to_0200")

TOTAL_EPISODES = int(os.environ.get("TOTAL_EPISODES", "200"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "20"))
MAX_EPISODES_THIS_RUN = int(os.environ.get("MAX_EPISODES_THIS_RUN", "40"))

MERGED_RANGES = [(111, 120), (121, 130), (133, 135), (138, 139), (140, 143)]

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Preferred model shortlists, in priority order. Suffix-matched against
# whatever /v1/models actually reports for this key if an exact ID isn't found.
CLEAN_MODEL_SHORTLIST = [
    "deepseek-ai/deepseek-v4-pro",
    "deepseek-ai/deepseek-v4-flash",
]
ANALYSIS_MODEL_SHORTLIST = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "deepseek-ai/deepseek-v4-pro",
    "deepseek-ai/deepseek-v4-flash",
]

LOCAL_ROOT = Path("veda_work")
LOCAL_TRACK_A = LOCAL_ROOT / "TRACK_A_CLEAN_EPISODES"
LOCAL_TRACK_B = LOCAL_ROOT / "TRACK_B_STORY_INTELLIGENCE"
LOCAL_TRAINING = LOCAL_ROOT / "TRAINING_DATASETS"
LOCAL_STATE = LOCAL_ROOT / "STATE"

REQUIRED_TRACK_B_KEYS = [
    "episode", "story_summary", "opening_state", "character_states",
    "active_plot_threads", "conflicts", "turning_points", "setups", "payoffs",
    "continuity_constraints", "reveals_and_knowledge", "cliffhanger",
    "next_episode_logic", "timeline_delta", "locations", "objects_or_resources",
    "continuity_memory_update", "evidence",
]

TRACK_B_SCHEMA_HINT = """{
  "episode": 0,
  "story_summary": "",
  "opening_state": {"situation": "", "active_problem": "", "immediate_goal": ""},
  "character_states": [{"name": "", "role": "", "current_goal": "", "emotion": "",
                         "knowledge": [], "relationships": [], "change_in_episode": ""}],
  "active_plot_threads": [],
  "conflicts": [],
  "turning_points": [],
  "setups": [],
  "payoffs": [],
  "continuity_constraints": [],
  "reveals_and_knowledge": [],
  "cliffhanger": {"type": "", "question_created": "", "ending_event": "",
                   "promised_next_pressure": ""},
  "next_episode_logic": {"must_continue": [], "likely_immediate_actions": [],
                          "unresolved_questions": [], "do_not_do": []},
  "timeline_delta": "",
  "locations": [],
  "objects_or_resources": [],
  "continuity_memory_update": [],
  "evidence": []
}"""

CLEAN_SYSTEM_PROMPT = (
    "You are a strict transcript-cleaning tool for a Hindi audio-drama script. "
    "Fix ASR mistakes, grammar, spelling, punctuation, spacing, and obvious name "
    "errors. Preserve story meaning, scene order, and episode identity exactly. "
    "Do not invent scenes, summarize, rewrite the story, continue the story, add "
    "new dialogue, or change chronology. Any text marked CONTEXT ONLY is background "
    "information from a neighboring episode used strictly to fix boundary overlap - "
    "never include it in your output. Return only the cleaned transcript text for "
    "the requested episode, with no preamble or explanation."
)

EXTRACT_SYSTEM_PROMPT = (
    "You are a strict story-analysis and extraction tool. You only extract facts "
    "explicitly supported by the given transcript. You never invent new story "
    "content, never continue the episode, never predict future canon as if it is "
    "fact, and never write the next episode or change the original plot. Respond "
    "with ONLY a single valid JSON object matching the given schema, with no prose "
    "before or after it."
)


# --------------------------------------------------------------------------
# Secret sanitization
# --------------------------------------------------------------------------

_INVISIBLE_CHARS_RE = re.compile(
    "[\u200B\u200E\u200F\u202A-\u202E\u2066-\u2069\uFEFF]"
)


def sanitize_secret(value):
    """Strip invisible Unicode direction marks and control characters."""
    if value is None:
        return None
    value = _INVISIBLE_CHARS_RE.sub("", value)
    value = "".join(ch for ch in value if ch in ("\n", "\t") or ord(ch) >= 32)
    return value.strip()


def load_secret(name):
    raw = os.environ.get(name)
    if not raw or not raw.strip():
        print(f"FATAL: required secret/env var '{name}' is missing or empty.", file=sys.stderr)
        sys.exit(1)
    return sanitize_secret(raw)


HF_TOKEN = load_secret("HF_TOKEN")
NVIDIA_API_KEY = load_secret("NVIDIA_API_KEY")

hf_api = HfApi(token=HF_TOKEN)


# --------------------------------------------------------------------------
# NVIDIA NIM: model discovery, model-specific payloads, robust calling
# --------------------------------------------------------------------------

class ModelUnavailableError(Exception):
    """This specific model can't serve the request; try the next one in the shortlist."""


class AllModelsFailedError(Exception):
    """Every candidate model in a shortlist failed."""


def fetch_available_models():
    url = f"{NVIDIA_BASE_URL}/models"
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}"}
    resp = requests.get(url, headers=headers, timeout=(30, 60))
    resp.raise_for_status()
    data = resp.json()
    return [m.get("id", "") for m in data.get("data", []) if m.get("id")]


def select_model(preferred_list, available_models):
    """Exact match first, then suffix match against whatever the account can see."""
    available_set = set(available_models)
    for candidate in preferred_list:
        if candidate in available_set:
            return candidate
    for candidate in preferred_list:
        suffix = candidate.split("/")[-1]
        for avail in available_models:
            if avail.split("/")[-1] == suffix or avail.endswith(suffix):
                return avail
    return None


def build_payload(model_id, messages, mode):
    """Different model families need different payload shapes. Never send the
    same generic body to every provider - that mismatch is the classic cause
    of repeated 400 errors."""
    model_lower = model_id.lower()

    if "nemotron" in model_lower and "ultra" in model_lower:
        return {
            "model": model_id,
            "messages": messages,
            "temperature": 1,
            "top_p": 0.95,
            "max_tokens": 16384 if mode == "extract" else 8192,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_budget": 4096,
        }

    if "deepseek" in model_lower:
        return {
            "model": model_id,
            "messages": messages,
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 8192 if mode == "extract" else 4096,
            "stream": False,
            "chat_template_kwargs": {"thinking": False},
        }

    # Unknown model family: minimal, conservative payload with no extra_body-style
    # extensions, since we don't know what this model accepts.
    return {
        "model": model_id,
        "messages": messages,
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 4096,
        "stream": False,
    }


def call_nvidia_chat(model_id, messages, mode, max_retries=4):
    url = f"{NVIDIA_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = build_payload(model_id, messages, mode)

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=(30, 900))
        except requests.exceptions.RequestException as exc:
            if attempt == max_retries:
                raise ModelUnavailableError(f"{model_id} network error: {exc}") from exc
            wait = attempt * 10
            print(f"  [{model_id}] network error (attempt {attempt}/{max_retries}): {exc}. Retrying in {wait}s.")
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            data = resp.json()
            choice = data["choices"][0]["message"]
            return choice.get("content") or ""

        if resp.status_code in (400, 401, 403, 404):
            # Not retryable for this model - payload/model mismatch or access issue.
            snippet = resp.text[:300].replace("\n", " ")
            raise ModelUnavailableError(f"{model_id} returned {resp.status_code}: {snippet}")

        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == max_retries:
                raise ModelUnavailableError(
                    f"{model_id} kept failing with {resp.status_code} after {max_retries} attempts"
                )
            wait = attempt * 15
            print(f"  [{model_id}] got {resp.status_code} (attempt {attempt}/{max_retries}). Retrying in {wait}s.")
            time.sleep(wait)
            continue

        raise ModelUnavailableError(f"{model_id} returned unexpected status {resp.status_code}: {resp.text[:300]}")

    raise ModelUnavailableError(f"{model_id} failed after {max_retries} attempts")


def run_with_shortlist(preferred_list, available_models, messages, mode):
    """Try each preferred model in order; on 400/401/403/404 or repeated
    429/5xx, drop that model and fall through to the next one."""
    remaining = list(preferred_list)
    working_available = list(available_models)
    tried = []

    while remaining:
        model_id = select_model(remaining, working_available)
        if model_id is None:
            break
        try:
            result = call_nvidia_chat(model_id, messages, mode)
            return result, model_id
        except ModelUnavailableError as exc:
            print(f"  Model unavailable, falling back: {exc}")
            tried.append(model_id)
            suffix = model_id.split("/")[-1]
            remaining = [m for m in remaining if m.split("/")[-1] != suffix]
            working_available = [m for m in working_available if m != model_id]
            continue

    raise AllModelsFailedError(f"All candidate models failed or unavailable: {tried}")


# --------------------------------------------------------------------------
# Boundary repair ranges
# --------------------------------------------------------------------------

def get_merged_range(ep):
    for start, end in MERGED_RANGES:
        if start <= ep <= end:
            return (start, end)
    return None


# --------------------------------------------------------------------------
# Track A: cleaning
# --------------------------------------------------------------------------

def build_clean_messages(ep, raw_text, prev_tail, next_head):
    parts = []
    if prev_tail:
        parts.append("CONTEXT ONLY - end of previous episode (do not include in output):\n" + prev_tail)
    parts.append(f"Episode {ep} raw transcript to clean:\n{raw_text}")
    if next_head:
        parts.append("CONTEXT ONLY - start of next episode (do not include in output):\n" + next_head)
    user_content = "\n\n---\n\n".join(parts)
    return [
        {"role": "system", "content": CLEAN_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# --------------------------------------------------------------------------
# Track B: extraction
# --------------------------------------------------------------------------

def build_extract_messages(ep, cleaned_text, memory_context):
    memory_note = ""
    if memory_context:
        memory_note = (
            "Continuity memory from previous episodes (background context only):\n"
            + json.dumps(memory_context, ensure_ascii=False)
            + "\n\n"
        )
    user_content = (
        memory_note
        + f"Episode number: {ep}\n\n"
        + f"Cleaned transcript:\n{cleaned_text}\n\n"
        + "Extract structured JSON using exactly this schema "
        + "(fill every field; use empty string/list/object when nothing applies):\n"
        + TRACK_B_SCHEMA_HINT
    )
    return [
        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_track_b_json(raw_response, ep):
    text = (raw_response or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"episode {ep}: no JSON object found in Track B response")
    data = json.loads(match.group(0))
    missing = [k for k in REQUIRED_TRACK_B_KEYS if k not in data]
    if missing:
        raise ValueError(f"episode {ep}: Track B JSON missing keys: {missing}")
    data["episode"] = ep
    return data


# --------------------------------------------------------------------------
# Hugging Face: source verification, download, resume, upload
# --------------------------------------------------------------------------

def check_source_completeness():
    files = hf_api.list_repo_files(repo_id=SOURCE_REPO_ID, repo_type=SOURCE_REPO_TYPE)
    prefix = f"{SOURCE_FOLDER}/"
    pattern = re.compile(r"Episode_(\d{4})\.txt$")
    seen = {}
    for f in files:
        if not f.startswith(prefix):
            continue
        m = pattern.search(f)
        if not m:
            continue
        seen.setdefault(int(m.group(1)), []).append(f)

    missing = [ep for ep in range(1, TOTAL_EPISODES + 1) if ep not in seen]
    duplicated = {ep: paths for ep, paths in seen.items() if len(paths) > 1}

    if missing:
        print(f"FATAL: missing source episodes: {missing}", file=sys.stderr)
        sys.exit(1)
    if duplicated:
        print(f"FATAL: duplicated source episodes: {duplicated}", file=sys.stderr)
        sys.exit(1)
    print(f"Source check OK: all {TOTAL_EPISODES} episodes present, no duplicates.")


_raw_cache = {}


def download_raw_episode(ep):
    if ep in _raw_cache:
        return _raw_cache[ep]
    filename = f"{SOURCE_FOLDER}/Episode_{ep:04d}.txt"
    local_path = hf_hub_download(
        repo_id=SOURCE_REPO_ID,
        repo_type=SOURCE_REPO_TYPE,
        filename=filename,
        token=HF_TOKEN,
    )
    text = Path(local_path).read_text(encoding="utf-8", errors="replace")
    _raw_cache[ep] = text
    return text


def list_completed_episodes():
    try:
        files = hf_api.list_repo_files(repo_id=OUTPUT_REPO_ID, repo_type=OUTPUT_REPO_TYPE)
    except Exception:
        return set()

    a_prefix = f"{EXPORT_FOLDER}/TRACK_A_CLEAN_EPISODES/"
    b_prefix = f"{EXPORT_FOLDER}/TRACK_B_STORY_INTELLIGENCE/"
    pattern = re.compile(r"Episode_(\d{4})\.(txt|json)$")
    a_done, b_done = set(), set()

    for f in files:
        m = pattern.search(f)
        if not m:
            continue
        ep_num = int(m.group(1))
        if f.startswith(a_prefix):
            a_done.add(ep_num)
        elif f.startswith(b_prefix):
            b_done.add(ep_num)

    return a_done & b_done


def download_existing_state_and_jsonl():
    """Pull whatever already exists on Hugging Face so this run can resume
    and keep the training JSONL files cumulative across runs."""
    memory = {}
    try:
        p = hf_hub_download(
            repo_id=OUTPUT_REPO_ID,
            repo_type=OUTPUT_REPO_TYPE,
            filename=f"{EXPORT_FOLDER}/STATE/story_memory.json",
            token=HF_TOKEN,
        )
        memory = json.loads(Path(p).read_text(encoding="utf-8"))
        print("Loaded existing STATE/story_memory.json.")
    except Exception:
        print("No existing story_memory.json found; starting fresh.")

    for name in ("track_a.jsonl", "track_b.jsonl"):
        target = LOCAL_TRAINING / name
        try:
            p = hf_hub_download(
                repo_id=OUTPUT_REPO_ID,
                repo_type=OUTPUT_REPO_TYPE,
                filename=f"{EXPORT_FOLDER}/TRAINING_DATASETS/{name}",
                token=HF_TOKEN,
            )
            shutil.copyfile(p, target)
            print(f"Loaded existing {name}.")
        except Exception:
            target.write_text("", encoding="utf-8")
            print(f"No existing {name} found; starting fresh.")

    return memory


def upload_batch(up_to_episode):
    hf_api.upload_folder(
        repo_id=OUTPUT_REPO_ID,
        repo_type=OUTPUT_REPO_TYPE,
        folder_path=str(LOCAL_ROOT),
        path_in_repo=EXPORT_FOLDER,
        token=HF_TOKEN,
        commit_message=f"Veda pipeline: batch update through episode {up_to_episode:04d}",
    )
    print(f"Uploaded batch through episode {up_to_episode:04d}.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    print("Veda pipeline starting.")
    print(f"  Source: {SOURCE_REPO_ID} ({SOURCE_REPO_TYPE}) / {SOURCE_FOLDER}")
    print(f"  Output: {OUTPUT_REPO_ID} ({OUTPUT_REPO_TYPE}) / {EXPORT_FOLDER}")
    print(f"  Total episodes: {TOTAL_EPISODES}, batch size: {BATCH_SIZE}, cap this run: {MAX_EPISODES_THIS_RUN}")

    check_source_completeness()

    for d in (LOCAL_TRACK_A, LOCAL_TRACK_B, LOCAL_TRAINING, LOCAL_STATE):
        d.mkdir(parents=True, exist_ok=True)

    completed = list_completed_episodes()
    print(f"Already complete on Hugging Face: {len(completed)} episodes.")

    story_memory = download_existing_state_and_jsonl()

    available_models = fetch_available_models()
    print(f"NVIDIA account reports {len(available_models)} available models.")

    processed_since_upload = 0
    processed_this_run = 0
    failures = []
    last_ep = 0

    for ep in range(1, TOTAL_EPISODES + 1):
        if ep in completed:
            continue
        if processed_this_run >= MAX_EPISODES_THIS_RUN:
            print(f"Reached MAX_EPISODES_THIS_RUN={MAX_EPISODES_THIS_RUN}; stopping cleanly for this run.")
            break

        print(f"--- Episode {ep} ---")
        try:
            raw_text = download_raw_episode(ep)

            prev_tail = next_head = None
            if get_merged_range(ep):
                if ep - 1 >= 1:
                    prev_tail = download_raw_episode(ep - 1)[-600:]
                if ep + 1 <= TOTAL_EPISODES:
                    next_head = download_raw_episode(ep + 1)[:600]

            clean_messages = build_clean_messages(ep, raw_text, prev_tail, next_head)
            cleaned_text, clean_model = run_with_shortlist(
                CLEAN_MODEL_SHORTLIST, available_models, clean_messages, mode="clean"
            )
            cleaned_text = cleaned_text.strip()
            if not cleaned_text:
                raise ValueError("Track A returned empty cleaned text")
            (LOCAL_TRACK_A / f"Episode_{ep:04d}.txt").write_text(cleaned_text, encoding="utf-8")

            extract_messages = build_extract_messages(ep, cleaned_text, story_memory)
            raw_json_response, extract_model = run_with_shortlist(
                ANALYSIS_MODEL_SHORTLIST, available_models, extract_messages, mode="extract"
            )
            track_b_data = parse_track_b_json(raw_json_response, ep)
            (LOCAL_TRACK_B / f"Episode_{ep:04d}.json").write_text(
                json.dumps(track_b_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            with open(LOCAL_TRAINING / "track_a.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"episode": ep, "text": cleaned_text}, ensure_ascii=False) + "\n")
            with open(LOCAL_TRAINING / "track_b.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(track_b_data, ensure_ascii=False) + "\n")

            story_memory[str(ep)] = track_b_data.get("continuity_memory_update", [])
            (LOCAL_STATE / "story_memory.json").write_text(
                json.dumps(story_memory, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            print(f"Episode {ep} done (clean={clean_model}, extract={extract_model}).")
            processed_since_upload += 1
            processed_this_run += 1
            last_ep = ep

        except AllModelsFailedError as exc:
            print(f"Episode {ep} FAILED (all models exhausted): {exc}")
            failures.append(ep)
        except Exception as exc:
            print(f"Episode {ep} FAILED (unexpected error): {exc}")
            failures.append(ep)

        if processed_since_upload >= BATCH_SIZE:
            print("Uploading batch to Hugging Face...")
            upload_batch(last_ep)
            processed_since_upload = 0

        time.sleep(2)  # small delay between episodes to stay well under the rate limit

    if processed_since_upload > 0:
        print("Uploading final partial batch for this run...")
        upload_batch(last_ep)

    print(f"Run finished. Processed {processed_this_run} episode(s) this run.")
    if failures:
        print(f"Episodes that failed this run (will retry automatically next run): {failures}")


if __name__ == "__main__":
    main()
