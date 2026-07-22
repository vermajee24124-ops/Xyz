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
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError

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

# Hard wall-clock budget (seconds) for a SINGLE track (clean OR extract) on a
# single episode, covering all NVIDIA retries/fallbacks plus the Gemini backup.
# This is the fix for the "stuck for 3-4 hours" problem: previously there was
# no ceiling, so retries + model fallbacks could silently multiply into hours.
# Worst case per episode is now roughly 2 x this value (clean + extract).
PROVIDER_TIME_BUDGET_SECONDS = int(os.environ.get("PROVIDER_TIME_BUDGET_SECONDS", "600"))

GEMINI_MODEL_ID = os.environ.get("GEMINI_MODEL_ID", "gemini-3.1-flash-lite")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

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


def load_optional_secret(name):
    """Like load_secret, but returns None instead of exiting - for secrets
    that are backups/optional rather than required to run at all."""
    raw = os.environ.get(name)
    if not raw or not raw.strip():
        return None
    return sanitize_secret(raw)


HF_TOKEN = load_secret("HF_TOKEN")
NVIDIA_API_KEY = load_secret("NVIDIA_API_KEY")

GEMINI_API_KEYS = [
    key
    for key in (
        load_optional_secret("GEMINI_API_KEY_1"),
        load_optional_secret("GEMINI_API_KEY_2"),
        load_optional_secret("GEMINI_API_KEY_3"),
    )
    if key
]

hf_api = HfApi(token=HF_TOKEN)


def resolve_repo_type(repo_id, preferred_type):
    """A repo can be type 'dataset' or 'model' on Hugging Face, and using the
    wrong one makes every call fail as if the repo doesn't exist. Try the
    configured type first, then the other one, so a wrong guess doesn't
    silently break the whole run."""
    candidates = [preferred_type] + [t for t in ("dataset", "model") if t != preferred_type]
    last_exc = None
    for candidate in candidates:
        try:
            hf_api.list_repo_files(repo_id=repo_id, repo_type=candidate)
            if candidate != preferred_type:
                print(f"NOTE: '{repo_id}' is actually repo_type='{candidate}', not '{preferred_type}'. Using '{candidate}'.")
            return candidate
        except RepositoryNotFoundError as exc:
            last_exc = exc
            continue
        except HfHubHTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 401:
                print(
                    f"FATAL: Hugging Face rejected HF_TOKEN as unauthorized (401) while checking '{repo_id}'. "
                    f"Check that the token is valid and not expired.",
                    file=sys.stderr,
                )
                sys.exit(1)
            last_exc = exc
            continue
    raise RuntimeError(
        f"Could not access repo '{repo_id}' as dataset or model with the given HF_TOKEN. "
        f"Check the repo ID spelling and that the token has access to it. Last error: {last_exc}"
    )


# --------------------------------------------------------------------------
# NVIDIA NIM: model discovery, model-specific payloads, robust calling
# --------------------------------------------------------------------------

class ModelUnavailableError(Exception):
    """This specific model can't serve the request; try the next one in the shortlist."""


class AllModelsFailedError(Exception):
    """Every candidate model in a shortlist failed."""


def fetch_available_models():
    url = f"{NVIDIA_BASE_URL}/models"
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Accept": "application/json"}
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
            "max_tokens": 16384 if mode == "extract" else 8192,
            "stream": False,
            "chat_template_kwargs": {"thinking": True},
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


def call_nvidia_chat(model_id, messages, mode, deadline, max_retries=3):
    url = f"{NVIDIA_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = build_payload(model_id, messages, mode)

    for attempt in range(1, max_retries + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ModelUnavailableError(f"{model_id}: time budget exceeded before attempt {attempt}")

        read_timeout = max(30, min(240, remaining))
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=(30, read_timeout))
        except requests.exceptions.RequestException as exc:
            if attempt == max_retries:
                raise ModelUnavailableError(f"{model_id} network error: {exc}") from exc
            wait = min(attempt * 10, max(0, deadline - time.monotonic()))
            if wait <= 0:
                raise ModelUnavailableError(f"{model_id}: time budget exceeded after network error")
            print(f"  [{model_id}] network error (attempt {attempt}/{max_retries}): {exc}. Retrying in {wait:.0f}s.")
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
            wait = min(attempt * 15, max(0, deadline - time.monotonic()))
            if wait <= 0:
                raise ModelUnavailableError(f"{model_id}: time budget exceeded after {resp.status_code}")
            print(f"  [{model_id}] got {resp.status_code} (attempt {attempt}/{max_retries}). Retrying in {wait:.0f}s.")
            time.sleep(wait)
            continue

        raise ModelUnavailableError(f"{model_id} returned unexpected status {resp.status_code}: {resp.text[:300]}")

    raise ModelUnavailableError(f"{model_id} failed after {max_retries} attempts")


def run_with_shortlist(preferred_list, available_models, messages, mode, deadline):
    """Try each preferred model in order; on 400/401/403/404 or repeated
    429/5xx, drop that model and fall through to the next one. Stops the
    moment the shared time budget runs out instead of trying forever."""
    remaining = list(preferred_list)
    working_available = list(available_models)
    tried = []

    while remaining:
        if time.monotonic() > deadline:
            tried.append("(stopped: time budget exceeded)")
            break
        model_id = select_model(remaining, working_available)
        if model_id is None:
            break
        try:
            result = call_nvidia_chat(model_id, messages, mode, deadline)
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
# Gemini: backup provider (used only after every NVIDIA model has failed)
# --------------------------------------------------------------------------

def to_gemini_payload(messages, mode):
    """Convert OpenAI-style {role, content} messages into Gemini's
    contents/systemInstruction shape, with thinking forced to HIGH."""
    system_text = None
    contents = []
    for msg in messages:
        if msg["role"] == "system":
            system_text = (system_text + "\n" if system_text else "") + msg["content"]
        else:
            gemini_role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": msg["content"]}]})

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 16384 if mode == "extract" else 8192,
            "thinkingConfig": {"thinkingLevel": "HIGH"},
        },
    }
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}
    return payload


def call_gemini_chat(messages, mode, deadline, max_retries_per_key=2):
    if not GEMINI_API_KEYS:
        raise ModelUnavailableError("no Gemini API keys configured")

    url = f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL_ID}:generateContent"
    payload = to_gemini_payload(messages, mode)
    last_err = "no keys tried"

    for key_index, api_key in enumerate(GEMINI_API_KEYS, start=1):
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

        for attempt in range(1, max_retries_per_key + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModelUnavailableError(f"gemini: time budget exceeded ({last_err})")

            read_timeout = max(30, min(180, remaining))
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=(30, read_timeout))
            except requests.exceptions.RequestException as exc:
                last_err = f"key #{key_index} network error: {exc}"
                wait = min(attempt * 10, max(0, deadline - time.monotonic()))
                if wait <= 0:
                    break
                time.sleep(wait)
                continue

            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    last_err = f"key #{key_index}: no candidates in response"
                    break
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts if "text" in p)
                if text.strip():
                    return text
                last_err = f"key #{key_index}: empty response text"
                break

            if resp.status_code in (400, 401, 403, 404):
                # Quota exhausted (403) or bad key - move to the next key, not worth retrying this one.
                last_err = f"key #{key_index} returned {resp.status_code}: {resp.text[:200]}"
                break

            if resp.status_code == 429 or resp.status_code >= 500:
                last_err = f"key #{key_index} returned {resp.status_code}: {resp.text[:200]}"
                wait = min(attempt * 15, max(0, deadline - time.monotonic()))
                if wait <= 0:
                    break
                time.sleep(wait)
                continue

            last_err = f"key #{key_index} unexpected status {resp.status_code}: {resp.text[:200]}"
            break

    raise ModelUnavailableError(f"all Gemini keys exhausted: {last_err}")


def process_track(nvidia_shortlist, available_models, messages, mode):
    """Try NVIDIA models (primary) within a time budget; only if every NVIDIA
    option is exhausted or the budget runs out, fall back to Gemini (backup)
    for whatever budget remains."""
    deadline = time.monotonic() + PROVIDER_TIME_BUDGET_SECONDS
    try:
        return run_with_shortlist(nvidia_shortlist, available_models, messages, mode, deadline)
    except AllModelsFailedError as nvidia_exc:
        if not GEMINI_API_KEYS:
            raise
        print(f"  NVIDIA models exhausted ({nvidia_exc}). Falling back to Gemini backup ({GEMINI_MODEL_ID}).")
        try:
            text = call_gemini_chat(messages, mode, deadline)
            return text, GEMINI_MODEL_ID
        except ModelUnavailableError as gemini_exc:
            raise AllModelsFailedError(
                f"NVIDIA failed ({nvidia_exc}) and Gemini backup also failed ({gemini_exc})"
            )


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
    except RepositoryNotFoundError:
        return set()
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status == 401:
            print(f"FATAL: HF_TOKEN unauthorized while reading '{OUTPUT_REPO_ID}'.", file=sys.stderr)
            sys.exit(1)
        print(f"WARNING: could not list files in '{OUTPUT_REPO_ID}' ({exc}); assuming nothing completed yet.")
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
    except (EntryNotFoundError, RepositoryNotFoundError):
        print("No existing story_memory.json found; starting fresh.")
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status == 401:
            print(f"FATAL: HF_TOKEN unauthorized while reading STATE/story_memory.json.", file=sys.stderr)
            sys.exit(1)
        print(f"WARNING: could not load story_memory.json ({exc}); starting fresh.")

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
        except (EntryNotFoundError, RepositoryNotFoundError):
            target.write_text("", encoding="utf-8")
            print(f"No existing {name} found; starting fresh.")
        except HfHubHTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 401:
                print(f"FATAL: HF_TOKEN unauthorized while reading {name}.", file=sys.stderr)
                sys.exit(1)
            target.write_text("", encoding="utf-8")
            print(f"WARNING: could not load {name} ({exc}); starting fresh.")

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
    global SOURCE_REPO_TYPE, OUTPUT_REPO_TYPE

    print("Veda pipeline starting.")
    print(f"  Source: {SOURCE_REPO_ID} ({SOURCE_REPO_TYPE}) / {SOURCE_FOLDER}")
    print(f"  Output: {OUTPUT_REPO_ID} ({OUTPUT_REPO_TYPE}) / {EXPORT_FOLDER}")
    print(f"  Total episodes: {TOTAL_EPISODES}, batch size: {BATCH_SIZE}, cap this run: {MAX_EPISODES_THIS_RUN}")
    print(f"  Per-track time budget: {PROVIDER_TIME_BUDGET_SECONDS}s (NVIDIA + Gemini backup combined)")
    print(f"  Gemini backup keys configured: {len(GEMINI_API_KEYS)} (model: {GEMINI_MODEL_ID})")

    # Confirm the source repo's real type (dataset vs model) before trusting the config default.
    SOURCE_REPO_TYPE = resolve_repo_type(SOURCE_REPO_ID, SOURCE_REPO_TYPE)

    # Resolve/create the output repo. If it's the same repo as the source, reuse the
    # type we already confirmed instead of re-querying.
    if OUTPUT_REPO_ID == SOURCE_REPO_ID:
        OUTPUT_REPO_TYPE = SOURCE_REPO_TYPE
    else:
        try:
            OUTPUT_REPO_TYPE = resolve_repo_type(OUTPUT_REPO_ID, OUTPUT_REPO_TYPE)
        except RuntimeError:
            print(f"Output repo '{OUTPUT_REPO_ID}' doesn't exist yet; creating it as repo_type='{OUTPUT_REPO_TYPE}'.")

    try:
        hf_api.create_repo(repo_id=OUTPUT_REPO_ID, repo_type=OUTPUT_REPO_TYPE, exist_ok=True)
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status == 401:
            print(
                f"FATAL: HF_TOKEN does not have write access to create/use '{OUTPUT_REPO_ID}'. "
                f"Generate a token with 'write' permission at huggingface.co/settings/tokens.",
                file=sys.stderr,
            )
            sys.exit(1)
        raise

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
            cleaned_text, clean_model = process_track(
                CLEAN_MODEL_SHORTLIST, available_models, clean_messages, mode="clean"
            )
            cleaned_text = cleaned_text.strip()
            if not cleaned_text:
                raise ValueError("Track A returned empty cleaned text")
            (LOCAL_TRACK_A / f"Episode_{ep:04d}.txt").write_text(cleaned_text, encoding="utf-8")

            extract_messages = build_extract_messages(ep, cleaned_text, story_memory)
            raw_json_response, extract_model = process_track(
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
