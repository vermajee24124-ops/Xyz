#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError

# -----------------------------------------------------------------------------
# Veda pipeline
# Hindi drama transcript cleaning + story-intelligence extraction
# - Downloads source transcripts from Hugging Face
# - Cleans them (Track A)
# - Extracts structured story-intelligence JSON (Track B)
# - Uploads results back to Hugging Face in batches
# - Never writes transcript or JSON data into the GitHub repo itself
# -----------------------------------------------------------------------------

SOURCE_REPO_ID = os.environ.get("HF_SOURCE_REPO_ID", "Kumarverma11/PocketFM_Audio")
SOURCE_REPO_TYPE = os.environ.get("HF_SOURCE_REPO_TYPE", "dataset")
SOURCE_FOLDER = os.environ.get("HF_SOURCE_FOLDER", "Transcripts_Episode_0001_to_0200")

OUTPUT_REPO_ID = os.environ.get("HF_OUTPUT_REPO_ID", SOURCE_REPO_ID)
OUTPUT_REPO_TYPE = os.environ.get("HF_OUTPUT_REPO_TYPE", "dataset")
EXPORT_FOLDER = os.environ.get("HF_EXPORT_FOLDER", "Veda_Final_Training_Export_0001_to_0200")

TOTAL_EPISODES = int(os.environ.get("TOTAL_EPISODES", "200"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "20"))
MAX_EPISODES_THIS_RUN = int(os.environ.get("MAX_EPISODES_THIS_RUN", "40"))
PROVIDER_TIME_BUDGET_SECONDS = int(os.environ.get("PROVIDER_TIME_BUDGET_SECONDS", "300"))

GEMINI_MODEL_ID = os.environ.get("GEMINI_MODEL_ID", "gemini-3.1-flash-lite")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

CLEAN_MODEL_SHORTLIST = [
    "deepseek-ai/deepseek-v4-pro",
    "deepseek-ai/deepseek-v4-flash",
]

ANALYSIS_MODEL_SHORTLIST = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "deepseek-ai/deepseek-v4-pro",
    "deepseek-ai/deepseek-v4-flash",
]

OPENROUTER_MODEL_SHORTLIST = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-20b:free",
]

LOCAL_ROOT = Path("veda_work")
LOCAL_TRACK_A = LOCAL_ROOT / "TRACK_A_CLEAN_EPISODES"
LOCAL_TRACK_B = LOCAL_ROOT / "TRACK_B_STORY_INTELLIGENCE"
LOCAL_TRAINING = LOCAL_ROOT / "TRAINING_DATASETS"
LOCAL_STATE = LOCAL_ROOT / "STATE"

MERGED_RANGES = [(111, 120), (121, 130), (133, 135), (138, 139), (140, 143)]
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
  "character_states": [{"name": "", "role": "", "current_goal": "", "emotion": "", "knowledge": [], "relationships": [], "change_in_episode": ""}],
  "active_plot_threads": [],
  "conflicts": [],
  "turning_points": [],
  "setups": [],
  "payoffs": [],
  "continuity_constraints": [],
  "reveals_and_knowledge": [],
  "cliffhanger": {"type": "", "question_created": "", "ending_event": "", "promised_next_pressure": ""},
  "next_episode_logic": {"must_continue": [], "likely_immediate_actions": [], "unresolved_questions": [], "do_not_do": []},
  "timeline_delta": "",
  "locations": [],
  "objects_or_resources": [],
  "continuity_memory_update": [],
  "evidence": []
}"""

CLEAN_SYSTEM_PROMPT = (
    "You are a strict transcript-cleaning tool for a Hindi audio-drama script. "
    "Fix ASR mistakes, grammar, spelling, punctuation, spacing, and obvious name errors. "
    "Preserve story meaning, scene order, and episode identity exactly. "
    "Do not invent scenes, summarize, rewrite the story, continue the story, add new dialogue, "
    "or change chronology. Any text marked CONTEXT ONLY is background information from a neighboring "
    "episode used strictly to fix boundary overlap - never include it in your output. Return only the "
    "cleaned transcript text for the requested episode, with no preamble or explanation."
)

EXTRACT_SYSTEM_PROMPT = (
    "You are a strict story-analysis and extraction tool. You only extract facts explicitly supported "
    "by the given transcript. You never invent new story content, never continue the episode, never "
    "predict future canon as if it is fact, and never write the next episode or change the original plot. "
    "Respond with ONLY a single valid JSON object matching the given schema, with no prose before or after it."
)

_INVISIBLE_CHARS_RE = re.compile(r"[\u200B\u200E\u200F\u202A-\u202E\u2066-\u2069\uFEFF]")

def sanitize_secret(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = _INVISIBLE_CHARS_RE.sub("", value)
    value = "".join(ch for ch in value if ch in ("\n", "\t") or ord(ch) >= 32)
    return value.strip()

def load_secret(name: str) -> str:
    raw = os.environ.get(name)
    if not raw or not raw.strip():
        print(f"FATAL: required secret/env var '{name}' is missing or empty.", file=sys.stderr)
        sys.exit(1)
    cleaned = sanitize_secret(raw)
    if not cleaned:
        print(f"FATAL: secret/env var '{name}' became empty after sanitization.", file=sys.stderr)
        sys.exit(1)
    return cleaned

def load_optional_secret(name: str) -> Optional[str]:
    raw = os.environ.get(name)
    if not raw or not raw.strip():
        return None
    return sanitize_secret(raw)

HF_TOKEN = load_secret("HF_TOKEN")
NVIDIA_API_KEY = load_secret("NVIDIA_API_KEY")
GEMINI_API_KEYS = [
    key for key in (
        load_optional_secret("GEMINI_API_KEY_1"),
        load_optional_secret("GEMINI_API_KEY_2"),
        load_optional_secret("GEMINI_API_KEY_3"),
    ) if key
]
OPENROUTER_API_KEY = load_optional_secret("OPENROUTER_API_KEY")

hf_api = HfApi(token=HF_TOKEN)

def hf_call(func, *args, **kwargs):
    """Run a Hugging Face Hub call under a hard wall-clock timeout."""
    def _handler(signum, frame):
        raise TimeoutError(f"Hugging Face call '{func.__name__}' exceeded the hard timeout")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(90)
    try:
        return func(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def resolve_repo_type(repo_id: str, preferred_type: str) -> str:
    candidates = [preferred_type] + [t for t in ("dataset", "model") if t != preferred_type]
    last_exc = None
    for candidate in candidates:
        try:
            hf_call(hf_api.list_repo_files, repo_id=repo_id, repo_type=candidate)
            return candidate
        except RepositoryNotFoundError as exc:
            last_exc = exc
            continue
        except TimeoutError as exc:
            last_exc = exc
            continue
        except HfHubHTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 401:
                print(f"FATAL: Hugging Face rejected HF_TOKEN as unauthorized (401) while checking '{repo_id}'.", file=sys.stderr)
                sys.exit(1)
            last_exc = exc
            continue
    raise RuntimeError(
        f"Could not access repo '{repo_id}' as dataset or model with the given HF_TOKEN. "
        f"Check the repo ID spelling and token access. Last error: {last_exc}"
    )

class ModelUnavailableError(Exception):
    pass

class AllModelsFailedError(Exception):
    pass

def fetch_available_models() -> List[str]:
    url = f"{NVIDIA_BASE_URL}/models"
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Accept": "application/json"}
    resp = requests.get(url, headers=headers, timeout=(30, 60))
    resp.raise_for_status()
    data = resp.json()
    return [m.get("id", "") for m in data.get("data", []) if m.get("id")]

def select_model(preferred_list: List[str], available_models: List[str]) -> Optional[str]:
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

def build_payload(model_id: str, messages: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    model_lower = model_id.lower()
    if "nemotron" in model_lower and "ultra" in model_lower:
        return {
            "model": model_id,
            "messages": messages,
            "temperature": 1,
            "top_p": 0.95,
            "max_tokens": 16384 if mode == "extract" else 8192,
            "stream": False,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_budget": 4096,
            },
        }
    if "deepseek" in model_lower:
        return {
            "model": model_id,
            "messages": messages,
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 16384 if mode == "extract" else 8192,
            "stream": False,
            "extra_body": {
                "chat_template_kwargs": {"thinking": True if mode == "extract" else False},
            },
        }
    return {
        "model": model_id,
        "messages": messages,
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 4096,
        "stream": False,
    }

def call_nvidia_chat(model_id: str, messages: List[Dict[str, Any]], mode: str, deadline: float, max_retries: int = 3) -> str:
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
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            data = resp.json()
            choice = data["choices"][0]["message"]
            return choice.get("content") or ""

        if resp.status_code in (400, 401, 403, 404):
            snippet = resp.text[:300].replace("\n", " ")
            raise ModelUnavailableError(f"{model_id} returned {resp.status_code}: {snippet}")

        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == max_retries:
                raise ModelUnavailableError(f"{model_id} kept failing with {resp.status_code} after {max_retries} attempts")
            wait = min(attempt * 15, max(0, deadline - time.monotonic()))
            if wait <= 0:
                raise ModelUnavailableError(f"{model_id}: time budget exceeded after {resp.status_code}")
            time.sleep(wait)
            continue

        raise ModelUnavailableError(f"{model_id} returned unexpected status {resp.status_code}: {resp.text[:300]}")

    raise ModelUnavailableError(f"{model_id} failed after {max_retries} attempts")

def run_with_shortlist(preferred_list: List[str], available_models: List[str], messages: List[Dict[str, Any]], mode: str, deadline: float) -> Tuple[str, str]:
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
            print(f" Model unavailable, falling back: {exc}")
            tried.append(model_id)
            suffix = model_id.split("/")[-1]
            remaining = [m for m in remaining if m.split("/")[-1] != suffix]
            working_available = [m for m in working_available if m != model_id]
            continue
    raise AllModelsFailedError(f"All candidate models failed or unavailable: {tried}")

def call_openrouter_chat(model_id: str, messages: List[Dict[str, Any]], mode: str, deadline: float, max_retries: int = 3) -> str:
    if not OPENROUTER_API_KEY:
        raise ModelUnavailableError("no OPENROUTER_API_KEY configured")
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Veda Pipeline",
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 16384 if mode == "extract" else 8192,
        "stream": False,
        "reasoning": {"effort": "high"},
    }

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
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                raise ModelUnavailableError(f"{model_id}: no choices in response")
            content = choices[0].get("message", {}).get("content") or ""
            return content

        if resp.status_code in (400, 401, 403, 404):
            snippet = resp.text[:300].replace("\n", " ")
            raise ModelUnavailableError(f"{model_id} returned {resp.status_code}: {snippet}")

        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == max_retries:
                raise ModelUnavailableError(f"{model_id} kept failing with {resp.status_code} after {max_retries} attempts")
            wait = min(attempt * 15, max(0, deadline - time.monotonic()))
            if wait <= 0:
                raise ModelUnavailableError(f"{model_id}: time budget exceeded after {resp.status_code}")
            time.sleep(wait)
            continue

        raise ModelUnavailableError(f"{model_id} returned unexpected status {resp.status_code}: {resp.text[:300]}")

    raise ModelUnavailableError(f"{model_id} failed after {max_retries} attempts")

def run_openrouter_shortlist(messages: List[Dict[str, Any]], mode: str, deadline: float) -> Tuple[str, str]:
    if not OPENROUTER_API_KEY:
        raise AllModelsFailedError("no OPENROUTER_API_KEY configured")
    tried = []
    for model_id in OPENROUTER_MODEL_SHORTLIST:
        if time.monotonic() > deadline:
            tried.append("(stopped: time budget exceeded)")
            break
        try:
            result = call_openrouter_chat(model_id, messages, mode, deadline)
            return result, model_id
        except ModelUnavailableError as exc:
            print(f" OpenRouter model unavailable, falling back: {exc}")
            tried.append(model_id)
            continue
    raise AllModelsFailedError(f"All OpenRouter free models failed or unavailable: {tried}")

def to_gemini_payload(messages: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
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

def call_gemini_chat(messages: List[Dict[str, Any]], mode: str, deadline: float, max_retries_per_key: int = 2) -> str:
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
                text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
                if text.strip():
                    return text
                last_err = f"key #{key_index}: empty response text"
                break
            if resp.status_code in (400, 401, 403, 404):
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

def process_track(preferred_shortlist: List[str], available_models: List[str], messages: List[Dict[str, Any]], mode: str) -> Tuple[str, str]:
    deadline = time.monotonic() + PROVIDER_TIME_BUDGET_SECONDS
    nvidia_err = None
    openrouter_err = None
    try:
        return run_with_shortlist(preferred_shortlist, available_models, messages, mode, deadline)
    except AllModelsFailedError as exc:
        nvidia_err = exc
    try:
        print(f" NVIDIA models exhausted ({nvidia_err}). Trying OpenRouter free models.")
        return run_openrouter_shortlist(messages, mode, deadline)
    except AllModelsFailedError as exc:
        openrouter_err = exc
    if not GEMINI_API_KEYS:
        raise AllModelsFailedError(f"NVIDIA failed ({nvidia_err}) and OpenRouter failed ({openrouter_err}); no Gemini keys configured")
    print(f" OpenRouter also exhausted ({openrouter_err}). Falling back to Gemini backup ({GEMINI_MODEL_ID}).")
    try:
        text = call_gemini_chat(messages, mode, deadline)
        return text, GEMINI_MODEL_ID
    except ModelUnavailableError as gemini_exc:
        raise AllModelsFailedError(f"NVIDIA failed ({nvidia_err}), OpenRouter failed ({openrouter_err}), and Gemini backup also failed ({gemini_exc})")

def get_merged_range(ep: int) -> Optional[Tuple[int, int]]:
    for start, end in MERGED_RANGES:
        if start <= ep <= end:
            return (start, end)
    return None

def build_clean_messages(ep: int, raw_text: str, prev_tail: Optional[str], next_head: Optional[str]) -> List[Dict[str, str]]:
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

def build_extract_messages(ep: int, cleaned_text: str, memory_context: Any) -> List[Dict[str, str]]:
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
        + "Extract structured JSON using exactly this schema (fill every field; use empty string/list/object when nothing applies):\n"
        + TRACK_B_SCHEMA_HINT
    )
    return [
        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

def parse_track_b_json(raw_response: str, ep: int) -> Dict[str, Any]:
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

def check_source_completeness() -> None:
    try:
        files = hf_call(hf_api.list_repo_files, repo_id=SOURCE_REPO_ID, repo_type=SOURCE_REPO_TYPE)
    except TimeoutError as exc:
        print(f"FATAL: {exc} while listing source repo files. Check network/Hugging Face and retry.", file=sys.stderr)
        sys.exit(1)

    prefix = f"{SOURCE_FOLDER}/"
    pattern = re.compile(r"Episode_(\d{4})\.txt$")
    seen: Dict[int, List[str]] = {}

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

_raw_cache: Dict[int, str] = {}

def download_raw_episode(ep: int) -> str:
    if ep in _raw_cache:
        return _raw_cache[ep]
    filename = f"{SOURCE_FOLDER}/Episode_{ep:04d}.txt"
    local_path = hf_call(
        hf_hub_download,
        repo_id=SOURCE_REPO_ID,
        repo_type=SOURCE_REPO_TYPE,
        filename=filename,
        token=HF_TOKEN,
    )
    text = Path(local_path).read_text(encoding="utf-8", errors="replace")
    _raw_cache[ep] = text
    return text

def list_completed_episodes() -> set[int]:
    try:
        files = hf_call(hf_api.list_repo_files, repo_id=OUTPUT_REPO_ID, repo_type=OUTPUT_REPO_TYPE)
    except RepositoryNotFoundError:
        return set()
    except TimeoutError as exc:
        print(f"WARNING: {exc} while checking completed episodes; assuming none completed yet.")
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

def download_existing_state_and_jsonl() -> Dict[str, Any]:
    """Pull existing state so the run can resume and cumulative JSONL stays current."""
    memory: Dict[str, Any] = {}

    try:
        p = hf_call(
            hf_hub_download,
            repo_id=OUTPUT_REPO_ID,
            repo_type=OUTPUT_REPO_TYPE,
            filename=f"{EXPORT_FOLDER}/STATE/story_memory.json",
            token=HF_TOKEN,
        )
        memory = json.loads(Path(p).read_text(encoding="utf-8"))
        print("Loaded existing STATE/story_memory.json.")
    except (EntryNotFoundError, RepositoryNotFoundError):
        print("No existing story_memory.json found; starting fresh.")
    except TimeoutError as exc:
        print(f"WARNING: {exc} while loading story_memory.json; starting fresh.")
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status == 401:
            print(f"FATAL: HF_TOKEN unauthorized while reading STATE/story_memory.json.", file=sys.stderr)
            sys.exit(1)
        print(f"WARNING: could not load story_memory.json ({exc}); starting fresh.")

    for name in ("track_a.jsonl", "track_b.jsonl"):
        target = LOCAL_TRAINING / name
        try:
            p = hf_call(
                hf_hub_download,
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
        except TimeoutError as exc:
            target.write_text("", encoding="utf-8")
            print(f"WARNING: {exc} while loading {name}; starting fresh.")
        except HfHubHTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 401:
                print(f"FATAL: HF_TOKEN unauthorized while reading {name}.", file=sys.stderr)
                sys.exit(1)
            target.write_text("", encoding="utf-8")
            print(f"WARNING: could not load {name} ({exc}); starting fresh.")

    return memory

def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def upload_batch(up_to_episode: int) -> None:
    hf_call(
        hf_api.upload_folder,
        repo_id=OUTPUT_REPO_ID,
        repo_type=OUTPUT_REPO_TYPE,
        folder_path=str(LOCAL_ROOT),
        path_in_repo=EXPORT_FOLDER,
        token=HF_TOKEN,
        commit_message=f"Veda pipeline: batch update through episode {up_to_episode:04d}",
    )
    print(f"Uploaded batch through episode {up_to_episode:04d}.")

def main() -> None:
    global SOURCE_REPO_TYPE, OUTPUT_REPO_TYPE

    print("Veda pipeline starting.")
    print(f" Source: {SOURCE_REPO_ID} ({SOURCE_REPO_TYPE}) / {SOURCE_FOLDER}")
    print(f" Output: {OUTPUT_REPO_ID} ({OUTPUT_REPO_TYPE}) / {EXPORT_FOLDER}")
    print(f" Total episodes: {TOTAL_EPISODES}, batch size: {BATCH_SIZE}, cap this run: {MAX_EPISODES_THIS_RUN}")
    print(f" Per-track time budget: {PROVIDER_TIME_BUDGET_SECONDS}s (NVIDIA + Gemini backup combined)")
    print(f" Gemini backup keys configured: {len(GEMINI_API_KEYS)} (model: {GEMINI_MODEL_ID})")
    print(f" OpenRouter configured: {bool(OPENROUTER_API_KEY)} (shortlist: {OPENROUTER_MODEL_SHORTLIST})")

    SOURCE_REPO_TYPE = resolve_repo_type(SOURCE_REPO_ID, SOURCE_REPO_TYPE)
    if OUTPUT_REPO_ID == SOURCE_REPO_ID:
        OUTPUT_REPO_TYPE = SOURCE_REPO_TYPE
    else:
        try:
            OUTPUT_REPO_TYPE = resolve_repo_type(OUTPUT_REPO_ID, OUTPUT_REPO_TYPE)
        except RuntimeError:
            print(f"Output repo '{OUTPUT_REPO_ID}' doesn't exist yet; creating it as repo_type='{OUTPUT_REPO_TYPE}'.")
            try:
                hf_call(hf_api.create_repo, repo_id=OUTPUT_REPO_ID, repo_type=OUTPUT_REPO_TYPE, exist_ok=True)
            except TimeoutError as exc:
                print(f"WARNING: {exc}; assuming the output repo already exists and continuing.")
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
    failures: List[int] = []
    last_ep = 0

    for ep in range(1, TOTAL_EPISODES + 1):
        if ep in completed:
            continue
        if processed_this_run >= MAX_EPISODES_THIS_RUN:
            print(f"Reached MAX_EPISODES_THIS_RUN={MAX_EPISODES_THIS_RUN}; stopping cleanly for this run.")
            break

        print(f"--- Episode {ep} ---")
        episode_start = time.monotonic()

        try:
            raw_text = download_raw_episode(ep)
            prev_tail = next_head = None
            if get_merged_range(ep):
                if ep - 1 >= 1:
                    prev_tail = download_raw_episode(ep - 1)[-600:]
                if ep + 1 <= TOTAL_EPISODES:
                    next_head = download_raw_episode(ep + 1)[:600]

            clean_messages = build_clean_messages(ep, raw_text, prev_tail, next_head)
            cleaned_text, clean_model = process_track(CLEAN_MODEL_SHORTLIST, available_models, clean_messages, mode="clean")
            cleaned_text = cleaned_text.strip()
            if not cleaned_text:
                raise ValueError("Track A returned empty cleaned text")

            min_required = max(int(MIN_CLEAN_LENGTH_CHARS), int(len(raw_text) * MIN_CLEAN_LENGTH_RATIO))
            if len(cleaned_text) < min_required:
                raise ValueError(
                    f"Track A output looks truncated/garbage: got {len(cleaned_text)} chars, "
                    f"expected at least {min_required} chars (raw was {len(raw_text)} chars). "
                    f"Rejecting instead of uploading a broken episode."
                )

            (LOCAL_TRACK_A / f"Episode_{ep:04d}.txt").write_text(cleaned_text, encoding="utf-8")

            extract_messages = build_extract_messages(ep, cleaned_text, story_memory)
            raw_json_response, extract_model = process_track(ANALYSIS_MODEL_SHORTLIST, available_models, extract_messages, mode="extract")
            track_b_data = parse_track_b_json(raw_json_response, ep)
            if len(track_b_data.get("story_summary", "")) < 15:
                raise ValueError("Track B output looks garbage: story_summary is empty/too short")

            (LOCAL_TRACK_B / f"Episode_{ep:04d}.json").write_text(
                json.dumps(track_b_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            append_jsonl(LOCAL_TRAINING / "track_a.jsonl", {"episode": ep, "text": cleaned_text})
            append_jsonl(LOCAL_TRAINING / "track_b.jsonl", track_b_data)

            story_memory[str(ep)] = track_b_data.get("continuity_memory_update", [])
            (LOCAL_STATE / "story_memory.json").write_text(
                json.dumps(story_memory, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            elapsed = time.monotonic() - episode_start
            print(f"Episode {ep} done in {elapsed:.0f}s (clean={clean_model}, extract={extract_model}).")

            processed_since_upload += 1
            processed_this_run += 1
            last_ep = ep

        except AllModelsFailedError as exc:
            elapsed = time.monotonic() - episode_start
            print(f"Episode {ep} FAILED after {elapsed:.0f}s (all models exhausted): {exc}")
            failures.append(ep)
        except Exception as exc:
            elapsed = time.monotonic() - episode_start
            print(f"Episode {ep} FAILED after {elapsed:.0f}s (unexpected error): {exc}")
            failures.append(ep)

        if processed_since_upload >= BATCH_SIZE:
            try:
                print("Uploading batch to Hugging Face...")
                upload_batch(last_ep)
                processed_since_upload = 0
            except Exception as exc:
                print(f"WARNING: batch upload failed ({exc}); will retry after the next episode.")

        time.sleep(2)  # small delay between episodes to stay well under the rate limit

    if processed_since_upload > 0:
        try:
            print("Uploading final partial batch for this run...")
            upload_batch(last_ep)
        except Exception as exc:
            print(f"WARNING: final batch upload failed ({exc}). Already-written local files are lost with the runner, but every earlier per-episode upload this run already succeeded on Hugging Face.")

    print(f"Run finished. Processed {processed_this_run} episode(s) this run.")
    if failures:
        print(f"Episodes that failed this run (will retry automatically next run): {failures}")

if __name__ == "__main__":
    main()
