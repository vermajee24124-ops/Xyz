import os
import sys
import json
import re
import time
import shutil
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
import google.generativeai as genai
from openai import OpenAI

# Line buffering setup for real-time live logs in GitHub Actions
sys.stdout.reconfigure(line_buffering=True)

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
SOURCE_REPO_ID = "Kumarverma11/PocketFM_Audio"
SOURCE_FOLDER = "Transcripts_Episode_0001_to_0200"

NEW_OUTPUT_REPO_ID = "Kumarverma11/VEDA_AI_Cleaned_Series"
EXPORT_FOLDER = "Veda_Final_Training_Export_0001_to_0200"
TOTAL_EPISODES = 200

UPLOAD_EVERY_EPISODES = 1  # Instant upload after each episode
GEMINI_MODEL_ID = "gemini-3.1-flash-lite"

print("="*65, flush=True)
print("🌟 VEDA AI MASTER PIPELINE ENGINE (PRODUCTION READY) 🌟", flush=True)
print("="*65, flush=True)
print(f"📍 SOURCE REPO   : {SOURCE_REPO_ID}/{SOURCE_FOLDER}", flush=True)
print(f"📍 TARGET REPO   : {NEW_OUTPUT_REPO_ID}", flush=True)
print(f"📂 EXPORT FOLDER : {EXPORT_FOLDER}/", flush=True)
print("="*65 + "\n", flush=True)

# ==========================================
# 2. SECRETS & CLIENT INITIALIZATION
# ==========================================
HF_TOKEN = os.environ.get("HF_TOKEN")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")

# Collect 3 Gemini Keys from environment
GEMINI_KEYS = [
    k for k in [
        os.environ.get("GEMINI_KEY_1"),
        os.environ.get("GEMINI_KEY_2"),
        os.environ.get("GEMINI_KEY_3")
    ] if k and k.strip()
]

print(f"🔑 Loaded {len(GEMINI_KEYS)} Gemini API Key(s).", flush=True)

if not HF_TOKEN:
    print("❌ FATAL ERROR: HF_TOKEN missing in Environment Secrets!", flush=True)
    sys.exit(1)

hf_api = HfApi(token=HF_TOKEN)

# Initialize Ollama Client
ollama_client = None
if OLLAMA_API_KEY and OLLAMA_API_KEY.strip():
    try:
        ollama_client = OpenAI(
            base_url="https://ollama.com/v1",
            api_key=OLLAMA_API_KEY.strip()
        )
        print("✅ Ollama Cloud Client Initialized.", flush=True)
    except Exception as e:
        print(f"⚠️ Ollama Client init note: {e}", flush=True)

gemini_safety_config = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

# Local Workspace Directories
LOCAL_ROOT = Path("veda_work")
LOCAL_TRACK_A = LOCAL_ROOT / "TRACK_A_CLEAN_EPISODES"
LOCAL_TRACK_B = LOCAL_ROOT / "TRACK_B_STORY_INTELLIGENCE"
LOCAL_TRAINING = LOCAL_ROOT / "TRAINING_DATASETS"
LOCAL_STATE = LOCAL_ROOT / "STATE"

for d in [LOCAL_TRACK_A, LOCAL_TRACK_B, LOCAL_TRAINING, LOCAL_STATE]:
    d.mkdir(parents=True, exist_ok=True)

# Ensure Target Repo Exists
try:
    hf_api.create_repo(repo_id=NEW_OUTPUT_REPO_ID, repo_type="dataset", exist_ok=True)
except Exception as e:
    print(f"ℹ️ Output repository ready: {e}", flush=True)

# ==========================================
# 3. SAFE AI CALLERS (CRASH-PROOF)
# ==========================================
def safe_extract_gemini_response(res):
    """Prevents crash when response.text is blocked or candidates are empty."""
    try:
        if res and hasattr(res, "candidates") and res.candidates and len(res.candidates) > 0:
            candidate = res.candidates[0]
            if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
                parts_text = "".join([p.text for p in candidate.content.parts if hasattr(p, "text") and p.text])
                if parts_text.strip():
                    return parts_text.strip()
    except Exception:
        pass
    return None

def call_gemini_with_fallback(prompt_text):
    """Rotates through 3 Gemini Keys."""
    for idx, key in enumerate(GEMINI_KEYS, start=1):
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(GEMINI_MODEL_ID, safety_settings=gemini_safety_config)
            res = model.generate_content(prompt_text)
            extracted_text = safe_extract_gemini_response(res)
            if extracted_text and len(extracted_text) > 20:
                print(f"   ✅ Success using Gemini Key #{idx}", flush=True)
                return extracted_text
            else:
                print(f"   ⚠️ Gemini Key #{idx} returned empty/blocked response.", flush=True)
        except Exception as e:
            print(f"   ⚠️ Gemini Key #{idx} Error ({e}). Trying next key...", flush=True)
    return None

# ==========================================
# 4. RESUME LOGIC & MEMORY LOADER
# ==========================================
def get_completed_episodes():
    try:
        files = hf_api.list_repo_files(repo_id=NEW_OUTPUT_REPO_ID, repo_type="dataset")
    except Exception:
        return set()
    
    a_done, b_done = set(), set()
    pattern = re.compile(r"Episode_(\d{4})\.(txt|json)$")
    a_prefix = f"{EXPORT_FOLDER}/TRACK_A_CLEAN_EPISODES/"
    b_prefix = f"{EXPORT_FOLDER}/TRACK_B_STORY_INTELLIGENCE/"

    for f in files:
        m = pattern.search(f)
        if not m: continue
        ep_num = int(m.group(1))
        if f.startswith(a_prefix): a_done.add(ep_num)
        elif f.startswith(b_prefix): b_done.add(ep_num)
            
    return a_done & b_done

def load_existing_memory_and_jsonl():
    memory = {}
    try:
        p = hf_hub_download(
            repo_id=NEW_OUTPUT_REPO_ID, repo_type="dataset",
            filename=f"{EXPORT_FOLDER}/STATE/story_memory.json", token=HF_TOKEN
        )
        memory = json.loads(Path(p).read_text(encoding="utf-8"))
        (LOCAL_STATE / "story_memory.json").write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
        print("✅ Loaded existing story_memory.json from Hugging Face.", flush=True)
    except Exception:
        print("ℹ️ Starting fresh story_memory.json", flush=True)

    for name in ["track_a.jsonl", "track_b.jsonl"]:
        target = LOCAL_TRAINING / name
        try:
            p = hf_hub_download(
                repo_id=NEW_OUTPUT_REPO_ID, repo_type="dataset",
                filename=f"{EXPORT_FOLDER}/TRAINING_DATASETS/{name}", token=HF_TOKEN
            )
            shutil.copyfile(p, target)
            print(f"✅ Loaded existing {name}", flush=True)
        except Exception:
            target.write_text("", encoding="utf-8")

    return memory

# ==========================================
# 5. PIPELINES (TRACK A & TRACK B)
# ==========================================
def run_track_a_cleaning(raw_text):
    # Plain string concatenation avoids invalid format specifier errors
    prompt = "You are a strict transcript-cleaning tool for a Hindi audio-drama script.\n" \
             "Fix ASR mistakes, grammar, spelling, punctuation, spacing, and obvious name errors.\n" \
             "Preserve story meaning, scene order, and episode identity exactly.\n" \
             "Do not invent scenes, summarize, rewrite the story, or change chronology.\n" \
             "Return ONLY the cleaned transcript text for the requested episode, with no preamble or explanation:\n\n" + raw_text

    print("   🥇 [Pri-1] Trying Gemini 3 Keys...", flush=True)
    gemini_out = call_gemini_with_fallback(prompt)
    if gemini_out: return gemini_out

    if ollama_client:
        for m_name in ["glm-5.2", "deepseek-v4-pro"]:
            try:
                print(f"   🥈 Trying Ollama ({m_name})...", flush=True)
                res = ollama_client.chat.completions.create(
                    model=m_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3, max_tokens=8192
                )
                out = res.choices[0].message.content.strip()
                if out: return out
            except Exception as e:
                print(f"   ⚠️ Ollama {m_name} failed: {e}", flush=True)

    raise RuntimeError("Track A: All primary and fallback models failed!")


def run_track_b_json(cleaned_text, ep_num, memory_context):
    schema_hint = """{
  "episode": 0,
  "story_summary": "",
  "opening_state": {"situation": "", "active_problem": "", "immediate_goal": ""},
  "character_states": [{"name": "", "role": "", "knowledge": [], "current_goal": "", "emotion": "", "relationships": [], "change_in_episode": ""}],
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

    mem_str = json.dumps(memory_context, ensure_ascii=False) if memory_context else "{}"

    prompt = "You are a strict story-analysis and extraction tool. You only extract facts explicitly supported by the given transcript.\n" \
             "Respond with ONLY a single valid JSON object matching the exact schema given below, with no prose before or after it.\n\n" \
             "Continuity memory from previous episodes:\n" + mem_str + "\n\n" \
             "Episode number: " + str(ep_num) + "\n\n" \
             "Cleaned transcript:\n" + cleaned_text + "\n\n" \
             "Extract structured JSON using exactly this schema:\n" + schema_hint

    print("   🥇 [Pri-1] Extracting Story Intelligence JSON with Gemini...", flush=True)
    gemini_out = call_gemini_with_fallback(prompt)
    if gemini_out and "{" in gemini_out: return gemini_out

    if ollama_client:
        for m_name in ["nemotron-3-ultra", "deepseek-v4-pro"]:
            try:
                print(f"   🥈 Extracting JSON with Ollama ({m_name})...", flush=True)
                res = ollama_client.chat.completions.create(
                    model=m_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2, max_tokens=8192
                )
                out = res.choices[0].message.content.strip()
                if out: return out
            except Exception as e:
                print(f"   ⚠️ Ollama {m_name} failed: {e}", flush=True)

    raise RuntimeError("Track B: All primary and fallback models failed!")

# ==========================================
# 6. MAIN ROUTINE
# ==========================================
def main():
    completed = get_completed_episodes()
    print(f"✅ Already completed on Hugging Face: {len(completed)} episodes.", flush=True)
    
    story_memory = load_existing_memory_and_jsonl()
    processed_count = 0

    for ep in range(1, TOTAL_EPISODES + 1):
        if ep in completed:
            continue

        print(f"\n▶️ [Episode {ep:04d}] Processing...", flush=True)
        
        # 1. Download Raw Episode
        try:
            raw_path = hf_hub_download(
                repo_id=SOURCE_REPO_ID, repo_type="dataset",
                filename=f"{SOURCE_FOLDER}/Episode_{ep:04d}.txt", token=HF_TOKEN,
                local_dir=LOCAL_ROOT / "RAW"
            )
            raw_text = Path(raw_path).read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"❌ Failed to download raw Episode {ep:04d}: {e}", flush=True)
            continue

        # 2. Track A: Clean Episode
        try:
            cleaned_text = run_track_a_cleaning(raw_text)
            (LOCAL_TRACK_A / f"Episode_{ep:04d}.txt").write_text(cleaned_text, encoding="utf-8")
            print("   ✅ Track A Cleaned & Saved!", flush=True)
        except Exception as e:
            print(f"   ❌ Track A Failed for Episode {ep:04d}: {e}", flush=True)
            continue

        # 3. Track B: Story Intelligence JSON
        try:
            raw_json_str = run_track_b_json(cleaned_text, ep, story_memory)
            
            clean_json_str = re.sub(r"^```json\s*", "", raw_json_str, flags=re.MULTILINE)
            clean_json_str = re.sub(r"```\s*$", "", clean_json_str, flags=re.MULTILINE).strip()
            
            match = re.search(r"\{.*\}", clean_json_str, re.DOTALL)
            if match: clean_json_str = match.group(0)
            
            track_b_data = json.loads(clean_json_str)
            track_b_data["episode"] = ep
            
            # Save Individual Episode JSON
            (LOCAL_TRACK_B / f"Episode_{ep:04d}.json").write_text(
                json.dumps(track_b_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            
            # Append to TRAINING_DATASETS (.jsonl)
            with open(LOCAL_TRAINING / "track_a.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"episode": ep, "text": cleaned_text}, ensure_ascii=False) + "\n")
                
            with open(LOCAL_TRAINING / "track_b.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(track_b_data, ensure_ascii=False) + "\n")
                
            # Update Memory in STATE
            story_memory[str(ep)] = track_b_data.get("continuity_memory_update", [])
            (LOCAL_STATE / "story_memory.json").write_text(
                json.dumps(story_memory, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            
            print("   ✅ Track B JSON & Memory State Updated!", flush=True)
        except Exception as e:
            print(f"   ❌ Track B Failed for Episode {ep:04d}: {e}", flush=True)
            continue

        processed_count += 1

        # 4. Instant Upload (Every 1 episode)
        if processed_count % UPLOAD_EVERY_EPISODES == 0:
            print(f"   ☁️ Uploading Episode {ep:04d} batch to Hugging Face...", flush=True)
            try:
                hf_api.upload_folder(
                    repo_id=NEW_OUTPUT_REPO_ID, repo_type="dataset",
                    folder_path=str(LOCAL_ROOT), path_in_repo=EXPORT_FOLDER,
                    token=HF_TOKEN, commit_message=f"VEDA Engine: Upload Ep {ep:04d}"
                )
                print(f"   🎉 SUCCESS: Episode {ep:04d} Live on Hugging Face!", flush=True)
            except Exception as e:
                print(f"   ⚠️ Upload delay: {e}. Will retry in next cycle.", flush=True)

        time.sleep(2)

if __name__ == "__main__":
    main()
                      
