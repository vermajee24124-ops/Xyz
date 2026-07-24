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

# Real-time stdout output buffering for GitHub Actions logs
sys.stdout.reconfigure(line_buffering=True)

# ==========================================
# 1. CONFIGURATION & BATCH SETTINGS
# ==========================================
SOURCE_REPO_ID = "Kumarverma11/PocketFM_Audio"
SOURCE_FOLDER = "Transcripts_Episode_0001_to_0200"

NEW_OUTPUT_REPO_ID = "Kumarverma11/VEDA_AI_Cleaned_Series"
EXPORT_FOLDER = "Veda_Final_Training_Export_0001_to_0200"
TOTAL_EPISODES = 200

# ⚙️ BATCH UPLOAD: Har 10 episodes hone par 1 Commit (Rate Limit 429 Fix)
BATCH_SIZE = 10 
GEMINI_MODEL_ID = "gemini-3.1-flash-lite"

print("="*65, flush=True)
print("🌟 VEDA AI MASTER PIPELINE ENGINE (VERIFIED FREE MODELS) 🌟", flush=True)
print("="*65, flush=True)
print(f"📍 SOURCE REPO   : {SOURCE_REPO_ID}/{SOURCE_FOLDER}", flush=True)
print(f"📍 TARGET REPO   : {NEW_OUTPUT_REPO_ID}", flush=True)
print(f"📂 EXPORT FOLDER : {EXPORT_FOLDER}/", flush=True)
print("="*65 + "\n", flush=True)

# ==========================================
# 2. SECRETS & CLIENTS INITIALIZATION
# ==========================================
HF_TOKEN = os.environ.get("HF_TOKEN")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")

GEMINI_KEYS = [
    k for k in [
        os.environ.get("GEMINI_KEY_1"),
        os.environ.get("GEMINI_KEY_2"),
        os.environ.get("GEMINI_KEY_3")
    ] if k and k.strip()
]

print(f"🔑 Loaded {len(GEMINI_KEYS)} Gemini Key(s) for Primary Rotation.", flush=True)

if not HF_TOKEN:
    print("❌ FATAL: HF_TOKEN missing in Secrets!", flush=True)
    sys.exit(1)

hf_api = HfApi(token=HF_TOKEN)

# Ollama Client setup
ollama_client = None
if OLLAMA_API_KEY and OLLAMA_API_KEY.strip():
    try:
        ollama_client = OpenAI(
            base_url="https://ollama.com/v1",
            api_key=OLLAMA_API_KEY.strip()
        )
        print("✅ Ollama Free-Tier Client Initialized.", flush=True)
    except Exception as e:
        print(f"⚠️ Ollama Client init note: {e}", flush=True)

gemini_safety_config = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

# Local Folder Architecture Setup
LOCAL_ROOT = Path("veda_work")
LOCAL_TRACK_A = LOCAL_ROOT / "TRACK_A_CLEAN_EPISODES"
LOCAL_TRACK_B = LOCAL_ROOT / "TRACK_B_STORY_INTELLIGENCE"
LOCAL_TRAINING = LOCAL_ROOT / "TRAINING_DATASETS"
LOCAL_STATE = LOCAL_ROOT / "STATE"

for d in [LOCAL_TRACK_A, LOCAL_TRACK_B, LOCAL_TRAINING, LOCAL_STATE]:
    d.mkdir(parents=True, exist_ok=True)

try:
    hf_api.create_repo(repo_id=NEW_OUTPUT_REPO_ID, repo_type="dataset", exist_ok=True)
except Exception:
    pass

# ==========================================
# 3. SAFE AI CALLERS (CRASH PROOF)
# ==========================================
def safe_extract_gemini_response(res):
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
    for idx, key in enumerate(GEMINI_KEYS, start=1):
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(GEMINI_MODEL_ID, safety_settings=gemini_safety_config)
            res = model.generate_content(prompt_text)
            extracted = safe_extract_gemini_response(res)
            if extracted and len(extracted) > 20:
                print(f"   ✅ Gemini Key #{idx} Success!", flush=True)
                return extracted
            else:
                print(f"   ⚠️ Gemini Key #{idx} empty/blocked response. Next key in 2s...", flush=True)
                time.sleep(2)
        except Exception as e:
            print(f"   ⚠️ Gemini Key #{idx} Error ({e}). Trying next key...", flush=True)
            time.sleep(2)
    return None

# ==========================================
# 4. RESUME LOGIC & STATE SYNC
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
            
    return a_done & b_done  # Only consider episodes complete if BOTH Track A and Track B exist

def load_existing_memory_and_jsonl():
    memory = {}
    try:
        p = hf_hub_download(
            repo_id=NEW_OUTPUT_REPO_ID, repo_type="dataset",
            filename=f"{EXPORT_FOLDER}/STATE/story_memory.json", token=HF_TOKEN
        )
        memory = json.loads(Path(p).read_text(encoding="utf-8"))
        (LOCAL_STATE / "story_memory.json").write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
        print("✅ Synced existing story_memory.json from Hugging Face.", flush=True)
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
            print(f"✅ Synced existing {name}", flush=True)
        except Exception:
            target.write_text("", encoding="utf-8")

    return memory

def upload_current_batch(last_ep):
    print(f"   ☁️ [BATCH UPLOAD] Uploading batch up to Episode {last_ep:04d}...", flush=True)
    try:
        hf_api.upload_folder(
            repo_id=NEW_OUTPUT_REPO_ID, repo_type="dataset",
            folder_path=str(LOCAL_ROOT), path_in_repo=EXPORT_FOLDER,
            token=HF_TOKEN, commit_message=f"VEDA Batch Update: Up to Episode {last_ep:04d}"
        )
        print(f"   🎉 SUCCESS: Batch through Episode {last_ep:04d} is Live on Hugging Face!", flush=True)
        return True
    except Exception as e:
        print(f"   ⚠️ Batch upload delay: {e}. Will retry in next cycle.", flush=True)
        return False

# ==========================================
# 5. TRACK PIPELINES (PRIMARY + TOP 3 FREE OLLAMA)
# ==========================================
def run_track_a_cleaning(raw_text):
    prompt = "You are a strict transcript-cleaning tool for a Hindi audio-drama script.\n" \
             "Fix ASR mistakes, grammar, spelling, punctuation, spacing, and obvious name errors.\n" \
             "Preserve story meaning, scene order, and episode identity exactly.\n" \
             "Do not invent scenes, summarize, rewrite the story, or change chronology.\n" \
             "Return ONLY the cleaned transcript text for the requested episode, with no preamble:\n\n" + raw_text

    print("   🥇 Trying Gemini 3 Keys...", flush=True)
    gemini_out = call_gemini_with_fallback(prompt)
    if gemini_out: return gemini_out

    # Top 3 Verified Free Ollama Models
    if ollama_client:
        for m_name in ["gpt-oss:120b", "minimax-m3", "nemotron-3-super"]:
            try:
                print(f"   🥈 Trying Free Ollama ({m_name})...", flush=True)
                res = ollama_client.chat.completions.create(
                    model=m_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3, max_tokens=8192
                )
                out = res.choices[0].message.content.strip()
                if out and len(out) > 50: return out
            except Exception as e:
                print(f"   ⚠️ Ollama {m_name} note: {e}", flush=True)

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

    print("   🥇 Extracting Story Intelligence JSON with Gemini...", flush=True)
    gemini_out = call_gemini_with_fallback(prompt)
    if gemini_out and "{" in gemini_out: return gemini_out

    # Top 3 Verified Free Ollama Models for JSON
    if ollama_client:
        for m_name in ["nemotron-3-ultra", "gpt-oss:120b", "minimax-m3"]:
            try:
                print(f"   🥈 Extracting JSON with Free Ollama ({m_name})...", flush=True)
                res = ollama_client.chat.completions.create(
                    model=m_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2, max_tokens=8192
                )
                out = res.choices[0].message.content.strip()
                if out and "{" in out: return out
            except Exception as e:
                print(f"   ⚠️ Ollama {m_name} note: {e}", flush=True)

    raise RuntimeError("Track B: All primary and fallback models failed!")

# ==========================================
# 6. MAIN EXECUTION PIPELINE
# ==========================================
def main():
    completed = get_completed_episodes()
    print(f"✅ Total Completed Episodes on HF: {len(completed)}", flush=True)
    
    story_memory = load_existing_memory_and_jsonl()
    processed_since_last_upload = 0
    last_processed_ep = 0

    for ep in range(1, TOTAL_EPISODES + 1):
        if ep in completed:
            continue  # Automatically skips completed and fills missing episodes

        print(f"\n▶️ [Processing Episode {ep:04d}]...", flush=True)
        
        # 1. Download Raw Transcript
        try:
            raw_path = hf_hub_download(
                repo_id=SOURCE_REPO_ID, repo_type="dataset",
                filename=f"{SOURCE_FOLDER}/Episode_{ep:04d}.txt", token=HF_TOKEN,
                local_dir=LOCAL_ROOT / "RAW"
            )
            raw_text = Path(raw_path).read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"❌ Could not download raw Ep {ep:04d}: {e}", flush=True)
            continue

        # 2. Track A: Clean Episode
        try:
            cleaned_text = run_track_a_cleaning(raw_text)
            (LOCAL_TRACK_A / f"Episode_{ep:04d}.txt").write_text(cleaned_text, encoding="utf-8")
            print("   ✅ Track A Cleaned!", flush=True)
        except Exception as e:
            print(f"   ❌ Track A Failed for Ep {ep:04d}: {e}", flush=True)
            continue

        # 3. Track B: Story Intelligence JSON & Memory State
        try:
            raw_json_str = run_track_b_json(cleaned_text, ep, story_memory)
            
            clean_json_str = re.sub(r"^```json\s*", "", raw_json_str, flags=re.MULTILINE)
            clean_json_str = re.sub(r"```\s*$", "", clean_json_str, flags=re.MULTILINE).strip()
            
            match = re.search(r"\{.*\}", clean_json_str, re.DOTALL)
            if match: clean_json_str = match.group(0)
            
            track_b_data = json.loads(clean_json_str)
            track_b_data["episode"] = ep
            
            # Save Episode JSON
            (LOCAL_TRACK_B / f"Episode_{ep:04d}.json").write_text(
                json.dumps(track_b_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            
            # Append to JSONL Datasets
            with open(LOCAL_TRAINING / "track_a.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"episode": ep, "text": cleaned_text}, ensure_ascii=False) + "\n")
                
            with open(LOCAL_TRAINING / "track_b.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(track_b_data, ensure_ascii=False) + "\n")
                
            # Update Memory Context in STATE
            story_memory[str(ep)] = track_b_data.get("continuity_memory_update", [])
            (LOCAL_STATE / "story_memory.json").write_text(
                json.dumps(story_memory, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            
            print("   ✅ Track B JSON & Memory State Updated!", flush=True)
        except Exception as e:
            print(f"   ❌ Track B Failed for Ep {ep:04d}: {e}", flush=True)
            continue

        processed_since_last_upload += 1
        last_processed_ep = ep

        # 4. Batch Upload (Every 10 episodes to prevent HF Rate Limit 429)
        if processed_since_last_upload >= BATCH_SIZE:
            if upload_current_batch(last_processed_ep):
                processed_since_last_upload = 0
                
        time.sleep(1)

    # Final Batch Upload for remaining processed episodes
    if processed_since_last_upload > 0:
        print("\n☁️ Uploading final remaining batch...", flush=True)
        upload_current_batch(last_processed_ep)

if __name__ == "__main__":
    main()
            
