import os
import sys
import json
import re
import time
import traceback
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError, HfHubHTTPError
import google.generativeai as genai
from openai import OpenAI

# ==========================================
# CONFIGURATION & CONFIGURABLE SETTINGS
# ==========================================
SOURCE_REPO_ID = "Kumarverma11/PocketFM_Audio"
SOURCE_FOLDER = "Transcripts_Episode_0001_to_0200"

NEW_OUTPUT_REPO_ID = "Kumarverma11/VEDA_AI_Cleaned_Series"
EXPORT_FOLDER = "Veda_Processed_Episodes"
TOTAL_EPISODES = 200

# ⚙️ UPLOAD FREQUENCY (अभी 1 पर सेट है, बाद में इसे 10 या 20 कर सकते हैं)
UPLOAD_EVERY_EPISODES = 1 

# Primary Gemini Model
GEMINI_MODEL_ID = "gemini-3.1-flash-lite"

# ==========================================
# AI ERROR DIAGNOSTIC ENGINE (कारण + उपाय)
# ==========================================
def diagnose_error(exception_obj, context=""):
    """
    यह फंक्शन किसी भी एरर को पकड़कर उसका हिंदी में सटीक कारण और समाधान बताता है।
    """
    err_str = str(exception_obj)
    err_type = type(exception_obj).__name__
    
    print("\n" + "🚨"*30)
    print(f"❌ [ERROR DIAGNOSED] Location: {context}")
    print(f"📌 Exception Type: {err_type}")
    print(f"📄 Raw Error: {err_str[:300]}")
    print("-" * 60)
    
    # Error Analysis Logic
    if "403" in err_str or "Forbidden" in err_str or "Authorization" in err_str:
        print("💡 [कारण / Cause]: API Key रिजेक्ट कर दी गई है या ऑथेंटिकेशन फेल हो गया है।")
        print("🔧 [समाधान / Solution]:")
        print("   1. GitHub Repo Settings -> Secrets and variables -> Actions में जाएँ।")
        print("   2. अपनी API Key (NVIDIA / Gemini / Ollama) दोबारा कॉपी-पेस्ट करके अपडेट करें।")
        print("   3. सुनिश्चित करें कि Key के आगे-पीछे कोई खाली स्पेस (Space) न हो।")
        
    elif "401" in err_str or "Unauthorized" in err_str:
        print("💡 [कारण / Cause]: Hugging Face Token में राइट (WRITE) परमिशन नहीं है।")
        print("🔧 [समाधान / Solution]: huggingface.co/settings/tokens पर जाएँ और 'Write' परमिशन वाला नया टोकन बनाएँ।")
        
    elif "429" in err_str or "Quota" in err_str or "ResourceExhausted" in err_str:
        print("💡 [कारण / Cause]: API की रेट लिमिट या फ्री क्रेडिट्स ख़त्म हो गए हैं।")
        print("🔧 [समाधान / Solution]: सिस्टम स्वतः दूसरे बैकअप मॉडल (Ollama/DeepSeek) पर स्विच कर रहा है। चिंता की बात नहीं है।")
        
    elif "json" in err_type.lower() or "JSONDecodeError" in err_type or "Expecting value" in err_str:
        print("💡 [कारण / Cause]: AI मॉडल ने शुद्ध JSON के बजाय कोई अतिरिक्त प्लेन टेक्स्ट या खराब फॉर्मेट दिया।")
        print("🔧 [समाधान / Solution]: कोड का Regex ऑटो-क्लीनर इसे फिक्स करने की कोशिश कर रहा है। यदि बार-बार हो, तो सिस्टम अगला मॉडल ट्राई करेगा।")
        
    elif "ConnectionError" in err_type or "Timeout" in err_type:
        print("💡 [कारण / Cause]: इंटरनेट या सर्वर नेटवर्क कनेक्शन ड्रॉप हुआ।")
        print("🔧 [समाधान / Solution]: कोड थोड़ी देर रुककर खुद री-ट्राई (Auto-Retry) करेगा।")
        
    else:
        print("💡 [कारण / Cause]: अनपेक्षित कोड या सर्वर एरर आया है।")
        print("🔧 [समाधान / Solution]: स्टैक ट्रेस देखें और रिपोजिटरी के इश्यू में चेक करें।")
        
    print("🚨"*30 + "\n")

# ==========================================
# LOAD ENVIRONMENT SECRETS (FOR PUBLIC REPO)
# ==========================================
HF_TOKEN = os.environ.get("HF_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")

if not HF_TOKEN or not GEMINI_API_KEY:
    print("FATAL: Required environment variables (HF_TOKEN, GEMINI_API_KEY) missing!")
    sys.exit(1)

# Initialize Clients
hf_api = HfApi(token=HF_TOKEN)

# Initialize Gemini with Safety OFF
genai.configure(api_key=GEMINI_API_KEY)
gemini_safety_config = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]
gemini_model = genai.GenerativeModel(GEMINI_MODEL_ID, safety_settings=gemini_safety_config)

# Initialize Ollama Cloud Client
ollama_client = None
if OLLAMA_API_KEY:
    ollama_client = OpenAI(
        base_url="https://api.ollama.com/v1",
        api_key=OLLAMA_API_KEY
    )

# Directories
LOCAL_ROOT = Path("veda_work")
LOCAL_RAW = LOCAL_ROOT / "RAW_TRANSCRIPTS"
LOCAL_TRACK_A = LOCAL_ROOT / "TRACK_A_RECONSTRUCTED"
LOCAL_TRACK_B = LOCAL_ROOT / "TRACK_B_STORY_JSON"

for d in [LOCAL_RAW, LOCAL_TRACK_A, LOCAL_TRACK_B]:
    d.mkdir(parents=True, exist_ok=True)

# Ensure Repo Exists
try:
    hf_api.create_repo(repo_id=NEW_OUTPUT_REPO_ID, repo_type="dataset", exist_ok=True)
except Exception as e:
    diagnose_error(e, "Creating Output Dataset Repo")

# ==========================================
# AUTO-RESUME ENGINE
# ==========================================
def get_completed_episodes():
    try:
        files = hf_api.list_repo_files(repo_id=NEW_OUTPUT_REPO_ID, repo_type="dataset")
    except Exception as e:
        diagnose_error(e, "Checking HF Completed Episodes")
        return set()
    
    a_done, b_done = set(), set()
    pattern = re.compile(r"Episode_(\d{4})\.(txt|json)$")
    for f in files:
        if EXPORT_FOLDER not in f: continue
        m = pattern.search(f)
        if m:
            ep_num = int(m.group(1))
            if ".txt" in f: a_done.add(ep_num)
            if ".json" in f: b_done.add(ep_num)
            
    return a_done & b_done

# ==========================================
# AI CALL PIPELINES (TRACK A & TRACK B)
# ==========================================
def run_track_a_cleaning(raw_text):
    prompt = f"""You are a professional Hindi script dialogue writer.
Fix ASR mistakes, grammar, spelling, and missing words in this Hindi transcript. 
Use deep reasoning to reconstruct broken sentences while preserving the story's emotional flow and cliffhangers.
Return ONLY the cleaned transcript text:

{raw_text}"""

    # 1. First Priority: Gemini
    try:
        print(f"   🥇 [Pri-1] Trying Gemini ({GEMINI_MODEL_ID})...")
        res = gemini_model.generate_content(prompt)
        if res.text and len(res.text.strip()) > 50:
            return res.text.strip()
    except Exception as e:
        diagnose_error(e, "Track A - Gemini 3.1 Flash-Lite")

    # 2. Second Priority: Ollama glm-5.2
    if ollama_client:
        try:
            print("   🥈 [Pri-2] Trying Ollama (glm-5.2)...")
            res = ollama_client.chat.completions.create(
                model="glm-5.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6, max_tokens=8192
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            diagnose_error(e, "Track A - Ollama glm-5.2")

        # 3. Third Priority: Ollama deepseek-v4-pro
        try:
            print("   🥉 [Pri-3] Trying Ollama (deepseek-v4-pro)...")
            res = ollama_client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6, max_tokens=8192
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            diagnose_error(e, "Track A - Ollama deepseek-v4-pro")

    raise RuntimeError("Track A: All primary and fallback models failed!")


def run_track_b_json(cleaned_text):
    prompt = f"""Analyze this Hindi audio drama episode and extract structured Story Intelligence.
Return ONLY a single valid JSON object matching this schema:
{{
  "story_summary": "brief summary",
  "character_states": [{"name": "", "role": "", "emotion": "", "goal": ""}],
  "conflicts": [],
  "cliffhanger": {"present": true, "ending_event": ""},
  "continuity_memory_update": []
}}

Cleaned Transcript:
{cleaned_text}"""

    # 1. First Priority: Gemini
    try:
        print(f"   🥇 [Pri-1] Extracting JSON with Gemini ({GEMINI_MODEL_ID})...")
        res = gemini_model.generate_content(prompt)
        if res.text and "{" in res.text:
            return res.text.strip()
    except Exception as e:
        diagnose_error(e, "Track B - Gemini")

    # 2. Second Priority: Ollama nemotron-3-ultra
    if ollama_client:
        try:
            print("   🥈 [Pri-2] Extracting JSON with Ollama (nemotron-3-ultra)...")
            res = ollama_client.chat.completions.create(
                model="nemotron-3-ultra",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=8192
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            diagnose_error(e, "Track B - Ollama nemotron-3-ultra")

        # 3. Third Priority: Ollama deepseek-v4-pro
        try:
            print("   🥉 [Pri-3] Extracting JSON with Ollama (deepseek-v4-pro)...")
            res = ollama_client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=8192
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            diagnose_error(e, "Track B - Ollama deepseek-v4-pro")

    raise RuntimeError("Track B: All primary and fallback models failed!")

# ==========================================
# MAIN ROUTINE
# ==========================================
def main():
    print("🚀 Starting VEDA AI Pipeline Engine...")
    completed = get_completed_episodes()
    print(f"✅ Already uploaded episodes: {len(completed)}")
    
    processed_count = 0

    for ep in range(1, TOTAL_EPISODES + 1):
        if ep in completed:
            continue

        print(f"\n▶️ [Episode {ep:04d}] Processing...")
        
        # 1. Download Raw
        try:
            raw_path = hf_hub_download(
                repo_id=SOURCE_REPO_ID, repo_type="dataset",
                filename=f"{SOURCE_FOLDER}/Episode_{ep:04d}.txt", token=HF_TOKEN,
                local_dir=LOCAL_RAW
            )
            with open(raw_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        except Exception as e:
            diagnose_error(e, f"Downloading Episode {ep:04d}")
            continue

        # 2. Track A: Clean & Reconstruct
        try:
            cleaned_text = run_track_a_cleaning(raw_text)
            clean_file = LOCAL_TRACK_A / f"Episode_{ep:04d}.txt"
            clean_file.write_text(cleaned_text, encoding="utf-8")
            print("   ✅ Track A Completed!")
        except Exception as e:
            diagnose_error(e, f"Track A Episode {ep:04d}")
            continue

        # 3. Track B: JSON Intelligence
        try:
            json_str = run_track_b_json(cleaned_text)
            clean_json_str = re.sub(r"^```json\s*", "", json_str, flags=re.MULTILINE)
            clean_json_str = re.sub(r"```\s*$", "", clean_json_str, flags=re.MULTILINE).strip()
            
            match = re.search(r"\{.*\}", clean_json_str, re.DOTALL)
            if match: clean_json_str = match.group(0)
            
            parsed_json = json.loads(clean_json_str)
            parsed_json["episode"] = ep
            
            json_file = LOCAL_TRACK_B / f"Episode_{ep:04d}.json"
            json_file.write_text(json.dumps(parsed_json, ensure_ascii=False, indent=2), encoding="utf-8")
            print("   ✅ Track B Completed!")
        except Exception as e:
            diagnose_error(e, f"Track B Episode {ep:04d}")
            continue

        processed_count += 1

        # 4. Configurable Batch Upload (Set to 1 for instant upload testing)
        if processed_count % UPLOAD_EVERY_EPISODES == 0:
            print(f"   ☁️ Uploading batch to Hugging Face...")
            try:
                hf_api.upload_folder(
                    repo_id=NEW_OUTPUT_REPO_ID, repo_type="dataset",
                    folder_path=str(LOCAL_ROOT), path_in_repo=EXPORT_FOLDER,
                    token=HF_TOKEN, commit_message=f"VEDA Engine: Upload Ep {ep:04d}"
                )
                print(f"   🎉 SUCCESS: Episode {ep:04d} Live on Hugging Face!")
            except Exception as e:
                diagnose_error(e, f"Uploading Batch up to Episode {ep:04d}")

        time.sleep(2)

if __name__ == "__main__":
    main()
    
