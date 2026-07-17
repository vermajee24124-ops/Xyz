import os
import time
import re
from google import genai
from google.genai import types
from huggingface_hub import HfApi, hf_hub_download

# ==========================================
# 1. SETUP & SECRETS
# ==========================================
KEYS = [
    os.environ.get("GEMINI_KEY_1", "").strip(),
    os.environ.get("GEMINI_KEY_2", "").strip(),
    os.environ.get("GEMINI_KEY_3", "").strip()
]
KEYS = [k for k in KEYS if k] 
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

if not HF_TOKEN or not KEYS:
    print("❌ ERROR: HF_TOKEN ya Gemini Keys GitHub Secrets mein missing hain!")
    exit(1)

REPO_ID = "Kumarverma11/PocketFM_Audio"
SOURCE_FOLDER = "Transcripts_Episode_0001_to_0200"
TARGET_FOLDER = "Studio_Grammar_Corrected_Deep_Reasoning" # Same target folder
MODEL_ID = 'gemini-3.1-flash-lite'

hf_api = HfApi()

current_key_idx = 0
def get_next_client():
    global current_key_idx
    client = genai.Client(api_key=KEYS[current_key_idx])
    current_key_idx = (current_key_idx + 1) % len(KEYS)
    return client

# ==========================================
# 2. FIND MISSING EPISODES DYNAMICALLY
# ==========================================
print("🔍 Step 1: Hugging Face repo check kar rahe hain ki kaun se episodes chhut gaye hain...")

# Repo ki saari files ki list nikalna
all_files = hf_api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")

target_eps = set()

# Pata lagana ki target folder mein kaun se episodes already hain
for f in all_files:
    if f.startswith(f"{TARGET_FOLDER}/Episode_"):
        match = re.search(r'Episode_(\d{4})\.txt', f)
        if match: 
            target_eps.add(int(match.group(1)))

# Humara target 1 se 200 tak hai
expected_eps = set(range(1, 201))

# Chhute hue (Missing) episodes nikalna
missing_eps = sorted(list(expected_eps - target_eps))

if not missing_eps:
    print("🎉 Badhai ho! Sabhi 200 episodes pehle se hi successfully upload ho chuke hain. Koi file missing nahi hai.")
    exit(0)

print(f"⚠️ Total {len(missing_eps)} episodes missing mile hain: {missing_eps}")
print("🚀 In bache hue episodes ko Deep Reasoning ke sath fix karna shuru kar rahe hain...\n")

# ==========================================
# 3. DEEP REASONING PROMPT & SAFETY CONFIG
# ==========================================
system_prompt = """
Tum ek Master Hindi Copy Editor aur Proofreader ho. Tumhare paas "Deep Reasoning" aur "High Thinking" ki kshamata hai.
Niche diya gaya text ek audio series ka speech-to-text transcript hai, jismein BHAYANKAR galtiyan hain (galat shabd, galat sentence structure, aur meaning ki galtiyan).

TUMHARA KAAM (Step-by-Step Deep Reasoning):
1. Poori kahani ke context ko dhyan se samjho.
2. Har ek sentence (ek-ek line) ko padho aur usme chhipi galtiyon ko pehchano. Speech-to-text ki wajah se jo shabd galat type ho gaye hain, unhe deep reasoning lagakar sahi, logical shabdon aur vakyon (sentences) mein badlo. 
3. Jaise: Agar likha hai "नेवी ब्लू रंग का था। तीन पीस सूट पहने करीब।" toh ise theek karo aur likho: "नेवी कार नीले रंग की थी। वह तीन पीस सूट पहने करीब आया।"
4. Bhasha ekdum professional, clear aur story-telling (audio series) ke flow mein honi chahiye.

STRICT RULES:
- GLOSSARY RULE: 'नेवी' (Navy) ek Gadi (Car) ka naam hai, koi color nahi. Ise dhyan mein rakh kar theek karein.
- Kahani ka koi bhi dialogue, character ya main plot delete NAHI hona chahiye.
- Output mein SIRF theek kiya hua final text dena hai. Koi explanation, intro, ya 'यहाँ आपका टेक्स्ट है' jaisi baatein BILKUL NAHI likhni hain.
"""

safe_config = types.GenerateContentConfig(
    temperature=0.2, 
    safety_settings=[
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    ]
)

# ==========================================
# 4. PROCESS ONLY MISSING EPISODES
# ==========================================
for idx, ep in enumerate(missing_eps):
    filename = f"Episode_{ep:04d}.txt"
    source_path = f"{SOURCE_FOLDER}/{filename}"
    
    print(f"[{idx+1}/{len(missing_eps)}] Processing {filename}...")
    
    try:
        # Download raw file
        local_path = hf_hub_download(repo_id=REPO_ID, filename=source_path, repo_type="dataset", token=HF_TOKEN)
        with open(local_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            
        fixed_text = None
        
        # 3 Retry Auto-Fallback (Error-Free Mechanism)
        for attempt in range(3):
            try:
                client = get_next_client()
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=f"{system_prompt}\n\nTEXT:\n{raw_text}",
                    config=safe_config
                )
                
                if response and response.text:
                    fixed_text = response.text.strip()
                    break 
                else:
                    print(f"  ⚠️ Attempt {attempt+1} failed (Empty Response). Retrying...")
                    time.sleep(3)
            except Exception as api_e:
                print(f"  ⚠️ Attempt {attempt+1} API Error: {api_e}. Retrying...")
                time.sleep(3)
        
        if not fixed_text:
            print(f"❌ {filename} completely failed after 3 attempts. Skipping for now.")
            continue
            
        # Clean AI Intro galti se aane par
        fixed_text = re.sub(r'^(यहाँ आपका टेक्स्ट.*?है:?\s*)', '', fixed_text, flags=re.IGNORECASE)
            
        # Save locally
        temp_save_path = f"./{filename}"
        with open(temp_save_path, 'w', encoding='utf-8') as f:
            f.write(fixed_text)
            
        # Upload directly to the specific target folder
        hf_api.upload_file(
            path_or_fileobj=temp_save_path,
            path_in_repo=f"{TARGET_FOLDER}/{filename}",
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message=f"Deep Reasoning Grammar Fix for missing {filename}"
        )
        print(f"  ✅ {filename} fixed & uploaded successfully!")
        os.remove(temp_save_path)
        
    except Exception as e:
        print(f"❌ Critical Error on {filename}: {e}")
        
    # Rate limit safe delay
    time.sleep(1.5)

print("\n🎉 Missing episodes processing complete!")
