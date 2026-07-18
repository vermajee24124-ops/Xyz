!pip install google-genai huggingface_hub -q

import os
import time
import re
from google import genai
from google.genai import types
from huggingface_hub import HfApi, hf_hub_download

# ==========================================
# 1. SETUP & SECRETS (Yahan Apni Keys Daalein)
# ==========================================
KEYS = [
    "YOUR_GEMINI_KEY_1",  # Yahan apni pehli key daalein
    "YOUR_GEMINI_KEY_2",  # Yahan apni dusri key daalein
    "YOUR_GEMINI_KEY_3"   # Yahan apni teesri key daalein
]
KEYS = [k.strip() for k in KEYS if k.strip() and "YOUR_" not in k]

HF_TOKEN = "YOUR_HUGGINGFACE_WRITE_TOKEN" # Yahan apna HF Token daalein

if not HF_TOKEN or "YOUR_" in HF_TOKEN or not KEYS:
    print("❌ ERROR: Kripya code mein apni Gemini Keys aur HF Token sahi se paste karein!")
    exit(1)

REPO_ID = "Kumarverma11/PocketFM_Audio"
SOURCE_FOLDER = "Transcripts_Episode_0001_to_0200"
TARGET_FOLDER = "Studio_Grammar_Corrected_Deep_Reasoning" 
MODEL_ID = 'gemini-3.1-flash-lite'

hf_api = HfApi()

current_key_idx = 0
def get_next_client():
    global current_key_idx
    client = genai.Client(api_key=KEYS[current_key_idx])
    current_key_idx = (current_key_idx + 1) % len(KEYS)
    return client

# ==========================================
# 2. THE MISSING EPISODES LIST
# ==========================================
missing_eps = [1, 3, 8, 9, 23, 24, 25, 28, 29, 30, 35, 42, 46, 51, 55, 62, 64, 68, 75, 79, 80, 81, 83, 84, 85, 88, 89, 91, 94, 97, 100, 113, 121, 123, 125, 129, 134, 135, 140, 141, 144, 146, 149, 156, 158, 160, 162, 170, 173, 175, 180, 182, 189, 193, 194, 195, 196, 197]

print(f"🚀 Sirf bache hue {len(missing_eps)} episodes ko process karna shuru kar rahe hain...\n")

# ==========================================
# 3. DEEP REASONING PROMPT & SAFE CONFIG
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
- TUMHE HAR HAAL MEIN PURA TEXT WAPAS DENA HAI. EMPTY RESPONSE MAT DENA.
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
    
    print(f"[{idx+1}/{len(missing_eps)}] Downloading & Processing {filename}...")
    
    try:
        local_path = hf_hub_download(repo_id=REPO_ID, filename=source_path, repo_type="dataset", token=HF_TOKEN)
        with open(local_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            
        fixed_text = None
        
        # 3 Retry Auto-Fallback (To combat Empty Responses)
        for attempt in range(3):
            try:
                client = get_next_client()
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=f"{system_prompt}\n\nTEXT:\n{raw_text}",
                    config=safe_config
                )
                
                # Check if text is successfully generated
                try:
                    text = response.text
                    if text:
                        fixed_text = text.strip()
                        break 
                except Exception as e:
                    # Agar API block karti hai toh exact reason print karega
                    reason = response.candidates[0].finish_reason if response.candidates else "Unknown Block"
                    print(f"  ⚠️ Attempt {attempt+1} Blocked. Reason: {reason}. Retrying...")
                    time.sleep(3)
                    
            except Exception as api_e:
                print(f"  ⚠️ Attempt {attempt+1} API Error: {api_e}. Retrying...")
                time.sleep(3)
        
        if not fixed_text:
            print(f"❌ {filename} 3 attempts ke baad bhi fail ho gaya. Isko manual dekhna padega.")
            continue
            
        # Extra AI text cleanup
        fixed_text = re.sub(r'^(यहाँ आपका टेक्स्ट.*?है:?\s*)', '', fixed_text, flags=re.IGNORECASE)
            
        # Save locally
        temp_save_path = f"./{filename}"
        with open(temp_save_path, 'w', encoding='utf-8') as f:
            f.write(fixed_text)
            
        # Upload
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
        print(f"❌ Download/System Error on {filename}: {e}")
        
    time.sleep(2) # 2 sec delay taaki rate limit cross na ho

print("\n🎉 Mission Accomplished! Saare bache hue episodes process ho gaye hain.")
