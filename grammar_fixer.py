import os
import time
from google import genai
from google.genai import types
from huggingface_hub import HfApi, hf_hub_download

# ==========================================
# 1. SETUP & SECRETS (Error-Free Loading)
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
TARGET_FOLDER = "Studio_Grammar_Corrected_Deep_Reasoning" # Naya aur Fresh Folder
MODEL_ID = 'gemini-3.1-flash-lite'

hf_api = HfApi()

current_key_idx = 0
def get_next_client():
    global current_key_idx
    client = genai.Client(api_key=KEYS[current_key_idx])
    current_key_idx = (current_key_idx + 1) % len(KEYS)
    return client

# ==========================================
# 2. DEEP REASONING & HIGH THINKING PROMPT
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
    temperature=0.2, # Deep reasoning aur logical rewrite ke liye best
    safety_settings=[
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    ]
)

# ==========================================
# 3. PROCESS & UPLOAD (AUTO-RETRY SYSTEM)
# ==========================================
total_episodes = 200
batch_size = 45 # 1 minute mein 45 episodes

for i in range(1, total_episodes + 1, batch_size):
    print(f"\n🚀 Processing Batch: Episodes {i} to {min(i + batch_size - 1, total_episodes)}")
    
    for ep in range(i, min(i + batch_size, total_episodes + 1)):
        filename = f"Episode_{ep:04d}.txt"
        source_path = f"{SOURCE_FOLDER}/{filename}"
        
        try:
            # 1. Download
            local_path = hf_hub_download(repo_id=REPO_ID, filename=source_path, repo_type="dataset", token=HF_TOKEN)
            with open(local_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
                
            # 2. Process with AUTO-RETRY (To fix NoneType errors)
            fixed_text = None
            for attempt in range(3): # Agar khali response aaya, toh 3 baar koshish karega
                try:
                    client = get_next_client()
                    response = client.models.generate_content(
                        model=MODEL_ID,
                        contents=f"{system_prompt}\n\nTEXT:\n{raw_text}",
                        config=safe_config
                    )
                    
                    if response and response.text:
                        fixed_text = response.text.strip()
                        break # Success hone par loop se bahar aa jayega
                    else:
                        print(f"⚠️ {filename} (Attempt {attempt+1}): API ne khali response diya. Retrying...")
                        time.sleep(3)
                except Exception as api_e:
                    print(f"⚠️ {filename} (Attempt {attempt+1}) API Error: {api_e}")
                    time.sleep(3)
            
            if not fixed_text:
                print(f"❌ {filename} completely failed after 3 attempts. Skipping...")
                continue # Agar 3 baar mein bhi fail ho, toh crash karne ke bajay aage badh jayega
                
            # 3. Save locally
            temp_save_path = f"./{filename}"
            with open(temp_save_path, 'w', encoding='utf-8') as f:
                f.write(fixed_text)
                
            # 4. Upload to Hugging Face
            hf_api.upload_file(
                path_or_fileobj=temp_save_path,
                path_in_repo=f"{TARGET_FOLDER}/{filename}",
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN,
                commit_message=f"Deep Reasoning Grammar Fix for {filename}"
            )
            print(f"✅ {filename} fixed & uploaded!")
            os.remove(temp_save_path)
            
        except Exception as e:
            print(f"❌ Critical Error on {filename}: {e}")
            
        time.sleep(1.5) # Request burst se bachne ke liye chhota delay
            
    print("⏳ Batch complete. Waiting 60 seconds for API rate limits to reset...")
    time.sleep(60)

print("🎉 All 200 episodes processed with Deep Reasoning!")
