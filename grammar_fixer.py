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
# 2. DEEP REASONING PROMPT & SAFE CONFIG
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
# 3. ZIDDI AUTO-LOOP (SELF-HEALING ARCHITECTURE)
# ==========================================
expected_eps = set(range(1, 201))
confirmation_count = 0
TARGET_CONFIRMATIONS = 3
pass_count = 1
MAX_PASSES = 25 # Loop tab tak chalega jab tak kaam khatam na ho (limit 25 ki hai taaki server hang na ho)

while pass_count <= MAX_PASSES:
    print(f"\n{'='*60}")
    print(f"🔄 --- LOOP PASS {pass_count} : HUGGING FACE SCANNING ---")
    print(f"{'='*60}\n")
    
    # 1. Hugging Face ko scan karna
    try:
        all_files = hf_api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    except Exception as e:
        print(f"❌ HF Scan Error: {e}. Retrying pass...")
        time.sleep(10)
        continue

    target_eps = set()
    for f in all_files:
        if f.startswith(f"{TARGET_FOLDER}/Episode_"):
            match = re.search(r'Episode_(\d{4})\.txt', f)
            if match: 
                target_eps.add(int(match.group(1)))

    missing_eps = sorted(list(expected_eps - target_eps))
    
    # 2. CONFIRMATION CHECK
    if not missing_eps:
        confirmation_count += 1
        print(f"✅ CONFIRMATION {confirmation_count}/{TARGET_CONFIRMATIONS} SUCCESS: 200/200 Episodes mil gaye!")
        
        if confirmation_count >= TARGET_CONFIRMATIONS:
            print("\n🎉 MISSION 100% ACCOMPLISHED! Teeno confirmations pass ho gaye. Ab script chain ki saans le rahi hai. BINGE-WORTHY series ready hai!")
            exit(0)
        else:
            print("⏳ Agli confirmation ke liye 15 seconds wait kar rahe hain...")
            time.sleep(15)
            continue
    else:
        confirmation_count = 0 # Agar ek bhi missing mil gaya, toh confirmation reset ho jayegi
        
    print(f"⚠️ {len(missing_eps)} Episodes abhi bhi bache hain: {missing_eps}")
    print("🚀 Firse shuru karte hain...\n")
    
    # 3. MISSING EPISODES PROCESSING
    for idx, ep in enumerate(missing_eps):
        filename = f"Episode_{ep:04d}.txt"
        source_path = f"{SOURCE_FOLDER}/{filename}"
        
        print(f"[{idx+1}/{len(missing_eps)}] Theek kiya jaa raha hai: {filename}...")
        
        try:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=source_path, repo_type="dataset", token=HF_TOKEN)
            with open(local_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
                
            fixed_text = None
            
            # API Retries (4 koshish karega har file par)
            for attempt in range(4):
                try:
                    client = get_next_client()
                    response = client.models.generate_content(
                        model=MODEL_ID,
                        contents=f"{system_prompt}\n\nTEXT:\n{raw_text}",
                        config=safe_config
                    )
                    
                    try:
                        text = response.text
                        if text:
                            fixed_text = text.strip()
                            break 
                    except Exception as e:
                        reason = response.candidates[0].finish_reason if response.candidates else "Unknown Block/Error"
                        print(f"  ⚠️ Attempt {attempt+1} Blocked/Failed ({reason}). Retrying...")
                        time.sleep(5) 
                except Exception as api_e:
                    print(f"  ⚠️ Attempt {attempt+1} API Network Error: {api_e}. Retrying...")
                    time.sleep(5)
            
            if not fixed_text:
                print(f"❌ {filename} is pass mein theek nahi ho paaya. Next pass mein isko fir se pakdenge.")
                continue
                
            # Extra text cleanup
            fixed_text = re.sub(r'^(यहाँ आपका टेक्स्ट.*?है:?\s*)', '', fixed_text, flags=re.IGNORECASE)
                
            temp_save_path = f"./{filename}"
            with open(temp_save_path, 'w', encoding='utf-8') as f:
                f.write(fixed_text)
                
            hf_api.upload_file(
                path_or_fileobj=temp_save_path,
                path_in_repo=f"{TARGET_FOLDER}/{filename}",
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN,
                commit_message=f"Auto-Loop Grammar Fix for {filename}"
            )
            print(f"  ✅ {filename} Successfully Uploaded!")
            if os.path.exists(temp_save_path): os.remove(temp_save_path)
            
        except Exception as e:
            print(f"❌ System Error on {filename}: {e}")
            
        time.sleep(2) # Safe zone
        
    print(f"\n⏳ Loop {pass_count} pura hua. API server ko saans lene ke liye 30 second de rahe hain...")
    time.sleep(30) # Uploaded files index hone ke liye lamba wait
    pass_count += 1

print("\n🚨 SCRIPT STOPPED: Max 25 passes poore ho gaye hain par files abhi bhi bachi hain. Logs check karein.")
