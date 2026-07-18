import os
import time
import re
from groq import Groq
from huggingface_hub import HfApi, hf_hub_download

# ==========================================
# 1. SETUP & SECRETS
# ==========================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

if not HF_TOKEN or not GROQ_API_KEY:
    print("❌ ERROR: HF_TOKEN ya GROQ_API_KEY GitHub Secrets mein missing hain!")
    exit(1)

REPO_ID = "Kumarverma11/PocketFM_Audio"
SOURCE_FOLDER = "Transcripts_Episode_0001_to_0200"
TARGET_FOLDER = "Studio_Grammar_Corrected_Deep_Reasoning" 

hf_api = HfApi()
client = Groq(api_key=GROQ_API_KEY)

# Naye Reasoning Models ki list (Qwen sabse upar, uske baad GPT-OSS)
ACTIVE_MODELS = [
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b"
]

# ==========================================
# 2. DEEP REASONING PROMPT
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

# ==========================================
# 3. ZIDDI AUTO-LOOP (SELF-HEALING ARCHITECTURE)
# ==========================================
expected_eps = set(range(1, 201))
confirmation_count = 0
TARGET_CONFIRMATIONS = 3
pass_count = 1
MAX_PASSES = 20 

while pass_count <= MAX_PASSES:
    print(f"\n{'='*60}")
    print(f"🔄 --- LOOP PASS {pass_count} : HUGGING FACE SCANNING ---")
    print(f"{'='*60}\n")
    
    # 1. Hugging Face Scan
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
            print("\n🎉 MISSION 100% ACCOMPLISHED! Teeno confirmations pass ho gaye. Naye Reasoning Models ne saara kaam nipta diya!")
            exit(0)
        else:
            print("⏳ Agli confirmation ke liye 10 seconds wait kar rahe hain...")
            time.sleep(10)
            continue
    else:
        confirmation_count = 0 
        
    print(f"⚠️ {len(missing_eps)} Episodes abhi bhi bache hain: {missing_eps}")
    print("🚀 Qwen & GPT-OSS Reasoning Models ke sath process shuru kar rahe hain...\n")
    
    # 3. MISSING EPISODES PROCESSING
    for idx, ep in enumerate(missing_eps):
        filename = f"Episode_{ep:04d}.txt"
        source_path = f"{SOURCE_FOLDER}/{filename}"
        
        print(f"[{idx+1}/{len(missing_eps)}] Theek kar raha hai: {filename}...")
        
        try:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=source_path, repo_type="dataset", token=HF_TOKEN)
            with open(local_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
                
            fixed_text = None
            
            # Auto-Switch Model Logic (Naye Reasoning Models ke sath)
            for attempt in range(len(ACTIVE_MODELS)):
                current_model = ACTIVE_MODELS[attempt]
                try:
                    # Naya API Call Structure
                    completion = client.chat.completions.create(
                        model=current_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"TEXT:\n{raw_text}"}
                        ],
                        temperature=0.3, # Grammar ke liye balanced
                        max_completion_tokens=4096, # Lamba episode cover karne ke liye
                        top_p=1,
                        reasoning_effort="high", # Deep thinking mode ON
                        stream=False # Auto-script ke liye False rakhna behtar hai
                    )
                    
                    text = completion.choices[0].message.content
                    if text:
                        fixed_text = text.strip()
                        print(f"  🧠 Processed successfully using {current_model}")
                        break 
                except Exception as api_e:
                    print(f"  ⚠️ {current_model} fail hua: {api_e}. Switching to next model...")
                    time.sleep(3)
            
            if not fixed_text:
                print(f"❌ {filename} is pass mein theek nahi ho paaya. Next pass mein isko fir se pakdenge.")
                continue
                
            # Extra text cleanup
            fixed_text = re.sub(r'^(यहाँ आपका टेक्स्ट.*?है:?\s*)', '', fixed_text, flags=re.IGNORECASE)
            fixed_text = re.sub(r'^(ये रहा आपका.*?है:?\s*)', '', fixed_text, flags=re.IGNORECASE)
                
            temp_save_path = f"./{filename}"
            with open(temp_save_path, 'w', encoding='utf-8') as f:
                f.write(fixed_text)
                
            hf_api.upload_file(
                path_or_fileobj=temp_save_path,
                path_in_repo=f"{TARGET_FOLDER}/{filename}",
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN,
                commit_message=f"Deep Reasoning Grammar Fix using {current_model} for {filename}"
            )
            print(f"  ✅ {filename} Successfully Uploaded!")
            if os.path.exists(temp_save_path): os.remove(temp_save_path)
            
        except Exception as e:
            print(f"❌ System Error on {filename}: {e}")
            
        # Groq Rate Limit Protection 
        time.sleep(3) 
        
    print(f"\n⏳ Loop {pass_count} pura hua. HF Update ke liye 30 second wait kar rahe hain...")
    time.sleep(30)
    pass_count += 1

print("\n🚨 SCRIPT STOPPED: Max passes poore ho gaye hain.")
