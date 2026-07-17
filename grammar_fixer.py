import os
import time
from google import genai
from google.genai import types
from huggingface_hub import HfApi, hf_hub_download

# ==========================================
# 1. SETUP & SECRETS
# ==========================================
# GitHub Secrets se API keys aur Token nikalna
KEYS = [
    os.environ.get("GEMINI_KEY_1"),
    os.environ.get("GEMINI_KEY_2"),
    os.environ.get("GEMINI_KEY_3")
]
HF_TOKEN = os.environ.get("HF_TOKEN")

REPO_ID = "Kumarverma11/PocketFM_Audio"
SOURCE_FOLDER = "Transcripts_Episode_0001_to_0200"
TARGET_FOLDER = "Mr_Laawaris_Grammar_Fixed_V1" # Naya Folder
MODEL_ID = 'gemini-3.1-flash-lite'

hf_api = HfApi()

# Round-Robin API Client Generator
current_key_idx = 0
def get_next_client():
    global current_key_idx
    client = genai.Client(api_key=KEYS[current_key_idx])
    current_key_idx = (current_key_idx + 1) % len(KEYS)
    return client

# ==========================================
# 2. HIGH THINKING PROMPT & GLOSSARY
# ==========================================
system_prompt = """
Tum ek expert Hindi grammar aur copy editor ho. 
Niche ek episode ka text diya gaya hai. Tumhara kaam iski grammar, sentence structure, aur spelling mistakes ko deep reasoning ke sath theek karna hai.

STRICT RULES:
1. Kahani ka emotion, dialogue aur flow change nahi hona chahiye.
2. Glossary Rule: Agar kahani mein "नेवी (Navy)" shabd aaye, toh yaad rakhna yeh ek Gadi (Car) ka naam hai, color nahi. Ise context ke hisaab se theek karna.
3. Output mein sirf theek kiya hua text dena hai, koi extra charcha ya intro (jaise 'यहाँ आपका टेक्स्ट है') nahi likhna.
"""

# ==========================================
# 3. PROCESS & UPLOAD (BATCH OF 45)
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
                
            # 2. Process with next API Key
            client = get_next_client()
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=f"{system_prompt}\n\nTEXT:\n{raw_text}",
                config=types.GenerateContentConfig(temperature=0.2)
            )
            fixed_text = response.text.strip()
            
            # 3. Save locally temporarily
            temp_save_path = f"./{filename}"
            with open(temp_save_path, 'w', encoding='utf-8') as f:
                f.write(fixed_text)
                
            # 4. Upload immediately to New Folder on Hugging Face
            hf_api.upload_file(
                path_or_fileobj=temp_save_path,
                path_in_repo=f"{TARGET_FOLDER}/{filename}",
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN,
                commit_message=f"Grammar fixed for {filename}"
            )
            print(f"✅ {filename} fixed and uploaded.")
            os.remove(temp_save_path) # Delete to keep runner clean
            
        except Exception as e:
            print(f"❌ Error on {filename}: {e}")
            
    # Har batch ke baad limit reset karne ke liye 60 seconds ka wait
    print("⏳ Batch complete. Waiting 60 seconds for API rate limits to reset...")
    time.sleep(60)

print("🎉 All 200 episodes processed and safely uploaded!")

