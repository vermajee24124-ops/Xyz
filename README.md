# Veda Pipeline

Code-only public GitHub workflow.

## Secrets
- HF_TOKEN
- NVIDIA_API_KEY
- OPENROUTER_API_KEY (optional)
- GEMINI_API_KEY_1 (optional)
- GEMINI_API_KEY_2 (optional)
- GEMINI_API_KEY_3 (optional)

## Source
- Repo: Kumarverma11/PocketFM_Audio
- Folder: Transcripts_Episode_0001_to_0200

## Output
- Folder: Veda_Final_Training_Export_0001_to_0200

## Behavior
- Track A: clean transcript
- Track B: extraction-only JSON
- Batch upload every 20 episodes
- Auto-resume by checking Hugging Face output files
