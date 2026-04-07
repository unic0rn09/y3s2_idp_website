import os
# 🚀 JETSON FIX: Prevents crashes from broken aarch64 torchcodec libraries
os.environ["TRANSFORMERS_NO_TORCHCODEC"] = "1"

import glob
import json
import time
import logging
import torch
import numpy as np
import soundfile as sf
import librosa
from datetime import datetime
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from openai import OpenAI
from dotenv import load_dotenv

# Optional: For Step 3 (Who Engine)
# pip install pyannote.audio
from pyannote.audio import Pipeline

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================
# CONFIGURATIONS
# ============================================


# Define where audio files will be temporarily saved
INSTANCE_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "instance")
os.makedirs(INSTANCE_FOLDER, exist_ok=True)

def _to_safe_visit_id(patient_id):
    """Ensures the patient ID is safe to use as a filename."""
    return str(patient_id).replace(" ", "_").replace("/", "_")

def clear_old_audio(patient_id):
    """Deletes temporary audio chunks after the consultation is done."""
    safe_vid = _to_safe_visit_id(patient_id)
    # Find all temporary .wav files associated with this patient
    pattern = os.path.join(INSTANCE_FOLDER, f"visit_{safe_vid}_*.wav")
    for file_path in glob.glob(pattern):
        try:
            os.remove(file_path)
            print(f"🧹 Cleaned up: {file_path}")
        except OSError:
            pass
            
BASE_MODEL_ID = "mesolitica/malaysian-whisper-medium-v2"
ADAPTER_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "rojak_medium_lora_adapter")
TARGET_SR = 16000 #sampling rate of audio 

# ============================================
# 1. ASR ENGINE (The "What" Engine)
# ============================================

def get_asr():
    """Optimized for Jetson 16GB VRAM"""
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use float16 to save 50% VRAM on Jetson
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, 
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    
    if os.path.isdir(ADAPTER_DIR):
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
        model = model.merge_and_unload()
    
    model.eval()
    return processor, model
    
def transcribe_wav(audio_path):
    """
    Step 1.5: Ultra-fast transcription for live 5-second UI chunks.
    Skips Pyannote and Timestamps to ensure real-time speed.
    """
    processor, model = get_asr()
    audio = _load_audio(audio_path)
    
    inputs = processor(audio, return_tensors="pt", sampling_rate=TARGET_SR).to("cuda", torch.float16)
    
    with torch.no_grad():
        generated_ids = model.generate(inputs.input_features, max_new_tokens=448)
    
    # Decode strictly to text
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

def transcribe_with_timestamps(audio_path):
    """
    Step 2: Returns the Metadata Log with exact milliseconds.
    If using a stronger GPU, change 'medium' to 'large-v3'.
    """
    processor, model = get_asr()
    audio = _load_audio(audio_path)
    
    inputs = processor(audio, return_tensors="pt", sampling_rate=TARGET_SR).to("cuda", torch.float16)
    
    with torch.no_grad():
        # return_timestamps=True is the 'Glue' for Pyannote
        generated_ids = model.generate(
            inputs.input_features, 
            return_timestamps=True, 
            max_new_tokens=448
        )
    
    # This generates a list of chunks with {'text': '...', 'timestamp': (start, end)}
    result = processor.tokenizer._decode_asr(
        generated_ids[0], 
        return_timestamps=True, 
        return_language=False
    )
    
    # Standardizing the Metadata Log format
    metadata_log = []
    for chunk in result['chunks']:
        metadata_log.append({
            "text": chunk['text'].strip(),
            "start": chunk['timestamp'][0],
            "end": chunk['timestamp'][1]
        })
    
    # 🧹 JETSON VRAM CLEANUP: Clear Whisper to make room for Pyannote
    del model
    torch.cuda.empty_cache()
    
    return metadata_log

def _load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1: 
        audio = np.mean(audio, axis=1).astype(np.float32)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return audio
# ============================================
# 2. DIARIZATION ENGINE (The "Who" Engine)
# ============================================

def get_speaker_map(audio_path):
    """
    Step 3: Pyannote Full Scan.
    Requires an HF_TOKEN in your .env for Pyannote models.
    """
    # Load Pyannote (ensure it's on CUDA)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HF_TOKEN")
    ).to(torch.device("cuda"))

    diarization = pipeline(audio_path)
    
    speaker_map = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_map.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    
    # 🧹 JETSON VRAM CLEANUP
    del pipeline
    torch.cuda.empty_cache()
    
    return speaker_map

# ============================================
# 3. TEMPORAL INTERSECTION (The "Glue")
# ============================================

def align_text_to_speakers(metadata_log, speaker_map):
    diarized_transcript = []
    
    for text_block in metadata_log:
        t_mid = (text_block["start"] + text_block["end"]) / 2
        
        assigned_speaker = "Unknown"
        for spk in speaker_map:
            if spk["start"] <= t_mid <= spk["end"]:
                assigned_speaker = spk["speaker"]
                break
        
        diarized_transcript.append(f"{assigned_speaker}: {text_block['text']}")
    
    return "\n".join(diarized_transcript)

# ============================================
# 4. GPT ENGINES (Deduction, Translation, Structuring)
# ============================================

def process_clinical_tasks(diarized_text, mode="all"):
    """
    Flexible GPT handler.
    'diarized_text' is the raw Speaker_00/Speaker_01 output.
    """
    # 1. Identify Roles (Doctor/Patient)
    role_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "system", 
            "content": "Identify Speaker_00 and Speaker_01 as 'Doctor' or 'Patient' based on context. Return the full text with correct labels."
        }, {"role": "user", "content": diarized_text}]
    )
    labeled_text = role_response.choices[0].message.content

    # 2. Translation (If requested by UI)
    translation = None
    if mode in ["all", "translate"]:
        trans_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Translate this medical transcript to formal English."}, 
                      {"role": "user", "content": labeled_text}]
        )
        translation = trans_resp.choices[0].message.content

    # 3. Structuring (CC, HPI, etc.)
    structuring = None
    if mode in ["all", "structure"]:
        struct_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": "Extract into JSON: chief_complaint, history_of_present_illness, social_history."}, 
                      {"role": "user", "content": translation or labeled_text}]
        )
        structuring = json.loads(struct_resp.choices[0].message.content)

    return labeled_text, translation, structuring

# ============================================
# MASTER EXECUTION
# ============================================

def run_post_consultation_pipeline(full_audio_path):
    print("⏳ Running ASR (The What)...")
    metadata_log = transcribe_with_timestamps(full_audio_path)
    
    print("⏳ Running Pyannote (The Who)...")
    speaker_map = get_speaker_map(full_audio_path)
    
    print("⏳ Gluing Timeline...")
    raw_diarized = align_text_to_speakers(metadata_log, speaker_map)
    
    print("⏳ GPT Clinical Processing...")
    labeled, translated, structured = process_clinical_tasks(raw_diarized)
    
    return {
        "ui_left_box": labeled,        # Labeled Doctor/Patient (Rojak)
        "ui_translate_box": translated, # Formal English
        "ui_right_box": structured      # Structured JSON
    }
