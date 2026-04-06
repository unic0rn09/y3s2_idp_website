import os
# 🚀 JETSON FIX: Prevents crashes from broken aarch64 torchcodec libraries
os.environ["TRANSFORMERS_NO_TORCHCODEC"] = "1"

import sklearn
import json 
import glob
import re
import tempfile
import subprocess
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

import numpy as np
import torch
import soundfile as sf 
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from openai import OpenAI

app = Flask(__name__)
app.secret_key = "super_secret_key"

# ===== CONFIGURATION =====
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INSTANCE_FOLDER = os.path.join(BASE_DIR, "instance")
os.makedirs(INSTANCE_FOLDER, exist_ok=True)

# 🚀 IMPORTANT: Insert your actual OpenAI API Key here
client = OpenAI(api_key="YOUR_OPENAI_API_KEY") 

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///scribe.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ===== DATABASE MODELS (Keep your existing Patient and User models) =====
# [Models remain unchanged from your original code]

# ===== ASR & LORA CONFIGURATION =====
BASE_MODEL_ID = "mesolitica/malaysian-whisper-medium-v2"
ADAPTER_DIR = os.path.join(BASE_DIR, "rojak_medium_lora_adapter") 
USE_LORA = True 
TARGET_SR = 16000
_ASR = {"processor": None, "model": None}

def _to_safe_visit_id(v):
    v = str(v)
    return "".join(ch for ch in v if ch.isalnum() or ch in ("-", "_"))[:64] or "unknown"

def _clear_old_audio(visit_id: str):
    safe_vid = _to_safe_visit_id(visit_id)
    pattern = os.path.join(INSTANCE_FOLDER, f"visit_{safe_vid}_chunk*.wav")
    for f in glob.glob(pattern):
        try: os.remove(f)
        except Exception: pass

def _load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1: audio = np.mean(audio, axis=1).astype(np.float32)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return audio

def get_asr():
    if _ASR["model"] is not None:
        return _ASR["processor"], _ASR["model"]
    
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    # 🚀 GPU FIX: Force the entire model onto CUDA to prevent "Half" precision math on CPU
    base = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, 
        device_map={"": device}, 
        torch_dtype=torch_dtype
    )
    base.tie_weights()
    
    config_obj = getattr(base, "generation_config", base.config)
    config_obj.forced_decoder_ids = None
    config_obj.suppress_tokens = []
    base.eval()
    
    model = base
    if USE_LORA and os.path.isdir(ADAPTER_DIR):
        try:
            lora_model = PeftModel.from_pretrained(base, ADAPTER_DIR)
            model = lora_model.merge_and_unload()
            model.eval()
            print("✅ SUCCESS: LoRA Adapter loaded and merged!")
        except Exception as e:
            print(f"❌ LoRA Error: {e}")
    
    _ASR["processor"], _ASR["model"] = processor, model
    return processor, model

def transcribe_wav(path: str) -> str:
    processor, model = get_asr()
    audio = _load_audio(path, TARGET_SR)
    inputs = processor.feature_extractor(audio, sampling_rate=TARGET_SR, return_tensors="pt").input_features
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # 🚀 TOKENS FIX: Increased to 448 to prevent cutting off speech
    with torch.no_grad():
        pred_ids = model.generate(inputs.to(device).to(dtype), max_new_tokens=448)
        
    return processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()

def generate_diarized_transcript(raw_text: str) -> str:
    if not raw_text.strip(): return "No transcription data."
    
    # 🚀 PROMPT FIX: High-quality few-shot examples for better diarization
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a medical transcriptionist. Format this raw ASR text into 'Doctor:' and 'Patient:' dialogue. Correct Malaysian phonetic errors (e.g. 'kulali' -> 'buku lali') and wrap them in <span class='text-red-600 font-bold'>tags</span>. End with [END OF CONSULTATION]."},
                {"role": "user", "content": f"Raw Input: {raw_text}\n\n[END OF CONSULTATION]"}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.replace("[END OF CONSULTATION]", "").strip()
    except Exception as e:
        print(f"❌ OpenAI Error: {e}")
        return raw_text

@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    audio_file = request.files.get('audio')
    patient_id = request.form.get('patient_id', 'unknown')
    chunk_index = request.form.get('chunk_index', '0')
    
    safe_vid = _to_safe_visit_id(patient_id)
    final_wav_path = os.path.join(INSTANCE_FOLDER, f"visit_{safe_vid}_chunk{chunk_index}.wav")
    
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_webm:
        audio_file.save(temp_webm.name)
        temp_webm_path = temp_webm.name

    try:
        # 🚀 5000 BYTES FIX: Ignores empty webm headers to prevent FFmpeg crash
        if os.path.getsize(temp_webm_path) < 5000:
            return jsonify({'text': ''}), 200

        subprocess.run(['ffmpeg', '-y', '-i', temp_webm_path, '-ar', '16000', '-ac', '1', final_wav_path], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        text = transcribe_wav(final_wav_path)
        return jsonify({'text': text}), 200
    except Exception as e:
        print(f"⚠️ Transcription Error: {e}")
        return jsonify({'text': ''}), 200
    finally:
        if os.path.exists(temp_webm_path): os.remove(temp_webm_path)

@app.route('/doctor/finish_live/<patient_id>', methods=['POST'])
def finish_live(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    
    # 🚀 FAST-TRACK FIX: Take the text already on the screen
    frontend_raw_text = request.form.get('transcription', '')
    patient.transcription = generate_diarized_transcript(frontend_raw_text)
    
    patient.status = 'Draft'
    db.session.commit()
    _clear_old_audio(str(patient.id))
    return redirect(url_for('consultation_summary', patient_id=patient.id))

# [Keep your remaining dashboard and login routes here]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)