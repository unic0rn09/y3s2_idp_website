import sklearn
import os
import json 
import glob
import re
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# LoRa adapter & Audio libraries
import tempfile
import numpy as np
import subprocess
import torch
import soundfile as sf 
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# OpenAI Library
from openai import OpenAI

app = Flask(__name__)
app.secret_key = "super_secret_key"

# ===== DIRECTORIES & OPENAI SETUP =====
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INSTANCE_FOLDER = os.path.join(BASE_DIR, "instance")
os.makedirs(INSTANCE_FOLDER, exist_ok=True)

#API KEY HERE!!!!!!!!!!!!!################
# #################IMPORTANT#####################
client = OpenAI(api_key="YOUR_OPENAI_API_KEY") 

# Configure SQLite Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///scribe.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ===== DATABASE MODELS =====
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    status = db.Column(db.String(20), default='offline')
    room = db.Column(db.String(20), nullable=True)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    ic = db.Column(db.String(20), nullable=False)
    age = db.Column(db.String(20))
    room = db.Column(db.String(20))
    symptoms = db.Column(db.Text)
    priority = db.Column(db.Boolean, default=False)
    status = db.Column(db.String(20), default='Waiting')
    date_added = db.Column(db.DateTime, default=datetime.utcnow)
    bp = db.Column(db.String(20), default="-")
    hr = db.Column(db.String(20), default="-")
    temp = db.Column(db.String(20), default="-")
    rr = db.Column(db.String(20), default="-") 
    
    # Clinical Draft & Report Fields
    transcription = db.Column(db.Text, default="")
    cc = db.Column(db.Text, default="")
    hpi = db.Column(db.Text, default="")
    pmh = db.Column(db.Text, default="")
    meds = db.Column(db.Text, default="")
    allergies = db.Column(db.Text, default="")
    
    # Updated Structured Social History Fields
    sh_occupation = db.Column(db.String(255), default="")
    sh_living = db.Column(db.String(255), default="")
    sh_smoking = db.Column(db.String(255), default="")
    sh_alcohol = db.Column(db.String(255), default="")
    sh_activity = db.Column(db.String(255), default="")
    sh_diet = db.Column(db.String(255), default="")
    sh_sleep = db.Column(db.String(255), default="")
    sh_others = db.Column(db.String(255), default="")


# ===== ASR & LORA CONFIGURATION =====
BASE_MODEL_ID = "mesolitica/malaysian-whisper-medium-v2"
# Lowercase to match your github folder structure exactly
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
        try:
            os.remove(f)
        except Exception:
            pass

def _load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return audio

def get_asr():
    if _ASR["model"] is not None:
        return _ASR["processor"], _ASR["model"]
        
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    try:
        base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID, device_map="auto", torch_dtype=torch_dtype)
    except Exception:
        base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID, torch_dtype=torch_dtype)
        base = base.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Safely handle config attribute differences between Transformers versions
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
            print("✅ SUCCESS: Bahasa Rojak LoRA Adapter loaded AND MERGED perfectly!")
        except Exception as e:
            print(f"❌ ERROR loading LoRA: {e}")
            model = base
    else:
        print("⚠️ WARNING: LoRA directory not found or USE_LORA is False. Using base model.")
        
    _ASR["processor"], _ASR["model"] = processor, model
    return processor, model

def transcribe_wav(path: str) -> str:
    processor, model = get_asr()
    audio = _load_audio(path, TARGET_SR)
    inputs = processor.feature_extractor(audio, sampling_rate=TARGET_SR, return_tensors="pt").input_features
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    inputs = inputs.to(model_device).to(model_dtype)
    
    gen_kwargs = dict(
        input_features=inputs, 
        max_new_tokens=256,
        condition_on_prev_tokens=False,
        repetition_penalty=1.1
    )
    
    with torch.no_grad():
        pred_ids = model.generate(**gen_kwargs)
        
    text = processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()
    return text

def generate_diarized_transcript(raw_text: str) -> str:
    if not raw_text or raw_text.strip() == "":
        return "No transcription data found."
        
    print("🧠 Calling OpenAI API for Semantic Diarization...")
    
    # Apply the Anchor Rule directly to the text passed from the frontend
    raw_text = raw_text.strip() + "\n\n[END OF CONSULTATION]"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a strict medical transcriptionist. Format this raw, messy ASR transcript into a clean 'Doctor:' and 'Patient:' dialogue. "
                        "CRITICAL RULES:\n"
                        "1. DIARIZATION: Accurately assign speakers. Pay attention to context.\n"
                        "2. MEDICAL & PHONETIC CORRECTIONS: The ASR makes severe phonetic mistakes with Malaysian accents. You MUST correct them based on clinical context "
                        "(e.g., changing 'very cold winds' to 'varicose veins', 'weakness insufficient' to 'venous insufficiency', 'kulali' to 'buku lali').\n"
                        "3. MANDATORY HTML TAGS: Every single time you correct an ASR misheard word, you ABSOLUTELY MUST wrap the corrected word in exact HTML tags. "
                        "Example: <span class='text-red-600 font-bold'>varicose veins</span>.\n"
                        "4. THE ANCHOR RULE: The raw text ends with the phrase [END OF CONSULTATION]. You MUST process every single word of the transcript and you are forbidden from stopping until you output the phrase [END OF CONSULTATION] exactly as it appears."
                    )
                },
                {"role": "user", "content": "Raw Input:\n" + raw_text}
            ],
            temperature=0.0,
            max_tokens=4000
        )
        final_transcript = response.choices[0].message.content.strip()
        final_transcript = final_transcript.replace("[END OF CONSULTATION]", "").strip()
    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        return f"OpenAI formatting failed. Raw text:\n\n{raw_text}"
        
    return final_transcript


# ===== REAL-TIME TRANSCRIPTION ROUTE =====
# ===== REAL-TIME TRANSCRIPTION ROUTE =====
@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
        
    audio_file = request.files['audio']
    patient_id = request.form.get('patient_id', 'unknown')
    fallback_index = str(int(datetime.now().timestamp() * 1000))
    chunk_index = request.form.get('chunk_index', fallback_index)
    
    safe_vid = _to_safe_visit_id(patient_id)
    final_wav_name = f"visit_{safe_vid}_chunk{chunk_index}.wav"
    final_wav_path = os.path.join(INSTANCE_FOLDER, final_wav_name)
    
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_webm:
        audio_file.save(temp_webm.name)
        temp_webm_path = temp_webm.name

    try:
        # 🚀 PREVENT CRASH: If file is basically empty (under 5KB), ignore it and return 200 OK
        if os.path.getsize(temp_webm_path) < 5000:
            return jsonify({'text': ''}), 200

        subprocess.run(['ffmpeg', '-y', '-i', temp_webm_path, '-ar', str(TARGET_SR), '-ac', '1', final_wav_path], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        text = transcribe_wav(final_wav_path)
        return jsonify({'text': text}), 200
        
    except subprocess.CalledProcessError:
        # 🚀 FIX: Ignore FFmpeg crash on the final chopped chunk
        return jsonify({'text': ''}), 200
    except Exception as e:
        print("Transcription Error:", e)
        return jsonify({'text': ''}), 200
    finally:
        if os.path.exists(temp_webm_path): os.remove(temp_webm_path)

# ===== DATA COLLECTION HELPER =====
def save_patient_data_to_folder(patient):
    try:
        if patient.ic == '999999-99-9999': return
        base_dir = os.path.join(INSTANCE_FOLDER, "patient_records")
        if isinstance(patient.date_added, datetime):
            date_visited = patient.date_added.strftime("%Y-%m-%d")
        elif patient.date_added:
            date_visited = str(patient.date_added).split(' ')[0]
        else:
            date_visited = datetime.now().strftime("%Y-%m-%d")
            
        safe_ic = str(patient.ic).replace('/', '-').replace('\\', '-').strip() if patient.ic else "UNKNOWN_IC"
        target_dir = os.path.join(base_dir, safe_ic, date_visited)
        os.makedirs(target_dir, exist_ok=True)
        
        if patient.status == 'Waiting': filename = "1_intake_and_vitals.json"
        elif patient.status == 'Draft': filename = "2_consultation_summary.json"
        elif patient.status == 'Completed': filename = "3_final_clinical_note.json"
        else:
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"record_{timestamp}.json"
            
        file_path = os.path.join(target_dir, filename)
        
        archive_data = {
            "metadata": {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "room": patient.room, "status": patient.status},
            "patient": {"name": patient.name, "ic": patient.ic, "age": patient.age},
            "vitals": {"bp": patient.bp, "hr": patient.hr, "temp": patient.temp, "rr": patient.rr},
            "clinical_notes": {
                "cc": patient.cc, "hpi": patient.hpi, "pmh": patient.pmh, 
                "meds": patient.meds, "allergies": patient.allergies,
                "social": {
                    "occupation": patient.sh_occupation, "living": patient.sh_living, "smoking": patient.sh_smoking, 
                    "alcohol": patient.sh_alcohol, "activity": patient.sh_activity, "diet": patient.sh_diet,
                    "sleep": patient.sh_sleep, "others": patient.sh_others
                }
            },
            "raw_transcription": patient.transcription
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(archive_data, f, indent=4)
    except Exception as e:
        print(f"Non-critical Error saving JSON file to folder: {e}")

def get_rooms_data():
    rooms = []
    for i in range(1, 6):
        room_num_str = str(i)
        doc = User.query.filter_by(role='doctor', room=room_num_str).first()
        patients_query = Patient.query.filter(
            Patient.room == room_num_str, 
            Patient.status.in_(['Waiting', 'Draft']),
            Patient.ic != '999999-99-9999' 
        ).order_by(Patient.priority.desc(), Patient.id.asc()).all()
        
        patient_list = [{
            "id": p.id, "name": p.name, "ic": p.ic, "symptoms": p.symptoms, 
            "priority": p.priority, "status": p.status, "time": p.date_added.strftime('%H:%M'),
            "bp": p.bp, "hr": p.hr, "temp": p.temp, "rr": p.rr
        } for p in patients_query]
        
        if doc:
            status = "Waiting" if patient_list else "Available"
            rooms.append({"id": f"Room {i}", "room_num": room_num_str, "doctor": doc.name, "doctor_email": doc.email, "status": status, "patients": patient_list, "active": True})
        else:
            rooms.append({"id": f"Room {i}", "room_num": room_num_str, "doctor": "-", "doctor_email": "", "status": "Not Available", "patients": [], "active": False})
    return rooms

# ===== CORE ROUTES =====
@app.route('/')
def login(): return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():
    email = request.form.get('email')
    password = request.form.get('password')
    user = User.query.filter_by(email=email).first()

    if user and check_password_hash(user.password_hash, password) and user.role == request.form.get('role'):
        session['user_id'] = user.id 
        if user.role == 'nurse':
            return redirect(url_for('nurse_dashboard'))
        elif user.role == 'doctor':
            selected_room = request.form.get('room')
            if selected_room:
                user.room = selected_room
                user.status = 'online'
                db.session.commit()
            return redirect(url_for('doctor_dashboard'))
    return "Invalid email or password", 401

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

# ===== NURSE ROUTES =====
@app.route('/register_patient', methods=['POST'])
def register_patient():
    room = request.form.get('room')
    active_rooms = [r for r in get_rooms_data() if r['active']]
    if not active_rooms: return redirect(request.referrer)
    if room == 'auto' or not room: room = min(active_rooms, key=lambda r: len(r['patients']))['room_num']
            
    new_patient = Patient(
        name=request.form.get('name'), ic=request.form.get('ic'), age=request.form.get('age'), 
        room=room, symptoms=request.form.get('symptoms'), 
        priority=True if request.form.get('priority') == 'on' else False,
        bp=request.form.get('bp') or "-", hr=request.form.get('hr') or "-", 
        temp=request.form.get('temp') or "-", rr=request.form.get('rr') or "-"
    )
    db.session.add(new_patient)
    db.session.commit()
    return redirect(request.referrer)

@app.route('/nurse/dashboard')
def nurse_dashboard(): return render_template('nurse_dashboard.html', rooms=get_rooms_data())

@app.route('/nurse/registration')
def patient_registration(): 
    history = Patient.query.filter(Patient.status=='Completed', Patient.ic != '999999-99-9999').order_by(Patient.id.desc()).all()
    return render_template('patient_registration.html', rooms=get_rooms_data(), history=history)

@app.route('/nurse/rooms')
def all_rooms(): return render_template('all_rooms.html', rooms=get_rooms_data())

@app.route('/nurse/history')
def patient_history(): 
    history = Patient.query.filter(Patient.ic != '999999-99-9999').order_by(Patient.id.desc()).all()
    return render_template('patient_history.html', history=history, rooms=get_rooms_data())

@app.route('/delete_patient/<patient_id>', methods=['POST'])
def delete_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    db.session.delete(patient)
    db.session.commit()
    return redirect(request.referrer)

# ===== DOCTOR ROUTES =====
@app.route('/doctor/dashboard')
def doctor_dashboard():
    if 'user_id' not in session: return redirect(url_for('login'))
    doctor = User.query.get(session['user_id'])
    
    test_p = Patient.query.filter_by(ic='999999-99-9999').first()
    if test_p and test_p.status == 'Completed':
        test_p.status = 'Waiting'
        test_p.transcription = ""
        test_p.cc = ""
        test_p.hpi = ""
        test_p.pmh = ""
        test_p.meds = ""
        test_p.allergies = ""
        test_p.sh_occupation = ""
        test_p.sh_living = ""
        test_p.sh_smoking = ""
        test_p.sh_alcohol = ""
        test_p.sh_activity = ""
        test_p.sh_diet = ""
        test_p.sh_sleep = ""
        test_p.sh_others = ""
        db.session.commit()

    queue = Patient.query.filter(Patient.room==doctor.room, Patient.status.in_(['Waiting', 'Consulting', 'Draft'])).order_by(Patient.priority.desc(), Patient.id.asc()).all()
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    all_completed = Patient.query.filter_by(room=doctor.room, status='Completed').order_by(Patient.id.desc()).all()
    completed_today = [p for p in all_completed if p.date_added.strftime('%Y-%m-%d') == today_str and p.ic != '999999-99-9999']
    
    return render_template('doctor_dashboard.html', doctor=doctor, queue=queue, completed_today=completed_today)

@app.route('/doctor/toggle_status', methods=['POST'])
def toggle_status():
    if 'user_id' in session:
        doctor = User.query.get(session['user_id'])
        doctor.status = 'online' if doctor.status == 'offline' else 'offline'
        db.session.commit()
    return redirect(request.referrer)

@app.route('/doctor/consult/<patient_id>')
def live_consultation(patient_id):
    if 'user_id' not in session: return redirect(url_for('login'))
    patient = Patient.query.get_or_404(patient_id)
    patient.status = 'Consulting'
    db.session.commit()
    
    # Wipe old audio before starting a new session just to be safe
    _clear_old_audio(str(patient.id))
    
    doctor_user = db.session.get(User, session.get('user_id'))
    return render_template('live_consultation_session.html', patient=patient, doctor=doctor_user)

@app.route('/doctor/cancel_live/<patient_id>')
def cancel_live(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    patient.status = 'Waiting'
    db.session.commit()
    
    # Wipe audio if cancelled
    _clear_old_audio(str(patient.id))
    
    return redirect(url_for('doctor_dashboard'))

# 🚀 CHANGED THIS ROUTE TO TRIGGER OPENAI
# 🚀 CHANGED THIS ROUTE TO TRIGGER OPENAI FASTER
@app.route('/doctor/finish_live/<patient_id>', methods=['POST'])
def finish_live(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    
    # Grab the completed raw text directly from the frontend UI
    frontend_raw_text = request.form.get('transcription', '')
    
    # Pass it directly to the Diarization function (skipping ASR re-transcription)
    patient.transcription = generate_diarized_transcript(frontend_raw_text)
    
    patient.status = 'Draft'
    db.session.commit()
    
    _clear_old_audio(str(patient.id))
    
    return redirect(url_for('consultation_summary', patient_id=patient.id))

@app.route('/doctor/summary/<patient_id>')
def consultation_summary(patient_id):
    if 'user_id' not in session: return redirect(url_for('login'))
    return render_template('consultation_summary.html', patient=Patient.query.get_or_404(patient_id), doctor=User.query.get(session['user_id']))

@app.route('/doctor/save_draft/<patient_id>', methods=['POST'])
def save_draft(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    patient.transcription = request.form.get('transcription', '')
    patient.cc = request.form.get('cc', '')
    patient.hpi = request.form.get('hpi', '')
    patient.pmh = request.form.get('pmh', '')
    patient.meds = request.form.get('meds', '')
    patient.allergies = request.form.get('allergies', '')
    
    patient.sh_occupation = request.form.get('sh_occupation', '')
    patient.sh_living = request.form.get('sh_living', '')
    patient.sh_smoking = request.form.get('sh_smoking', '')
    patient.sh_alcohol = request.form.get('sh_alcohol', '')
    patient.sh_activity = request.form.get('sh_activity', '')
    patient.sh_diet = request.form.get('sh_diet', '')
    patient.sh_sleep = request.form.get('sh_sleep', '')
    patient.sh_others = request.form.get('sh_others', '')
    
    patient.status = 'Draft'
    db.session.commit()
    return redirect(url_for('doctor_dashboard'))

@app.route('/doctor/generate_report/<patient_id>', methods=['POST'])
def generate_report(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    patient.cc = request.form.get('cc', '')
    patient.hpi = request.form.get('hpi', '')
    patient.pmh = request.form.get('pmh', '')
    patient.meds = request.form.get('meds', '')
    patient.allergies = request.form.get('allergies', '')
    
    patient.sh_occupation = request.form.get('sh_occupation', '')
    patient.sh_living = request.form.get('sh_living', '')
    patient.sh_smoking = request.form.get('sh_smoking', '')
    patient.sh_alcohol = request.form.get('sh_alcohol', '')
    patient.sh_activity = request.form.get('sh_activity', '')
    patient.sh_diet = request.form.get('sh_diet', '')
    patient.sh_sleep = request.form.get('sh_sleep', '')
    patient.sh_others = request.form.get('sh_others', '')
    
    patient.status = 'Completed'
    save_patient_data_to_folder(patient)
    db.session.commit()
    
    return redirect(url_for('final_medical_note', patient_id=patient.id))

@app.route('/doctor/report/<patient_id>')
def final_medical_note(patient_id):
    if 'user_id' not in session: return redirect(url_for('login'))
    return render_template('final_medical_note.html', patient=Patient.query.get_or_404(patient_id), doctor=User.query.get(session['user_id']))

@app.route('/doctor/history')
def consultation_history():
    if 'user_id' not in session: return redirect(url_for('login'))
    doctor = User.query.get(session['user_id'])
    history = Patient.query.filter(Patient.status=='Completed', Patient.ic != '999999-99-9999').order_by(Patient.id.desc()).all()
    return render_template('consultation_history.html', history=history, doctor=doctor)

@app.route('/doctor/mock_consultation')
def mock_consultation():
    if 'user_id' not in session: return redirect(url_for('login'))
    return render_template('mock_consultation.html', doctor=User.query.get(session['user_id']))

# ===== HELP AND FEEDBACK ROUTE =====
@app.route('/help_feedback')
def help_feedback():
    return render_template('help_feedback.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    topic = request.form.get('topic')
    message = request.form.get('message')
    flash("Thank you! Your feedback has been sent to the development team.", "success")
    return redirect(url_for('help_feedback'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(email='nurse@test.com').first():
            db.session.add(User(name="Nurse Joy", email='nurse@test.com', password_hash=generate_password_hash('nurse123'), role='nurse'))
        if not User.query.filter_by(email='doctor@test.com').first():
            db.session.add(User(name="Dr. Lim", email='doctor@test.com', password_hash=generate_password_hash('doctor123'), role='doctor', status='online', room='1'))
        db.session.commit()
        
        test_patient = Patient.query.filter_by(ic='999999-99-9999').first()
        if not test_patient:
            test_patient = Patient(name="Auto Test Patient", ic="999999-99-9999", age="25", room="1", symptoms="Mock Test", status='Waiting')
            db.session.add(test_patient)
        else:
            test_patient.status = 'Waiting'
        db.session.commit()
        
    app.run(host='0.0.0.0', port=5000, debug=True)