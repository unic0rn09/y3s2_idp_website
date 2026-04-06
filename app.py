import os
import json 
import tempfile
import subprocess
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# 🚀 IMPORT FROM YOUR NEW AI ENGINE FILE
from ai_engine import transcribe_wav, run_pipeline, clear_old_audio, _to_safe_visit_id, INSTANCE_FOLDER, TARGET_SR

app = Flask(__name__)
app.secret_key = "super_secret_key"

# Configure SQLite Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///scribe.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Models
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
    status = db.Column(db.String(20), default='Waiting') # Waiting, Consulting, Draft, Completed
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


# --- DATA COLLECTION HELPER ---
def save_patient_data_to_folder(patient):
    """Saves patient information into a structured folder hierarchy:
       instance/patient_records/IC_NUMBER/DATE_VISITED/file.json"""
       
    # DO NOT save data if it is the Mock Test Patient
    if patient.ic == '999999-99-9999':
        return

    # 1. Start inside the local 'instance' folder
    base_dir = os.path.join("instance", "patient_records")
    
    # 2. Extract the exact date the patient visited
    date_visited = patient.date_added.strftime("%Y-%m-%d")
    
    # 3. Create the nested directory: instance/patient_records/IC/Date/
    target_dir = os.path.join(base_dir, patient.ic, date_visited)
    os.makedirs(target_dir, exist_ok=True)
    
    # 4. Intelligently name the file based on the stage of the visit
    if patient.status == 'Waiting':
        filename = "1_intake_and_vitals.json"
    elif patient.status == 'Draft':
        filename = "2_consultation_summary.json"
    elif patient.status == 'Completed':
        filename = "3_final_clinical_note.json"
    else:
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"record_{timestamp}.json"
        
    # 5. Structure the data to be saved
    archive_data = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            "room": patient.room,
            "status": patient.status
        },
        "patient": {"name": patient.name, "ic": patient.ic, "age": patient.age},
        "vitals": {"bp": patient.bp, "hr": patient.hr, "temp": patient.temp, "rr": patient.rr},
        "clinical_notes": {
            "cc": patient.cc, "hpi": patient.hpi, "pmh": patient.pmh, 
            "meds": patient.meds, "allergies": patient.allergies,
            "social": {
                "occupation": patient.sh_occupation, 
                "living": patient.sh_living, 
                "smoking": patient.sh_smoking, 
                "alcohol": patient.sh_alcohol,
                "activity": patient.sh_activity,
                "diet": patient.sh_diet,
                "sleep": patient.sh_sleep,
                "others": patient.sh_others
            }
        },
        "raw_transcription": patient.transcription
    }
    
    # 6. Save the file into the new folder structure
    with open(file_path, 'w') as f:
        json.dump(archive_data, f, indent=4)
#--------------------------------------------------------#

def get_rooms_data():
    rooms = []
    for i in range(1, 6):
        room_num_str = str(i)
        doc = User.query.filter_by(role='doctor', room=room_num_str).first()
        
        # Exclude the test patient (IC: 999999-99-9999) from nurse dashboard entirely
        patients_query = Patient.query.filter(
            Patient.room == room_num_str, 
            Patient.status.in_(['Waiting', 'Draft']),
            Patient.ic != '999999-99-9999' # Filter out mock data
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

# --- NURSE ROUTES ----------
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
    # Exclude test patient from nurse history
    history = Patient.query.filter(Patient.status=='Completed', Patient.ic != '999999-99-9999').order_by(Patient.id.desc()).all()
    return render_template('patient_registration.html', rooms=get_rooms_data(), history=history)

@app.route('/nurse/rooms')
def all_rooms(): return render_template('all_rooms.html', rooms=get_rooms_data())

@app.route('/nurse/history')
def patient_history(): 
    # Exclude test patient from global history
    history = Patient.query.filter(Patient.ic != '999999-99-9999').order_by(Patient.id.desc()).all()
    return render_template('patient_history.html', history=history, rooms=get_rooms_data())

@app.route('/delete_patient/<patient_id>', methods=['POST'])
def delete_patient(patient_id):
    # Find the patient in the database
    patient = Patient.query.get_or_404(patient_id)
    
    # Delete the record and save changes
    db.session.delete(patient)
    db.session.commit()
    
    # Refresh the page automatically so the patient disappears from the table
    return redirect(request.referrer)

# --- DOCTOR ROUTES -------

# 🚀 THE REAL-TIME AUDIO ROUTE 
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
        if os.path.getsize(temp_webm_path) < 5000:
            return jsonify({'text': ''}), 200

        subprocess.run(['ffmpeg', '-y', '-i', temp_webm_path, '-ar', str(TARGET_SR), '-ac', '1', final_wav_path], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # Calls the function from ai_engine.py
        text = transcribe_wav(final_wav_path)
        return jsonify({'text': text}), 200
    except Exception as e:
        print(f"⚠️ Transcription Error: {e}")
        return jsonify({'text': ''}), 200
    finally:
        if os.path.exists(temp_webm_path): os.remove(temp_webm_path)

# 🚀 THE FAST-TRACK FINALIZATION ROUTE
@app.route('/doctor/finish_live/<patient_id>', methods=['POST'])
def finish_live(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    
    # 1. Grab the raw text from the screen
    frontend_raw_text = request.form.get('transcription', '')
    
    # 2. 🚀 RUN YOUR NEW PIPELINE
    pipeline_result = run_pipeline(frontend_raw_text)
    
    if pipeline_result and pipeline_result["status"] != "NON_MEDICAL":
        # Save the English translation as the main transcript
        patient.transcription = pipeline_result["translation"]
        
        # 3. 🚀 AUTO-FILL THE DATABASE FIELDS!
        clerking_data = pipeline_result["clerking_json"]
        
        patient.cc = clerking_data.get("chief_complaint", "")
        
        # Convert JSON arrays into readable text for the database
        patient.hpi = "\n".join([item["finding"] for item in clerking_data.get("history_of_present_illness", []) if isinstance(item, dict)])
        patient.pmh = "\n".join([item["finding"] for item in clerking_data.get("past_medical_history", []) if isinstance(item, dict)])
        patient.meds = "\n".join([item["finding"] for item in clerking_data.get("medication_history", []) if isinstance(item, dict)])
        
        # Handle Allergies (which is a dictionary in your new JSON)
        allergies = clerking_data.get("allergies", {})
        patient.allergies = allergies.get("status", "Not mentioned")

        # Handle Family and Social History
        family_hist = "\n".join([item["finding"] for item in clerking_data.get("family_history", []) if isinstance(item, dict)])
        social_hist = "\n".join([item["finding"] for item in clerking_data.get("social_history", []) if isinstance(item, dict)])
        patient.sh_others = f"Family: {family_hist}\nSocial: {social_hist}"
        
        # Optional: Save the verification verdict so the UI can warn the doctor!
        # patient.verification_status = pipeline_result["status"] 

    else:
        patient.transcription = "Error or Non-Medical content detected."

    patient.status = 'Draft'
    db.session.commit()
    clear_old_audio(str(patient.id))
    
    return redirect(url_for('consultation_summary', patient_id=patient.id))

@app.route('/doctor/dashboard')
def doctor_dashboard():
    if 'user_id' not in session: return redirect(url_for('login'))
    doctor = User.query.get(session['user_id'])
    
    # --- AUTO RESET MOCK PATIENT ---
    # Secretly wipe the mock patient data clean when the doctor returns to dashboard 
    # so it does not permanently stay in the database as "Completed"
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
    # Exclude test patient from "Completed Today"
    completed_today = [p for p in all_completed if p.date_added.strftime('%Y-%m-%d') == today_str and p.ic != '999999-99-9999']
    
    return render_template('doctor_dashboard.html', doctor=doctor, queue=queue, completed_today=completed_today)

@app.route('/doctor/toggle_status', methods=['POST'])
def toggle_status():
    if 'user_id' in session:
        doctor = User.query.get(session['user_id'])
        doctor.status = 'online' if doctor.status == 'offline' else 'offline'
        db.session.commit()
    return redirect(request.referrer)

#changes here#
@app.route('/doctor/consult/<patient_id>')
def live_consultation(patient_id):
    if 'user_id' not in session: return redirect(url_for('login'))
    patient = Patient.query.get_or_404(patient_id)
    patient.status = 'Consulting'
    db.session.commit()
    
    # Clean old audio using the imported function
    clear_old_audio(str(patient.id))
    
    doctor_user = db.session.get(User, session.get('user_id'))
    return render_template('live_consultation_session.html', patient=patient, doctor=doctor_user)
#changes here#
@app.route('/doctor/cancel_live/<patient_id>')
def cancel_live(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    patient.status = 'Waiting'
    db.session.commit()
    clear_old_audio(str(patient.id))
    return redirect(url_for('doctor_dashboard'))

# @app.route('/doctor/finish_live/<patient_id>', methods=['POST'])
# def finish_live(patient_id):
#     patient = Patient.query.get_or_404(patient_id)
#     patient.transcription = request.form.get('transcription', '')
#     patient.status = 'Draft'
#     db.session.commit()
#     return redirect(url_for('consultation_summary', patient_id=patient.id))

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
    
    # Trigger Folder-based Data Collection (Test patient is blocked inside this function)
    save_patient_data_to_folder(patient)
    
    patient.status = 'Completed'
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
    # Exclude test patient from doctor history
    history = Patient.query.filter(Patient.status=='Completed', Patient.ic != '999999-99-9999').order_by(Patient.id.desc()).all()
    return render_template('consultation_history.html', history=history, doctor=doctor)

@app.route('/doctor/mock_consultation')
def mock_consultation():
    if 'user_id' not in session: return redirect(url_for('login'))
    return render_template('mock_consultation.html', doctor=User.query.get(session['user_id']))



#-------help and feedback route------------
@app.route('/help_feedback')
def help_feedback():
    return render_template('help_feedback.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    topic = request.form.get('topic')
    message = request.form.get('message')
    
    # Send a flash message back to the user to confirm success
    flash("Thank you! Your feedback has been sent to the development team.", "success")
    return redirect(url_for('help_feedback'))
#--------------------------------------------

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(email='nurse@test.com').first():
            db.session.add(User(name="Nurse Joy", email='nurse@test.com', password_hash=generate_password_hash('nurse123'), role='nurse'))
        if not User.query.filter_by(email='doctor@test.com').first():
            db.session.add(User(name="Dr. Lim", email='doctor@test.com', password_hash=generate_password_hash('doctor123'), role='doctor', status='online', room='1'))
        db.session.commit()
        
        # ==============================================================
        # ⬇️ TEST PATIENT CREATION TOGGLE ⬇️
        # Uncomment the lines below to spawn the test patient on startup.
        # Leave them commented out to run the app normally.
        # ==============================================================
        test_patient = Patient.query.filter_by(ic='999999-99-9999').first()
        if not test_patient:
            test_patient = Patient(name="Auto Test Patient", ic="999999-99-9999", age="25", room="1", symptoms="Mock Test", status='Waiting')
            db.session.add(test_patient)
        else:
            test_patient.status = 'Waiting'
        db.session.commit()
        # ==============================================================
        
    app.run(host='0.0.0.0', port=5000, debug=True)