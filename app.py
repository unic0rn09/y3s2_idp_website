import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

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
    sh_smoke = db.Column(db.Boolean, default=False)
    sh_alcohol = db.Column(db.Boolean, default=False)
    sh_living = db.Column(db.String(50), default="")
    sh_others = db.Column(db.String(255), default="")

def get_rooms_data():
    rooms = []
    for i in range(1, 6):
        room_num_str = str(i)
        doc = User.query.filter_by(role='doctor', room=room_num_str).first()
        patients_query = Patient.query.filter(Patient.room==room_num_str, Patient.status.in_(['Waiting', 'Draft'])).order_by(Patient.priority.desc(), Patient.id.asc()).all()
        
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

# --- NURSE ROUTES ---
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
def patient_registration(): return render_template('patient_registration.html', rooms=get_rooms_data(), history=Patient.query.filter_by(status='Completed').order_by(Patient.id.desc()).all())
@app.route('/nurse/rooms')
def all_rooms(): return render_template('all_rooms.html', rooms=get_rooms_data())
@app.route('/nurse/history')
def patient_history(): return render_template('patient_history.html', history=Patient.query.order_by(Patient.id.desc()).all(), rooms=get_rooms_data())


# --- DOCTOR ROUTES ---
@app.route('/doctor/dashboard')
def doctor_dashboard():
    if 'user_id' not in session: return redirect(url_for('login'))
    doctor = User.query.get(session['user_id'])
    
    # Queue includes Waiting, Consulting (if accidentally stuck), and Draft patients
    queue = Patient.query.filter(Patient.room==doctor.room, Patient.status.in_(['Waiting', 'Consulting', 'Draft'])).order_by(Patient.priority.desc(), Patient.id.asc()).all()
    
    # Completed Today Logic
    today_str = datetime.now().strftime('%Y-%m-%d')
    all_completed = Patient.query.filter_by(room=doctor.room, status='Completed').order_by(Patient.id.desc()).all()
    completed_today = [p for p in all_completed if p.date_added.strftime('%Y-%m-%d') == today_str]
    
    return render_template('doctor_dashboard.html', doctor=doctor, queue=queue, completed_today=completed_today)

@app.route('/doctor/toggle_status', methods=['POST'])
def toggle_status():
    if 'user_id' in session:
        doctor = User.query.get(session['user_id'])
        doctor.status = 'online' if doctor.status == 'offline' else 'offline'
        db.session.commit()
    return redirect(request.referrer)

@app.route('/doctor/consult/<int:patient_id>')
def live_consultation(patient_id):
    if 'user_id' not in session: return redirect(url_for('login'))
    patient = Patient.query.get_or_404(patient_id)
    patient.status = 'Consulting'
    db.session.commit()
    return render_template('live_consultation_session.html', patient=patient, doctor=User.query.get(session['user_id']))

@app.route('/doctor/cancel_live/<int:patient_id>')
def cancel_live(patient_id):
    # If doctor quits live session, revert to Waiting
    patient = Patient.query.get_or_404(patient_id)
    patient.status = 'Waiting'
    db.session.commit()
    return redirect(url_for('doctor_dashboard'))

@app.route('/doctor/finish_live/<int:patient_id>', methods=['POST'])
def finish_live(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    patient.transcription = request.form.get('transcription', '')
    patient.status = 'Draft'
    db.session.commit()
    return redirect(url_for('consultation_summary', patient_id=patient.id))

@app.route('/doctor/summary/<int:patient_id>')
def consultation_summary(patient_id):
    if 'user_id' not in session: return redirect(url_for('login'))
    return render_template('consultation_summary.html', patient=Patient.query.get_or_404(patient_id), doctor=User.query.get(session['user_id']))

@app.route('/doctor/save_draft/<int:patient_id>', methods=['POST'])
def save_draft(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    # Save all form data to database as a draft
    patient.transcription = request.form.get('transcription', '')
    patient.cc = request.form.get('cc', '')
    patient.hpi = request.form.get('hpi', '')
    patient.pmh = request.form.get('pmh', '')
    patient.meds = request.form.get('meds', '')
    patient.allergies = request.form.get('allergies', '')
    patient.sh_smoke = True if request.form.get('sh_smoke') else False
    patient.sh_alcohol = True if request.form.get('sh_alcohol') else False
    patient.sh_living = request.form.get('sh_living', '')
    patient.sh_others = request.form.get('sh_others', '')
    patient.status = 'Draft'
    db.session.commit()
    return redirect(url_for('doctor_dashboard'))

@app.route('/doctor/generate_report/<int:patient_id>', methods=['POST'])
def generate_report(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    # Save final form data
    patient.cc = request.form.get('cc', '')
    patient.hpi = request.form.get('hpi', '')
    patient.pmh = request.form.get('pmh', '')
    patient.meds = request.form.get('meds', '')
    patient.allergies = request.form.get('allergies', '')
    patient.sh_smoke = True if request.form.get('sh_smoke') else False
    patient.sh_alcohol = True if request.form.get('sh_alcohol') else False
    patient.sh_living = request.form.get('sh_living', '')
    patient.sh_others = request.form.get('sh_others', '')
    patient.status = 'Completed'
    db.session.commit()
    return redirect(url_for('final_medical_note', patient_id=patient.id))

@app.route('/doctor/report/<int:patient_id>')
def final_medical_note(patient_id):
    if 'user_id' not in session: return redirect(url_for('login'))
    return render_template('final_medical_note.html', patient=Patient.query.get_or_404(patient_id), doctor=User.query.get(session['user_id']))

@app.route('/doctor/history')
def consultation_history():
    if 'user_id' not in session: return redirect(url_for('login'))
    doctor = User.query.get(session['user_id'])
    history = Patient.query.filter_by(status='Completed').order_by(Patient.id.desc()).all()
    return render_template('consultation_history.html', history=history, doctor=doctor)

@app.route('/doctor/mock_consultation')
def mock_consultation():
    if 'user_id' not in session: return redirect(url_for('login'))
    return render_template('mock_consultation.html', doctor=User.query.get(session['user_id']))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(email='nurse@test.com').first():
            db.session.add(User(name="Nurse Joy", email='nurse@test.com', password_hash=generate_password_hash('nurse123'), role='nurse'))
        if not User.query.filter_by(email='doctor@test.com').first():
            db.session.add(User(name="Dr. Lim", email='doctor@test.com', password_hash=generate_password_hash('doctor123'), role='doctor', status='online', room='1'))

# ---------------------------------- CREATE TEST PATIENT --------------------------#
        # Checks if the test patient already exists by IC so it doesn't create duplicates
# --- SEED TEST PATIENT ---
        test_patient = Patient.query.filter_by(ic='999999-99-9999').first()
        
        if not test_patient:
            # Create the test patient if they don't exist at all
            test_patient = Patient(
                name="Auto Test Patient",
                ic="999999-99-9999",
                age="25",
                room="1",  
                symptoms="Automated test patient for development."<br>"Experiencing mild headache and fatigue.",
                priority=False,
                status='Waiting',
                bp="120/80",
                hr="72",
                temp="37.2",
                rr="18"
            )
            db.session.add(test_patient)
        else:
            # If they DO exist, reset them back to a fresh "Waiting" state for new testing
            test_patient.status = 'Waiting'
            test_patient.room = '1'
            test_patient.transcription = ""
            test_patient.cc = ""
            test_patient.hpi = ""
            test_patient.pmh = ""
            test_patient.meds = ""
            test_patient.allergies = ""
            test_patient.sh_others = ""
            
        db.session.commit()
        # -------------------------
        
        db.session.commit()
    app.run(host='0.0.0.0', port=5000, debug=True)