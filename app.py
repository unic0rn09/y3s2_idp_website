import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "super_secret_key"

# Configure SQLite Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///scribe.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False) # 'nurse' or 'doctor'
    status = db.Column(db.String(20), default='offline')
    room = db.Column(db.String(20), nullable=True)

# --- Dummy Data ---
rooms_data = [
    {"id": "Room 1", "doctor": "Dr. Ahmad bin Hassan", "status": "Consulting", "patients": [{"name": "Tan Mei Ling", "ic": "950315-08-1234", "symptoms": "Fever, headache"}], "duration": "45 mins"},
    {"id": "Room 2", "doctor": "Dr. Sarah Lee", "status": "Available", "patients": [], "duration": "0 mins"},
]

# Fixed: Now using actual datetime objects so .strftime() works in the template
history_data = [
    {
        "id": 1,
        "name": "Tan Mei Ling", 
        "ic": "950315-08-1234", 
        "symptoms": "Fever, headache", 
        "doctor": "Dr. Ahmad bin Hassan", 
        "room": "Room 1", 
        "date_added": datetime(2024, 1, 15, 9, 15), 
        "duration": "15 mins"
    }
]

queue_data = [
    {"name": "Tan Mei Ling", "ic": "950315-08-1234", "symptoms": "Fever, headache", "status": "In Progress"},
    {"name": "Kumar Raj", "ic": "880722-14-5678", "symptoms": "Cough, sore throat", "status": "Waiting"}
]

# Dummy patient for the doctor views
dummy_patient = {
    "id": 1, "name": "Tan Mei Ling", "ic": "950315-08-1234", "age": "29", 
    "symptoms": "Fever, headache", "temp": "38.2°C", "hr": "88", "rr": "20", 
    "bp": "120/80", "room": "1", "date_added": datetime.now()
}

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():
    email = request.form.get('email')
    password = request.form.get('password')
    role_selected = request.form.get('role')

    user = User.query.filter_by(email=email).first()

    if user and check_password_hash(user.password_hash, password):
        if user.role == role_selected:
            if user.role == 'nurse':
                return redirect(url_for('nurse_dashboard'))
            elif user.role == 'doctor':
                return redirect(url_for('doctor_dashboard'))
        else:
            return "Role mismatch error", 403
    else:
        return "Invalid email or password", 401

@app.route('/logout')
def logout():
    return redirect(url_for('login'))

# --- NURSE ROUTES ---
@app.route('/nurse/dashboard')
def nurse_dashboard():
    return render_template('nurse_dashboard.html', rooms=rooms_data)

@app.route('/nurse/registration')
def patient_registration():
    return render_template('patient_registration.html', rooms=rooms_data)

@app.route('/nurse/rooms')
def all_rooms():
    return render_template('all_rooms.html', rooms=rooms_data)

@app.route('/nurse/history')
def patient_history():
    # Pass history_data directly; the nurse template currently uses string splits, so we should map it for safety if needed, 
    # but based on your previous nurse template, let's keep it compatible or let it ride. 
    # For now, we will pass the datetime objects.
    return render_template('patient_history.html', history=history_data, rooms=rooms_data)

# --- DOCTOR ROUTES ---
@app.route('/doctor/dashboard')
def doctor_dashboard():
    # Pass the doctor object so doctor_base.html can render the name and status
    current_doctor = User.query.filter_by(role='doctor').first()
    return render_template('doctor_dashboard.html', queue=queue_data, doctor=current_doctor)

@app.route('/doctor/live_session')
def live_consultation_session():
    current_doctor = User.query.filter_by(role='doctor').first()
    return render_template('live_consultation_session.html', doctor=current_doctor, patient=dummy_patient)

# Fixed route to handle the specific patient ID from live_session
@app.route('/doctor/summary/<int:patient_id>')
def consultation_summary(patient_id):
    current_doctor = User.query.filter_by(role='doctor').first()
    return render_template('consultation_summary.html', doctor=current_doctor, patient=dummy_patient)

# Added missing route: Handles the form submission to generate the final note
@app.route('/doctor/report/<int:patient_id>', methods=['GET', 'POST'])
def doctor_report(patient_id):
    current_doctor = User.query.filter_by(role='doctor').first()
    # Grab the form data submitted from the summary page
    form_data = request.form if request.method == 'POST' else {}
    return render_template('final_medical_note.html', doctor=current_doctor, patient=dummy_patient, form=form_data)

@app.route('/doctor/history')
def doctor_consultation_history():
    current_doctor = User.query.filter_by(role='doctor').first()
    return render_template('consultation_history.html', history=history_data, doctor=current_doctor)

# Added missing route: Handles the status toggle button in the sidebar
@app.route('/doctor/toggle_status', methods=['POST'])
def toggle_status():
    current_doctor = User.query.filter_by(role='doctor').first()
    if current_doctor:
        current_doctor.status = 'offline' if current_doctor.status == 'online' else 'online'
        db.session.commit()
    # Redirect back to wherever they clicked the button from
    return redirect(request.referrer or url_for('doctor_dashboard'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(email='nurse@test.com').first():
            db.session.add(User(name="Nurse Joy", email='nurse@test.com', password_hash=generate_password_hash('nurse123'), role='nurse'))
        if not User.query.filter_by(email='doctor@test.com').first():
            db.session.add(User(name="Dr. Lim", email='doctor@test.com', password_hash=generate_password_hash('doctor123'), role='doctor', status='online', room='1'))
        db.session.commit()
    app.run(host='0.0.0.0', port=5000, debug=False)