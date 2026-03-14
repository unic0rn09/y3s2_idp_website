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
    role = db.Column(db.String(20), nullable=False) # 'nurse' or 'doctor'
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
    
    # Vital Signs
    bp = db.Column(db.String(20), default="-")
    hr = db.Column(db.String(20), default="-")
    temp = db.Column(db.String(20), default="-")
    rr = db.Column(db.String(20), default="-") 

def get_rooms_data():
    rooms = []
    for i in range(1, 6):
        room_num_str = str(i)
        doc = User.query.filter_by(role='doctor', room=room_num_str).first()
        patients_query = Patient.query.filter_by(room=room_num_str, status='Waiting').order_by(Patient.priority.desc(), Patient.id.asc()).all()
        
        patient_list = [{
            "id": p.id, "name": p.name, "ic": p.ic, "symptoms": p.symptoms, 
            "priority": p.priority, "status": p.status, "time": p.date_added.strftime('%H:%M'),
            "bp": p.bp, "hr": p.hr, "temp": p.temp, "rr": p.rr
        } for p in patients_query]
        
        if doc:
            status = "Waiting" if patient_list else "Available"
            rooms.append({
                "id": f"Room {i}", "room_num": room_num_str, "doctor": doc.name, 
                "doctor_email": doc.email, "status": status, "patients": patient_list, "active": True
            })
        else:
            rooms.append({
                "id": f"Room {i}", "room_num": room_num_str, "doctor": "-", 
                "doctor_email": "", "status": "Not Available", "patients": [], "active": False
            })
    return rooms

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
            session['user_id'] = user.id # Securely log the user in
            
            if user.role == 'nurse':
                return redirect(url_for('nurse_dashboard'))
            
            elif user.role == 'doctor':
                # Capture the room the doctor selected and update the database
                selected_room = request.form.get('room')
                if selected_room:
                    user.room = selected_room
                    user.status = 'online'
                    db.session.commit()
                return redirect(url_for('doctor_dashboard'))
        else:
            return "Role mismatch error", 403
    return "Invalid email or password", 401

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

# --- NURSE ROUTES ---
@app.route('/register_patient', methods=['POST'])
def register_patient():
    name = request.form.get('name')
    ic = request.form.get('ic')
    age = request.form.get('age')
    room = request.form.get('room')
    symptoms = request.form.get('symptoms')
    priority = True if request.form.get('priority') == 'on' else False
    
    bp = request.form.get('bp') or "-"
    hr = request.form.get('hr') or "-"
    temp = request.form.get('temp') or "-"
    rr = request.form.get('rr') or "-"
    
    active_rooms = [r for r in get_rooms_data() if r['active']]
    if not active_rooms:
        flash("Registration Failed: No doctors are currently available.", "error")
        return redirect(request.referrer)

    if room == 'auto' or not room:
        best_room = min(active_rooms, key=lambda r: len(r['patients']))
        room = best_room['room_num']
            
    new_patient = Patient(name=name, ic=ic, age=age, room=room, symptoms=symptoms, priority=priority, bp=bp, hr=hr, temp=temp, rr=rr)
    db.session.add(new_patient)
    db.session.commit()
    flash(f"Patient {name} successfully assigned to Room {room}", "success")
    return redirect(request.referrer)

@app.route('/edit_patient/<int:patient_id>', methods=['POST'])
def edit_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    patient.name = request.form.get('name')
    patient.ic = request.form.get('ic')
    patient.age = request.form.get('age')
    patient.symptoms = request.form.get('symptoms')
    db.session.commit()
    flash(f"Patient {patient.name}'s details have been updated.", "success")
    return redirect(url_for('patient_history'))

@app.route('/delete_patient/<int:patient_id>', methods=['POST'])
def delete_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    name = patient.name
    db.session.delete(patient)
    db.session.commit()
    flash(f"Patient {name} has been deleted from the records.", "success")
    return redirect(url_for('patient_history'))

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
    if not doctor:
        session.pop('user_id', None)
        return redirect(url_for('login'))
    
    queue = Patient.query.filter_by(room=doctor.room, status='Waiting').order_by(Patient.priority.desc(), Patient.id.asc()).all()
    completed = Patient.query.filter_by(room=doctor.room, status='Completed').order_by(Patient.id.desc()).all()
    
    return render_template('doctor_dashboard.html', doctor=doctor, queue=queue, completed=completed)

@app.route('/doctor/toggle_status', methods=['POST'])
def toggle_status():
    if 'user_id' in session:
        doctor = User.query.get(session['user_id'])
        doctor.status = 'online' if doctor.status == 'offline' else 'offline'
        db.session.commit()
    return redirect(request.referrer or url_for('doctor_dashboard'))

@app.route('/doctor/consult/<int:patient_id>')
def live_consultation(patient_id):
    if 'user_id' not in session: return redirect(url_for('login'))
    patient = Patient.query.get_or_404(patient_id)
    patient.status = 'Consulting'
    db.session.commit()
    doctor = User.query.get(session['user_id'])
    return render_template('live_consultation_session.html', patient=patient, doctor=doctor)

@app.route('/doctor/summary/<int:patient_id>')
def consultation_summary(patient_id):
    if 'user_id' not in session: return redirect(url_for('login'))
    patient = Patient.query.get_or_404(patient_id)
    doctor = User.query.get(session['user_id'])
    return render_template('consultation_summary.html', patient=patient, doctor=doctor)

@app.route('/doctor/report/<int:patient_id>', methods=['POST', 'GET'])
def final_medical_note(patient_id):
    if 'user_id' not in session: return redirect(url_for('login'))
    patient = Patient.query.get_or_404(patient_id)
    patient.status = 'Completed'
    db.session.commit()
    doctor = User.query.get(session['user_id'])
    
    # Grab the form data submitted from the summary page
    form_data = request.form if request.method == 'POST' else {}
    return render_template('final_medical_note.html', patient=patient, doctor=doctor, form=form_data)

@app.route('/doctor/history')
def consultation_history():
    if 'user_id' not in session: return redirect(url_for('login'))
    doctor = User.query.get(session['user_id'])
    history = Patient.query.filter_by(room=doctor.room, status='Completed').order_by(Patient.id.desc()).all()
    return render_template('consultation_history.html', history=history, doctor=doctor)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(email='nurse@test.com').first():
            db.session.add(User(name="Nurse Joy", email='nurse@test.com', password_hash=generate_password_hash('nurse123'), role='nurse'))
        if not User.query.filter_by(email='doctor@test.com').first():
            db.session.add(User(name="Dr. Lim", email='doctor@test.com', password_hash=generate_password_hash('doctor123'), role='doctor', status='online', room='1'))
        
        sim_docs = [
            {'n':'Dr. Jackson', 'e':'jackson@hospital.com', 'p':'jackson123', 'r':'2'},
            {'n':'Dr. Taylor', 'e':'taylor@hospital.com', 'p':'taylor123', 'r':'3'}
        ]
        for d in sim_docs:
            if not User.query.filter_by(email=d['e']).first():
                db.session.add(User(name=d['n'], email=d['e'], password_hash=generate_password_hash(d['p']), role='doctor', status='online', room=d['r']))
        db.session.commit()
    # DEBUG MODE IS TRUE
    app.run(host='0.0.0.0', port=5000, debug=True)