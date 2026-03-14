import os
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

# Dummy data for UI rendering (until we hook up patient tables)
rooms_data = [
    {"id": "Room 1", "doctor": "Dr. Ahmad bin Hassan", "status": "Consulting", "patients": [{"name": "Tan Mei Ling", "ic": "950315-08-1234", "symptoms": "Fever, headache"}], "duration": "45 mins"},
    {"id": "Room 2", "doctor": "Dr. Sarah Lee", "status": "Available", "patients": [], "duration": "0 mins"},
    {"id": "Room 3", "doctor": "Dr. Raj Kumar", "status": "Available", "patients": [], "duration": "0 mins"},
    {"id": "Room 4", "doctor": "-", "status": "Not Available", "patients": [], "duration": "-"},
]

history_data = [
    {"name": "Tan Mei Ling", "ic": "950315-08-1234", "symptoms": "Fever, headache", "doctor": "Dr. Ahmad bin Hassan", "room": "Room 1", "date": "2024-01-15 09:15 AM", "duration": "15 mins"}
]

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():
    # Grab the data submitted from the form
    email = request.form.get('email')
    password = request.form.get('password')
    role_selected = request.form.get('role')

    # Query the database for the user
    user = User.query.filter_by(email=email).first()

    # Check if user exists and password matches
    if user and check_password_hash(user.password_hash, password):
        if user.role == role_selected:
            if user.role == 'nurse':
                return redirect(url_for('nurse_dashboard'))
            elif user.role == 'doctor':
                # Route to doctor dashboard later; redirecting to login for now
                return redirect(url_for('login'))
        else:
            # Prevent a doctor from logging in through the nurse portal and vice versa
            return "Role mismatch error", 403
    else:
        return "Invalid email or password", 401

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
    return render_template('patient_history.html', history=history_data, rooms=rooms_data)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(email='nurse@test.com').first():
            db.session.add(User(name="Nurse Joy", email='nurse@test.com', password_hash=generate_password_hash('nurse123'), role='nurse'))
        if not User.query.filter_by(email='doctor@test.com').first():
            db.session.add(User(name="Lim", email='doctor@test.com', password_hash=generate_password_hash('doctor123'), role='doctor', status='online'))
        
        sim_docs = [
            {'n':'Jackson', 'e':'jackson@hospital.com', 'p':'jackson123', 'r':'3'},
            {'n':'Taylor', 'e':'taylor@hospital.com', 'p':'taylor123', 'r':'5'},
            {'n':'Aida', 'e':'aida@hospital.com', 'p':'aida123', 'r':'8'},
            {'n':'Aiman', 'e':'aiman@hospital.com', 'p':'aiman123', 'r':'9'},
            {'n':'Jayden', 'e':'jayden@hospital.com', 'p':'jayden123', 'r':'10'}
        ]
        for d in sim_docs:
            if not User.query.filter_by(email=d['e']).first():
                db.session.add(User(name=d['n'], email=d['e'], password_hash=generate_password_hash(d['p']), role='doctor', status='online', room=d['r']))
        db.session.commit()
    app.run(host='0.0.0.0', port=5000, debug=False)