from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "super_secret_key"

# Dummy data to simulate a database for demonstration
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
    # In a real app, verify credentials here
    role = request.form.get('role')
    if role == 'nurse':
        return redirect(url_for('nurse_dashboard'))
    return redirect(url_for('login')) # Doctor routing would go here

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
    app.run(debug=True)