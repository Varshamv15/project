import os
import numpy as np
import tensorflow as tf
import librosa
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import joblib  # To load the MinMaxScaler

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "secret_key"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    username = db.Column(db.String(80), unique=True, nullable=False, primary_key=True)
    password = db.Column(db.String(80), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # doctor, patient, admin

class AudioResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(80), nullable=False)
    result = db.Column(db.String(255), nullable=False)
    recommendation = db.Column(db.Text, nullable=True)
    uploaded_at = db.Column(db.DateTime, default=db.func.current_timestamp())

# Load pre-trained model (saved as .h5)
try:
    model = load_model(r"C:\Users\HP\Downloads\Final Projj(FIN)\Final Projj\parkinsons_model.h5")  # Update with the correct path
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the MinMaxScaler (assuming it's saved as a pickle file)
try:
    mmsc = joblib.load(r"C:\Users\HP\Downloads\Final Projj(FIN)\Final Projj\scaler.pkl")  # Update with the correct path
    print("MinMax Scaler loaded successfully!")
except Exception as e:
    print(f"Error loading MinMax Scaler: {e}")
    scaler = None

# Feature Extraction function for the 22 features
def extract_features_from_wav(file_path, duration=3, sr=22050):
    # Load the audio file
    y, _ = librosa.load(file_path, sr=sr, duration=duration)
    y = librosa.util.fix_length(y, size=sr * duration)  # Pad or trim to fixed duration
    
    # 1. MDVP:Fo (Spectral Centroid) -> Frequency characteristic
    mdvp_fo = librosa.feature.spectral_centroid(y=y, sr=sr)
    mdvp_fo = np.mean(mdvp_fo)  # Mean across time
    
    # 2. MDVP:Fhi (Spectral Bandwidth) -> Frequency characteristic
    mdvp_fhi = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    mdvp_fhi = np.mean(mdvp_fhi)  # Mean across time
    
    # 3. MDVP:Flo (Zero Crossing Rate) -> Frequency characteristic
    mdvp_flo = librosa.feature.zero_crossing_rate(y=y)
    mdvp_flo = np.mean(mdvp_flo)  # Mean across time
    
    # 4. MDVP:Jitter(%) - Standard deviation of pitch (Jitter)
    jitter_percent = librosa.feature.tempogram(y=y)
    jitter_percent = np.mean(jitter_percent)  # Mean across time
    
    # 5. MDVP:Jitter(Abs) - Standard deviation of pitch
    jitter_abs = np.std(y)  # Scalar, already computed
    
    # 6. MDVP:RAP (Root Mean Square)
    rap = librosa.feature.rms(y=y)
    rap = np.mean(rap)  # Mean across time
    
    # 7. MDVP:PPQ (Spectral Flatness)
    ppq = librosa.feature.spectral_flatness(y=y)
    ppq = np.mean(ppq)  # Mean across time
    
    # 8. Jitter:DDP (Delta Periodicity)
    jitter_ddp = np.std(np.diff(y))  # Difference in signal
    
    # 9. MDVP:Shimmer (Amplitude Perturbation)
    shimmer = librosa.feature.chroma_stft(y=y, sr=sr)
    shimmer = np.mean(shimmer, axis=0)  # Average across time frames (mean per frequency bin)
    
    # 10. MDVP:Shimmer(dB) (Logarithmic Scale)
    shimmer_dB = librosa.feature.spectral_flatness(y=y)
    shimmer_dB = np.mean(shimmer_dB)  # Mean across time
    
    # 11. Shimmer:DDA (Dynamic Amplitude)
    shimmer_dda = np.std(np.diff(shimmer))  # Difference in shimmer
    
    # 12. NHR (Harmonic-to-Noise Ratio)
    nhr = librosa.effects.harmonic(y)
    nhr_mean = np.mean(nhr)  # Scalar value
    
    # 13. HNR (Harmonic-to-Noise Ratio for the entire signal)
    hnr = librosa.effects.harmonic(y)
    hnr_mean = np.mean(hnr)  # Scalar value
    
    # 14. RPDE (Recurrence Period Density Entropy) - Placeholder
    rpde = np.std(y)  # Scalar
    
    # 15. DFA (Detrended Fluctuation Analysis) - Placeholder
    dfa = np.std(y)  # Scalar
    
    # 16. Spread1 (Spread Measures)
    spread1 = np.std(y)  # Scalar
    
    # 17. Spread2 (Spread Measures)
    spread2 = np.std(y)  # Scalar
    
    # 18. D2 (Dynamical Measure)
    d2 = np.std(y)  # Scalar
    
    # 19. PPE (Perturbation of Periodic Energy)
    ppe = np.std(y)  # Scalar

     # 20. Shimmer:APQ3 (Amplitude Perturbation Quotient with a 3-point filter)
    apq3 = np.mean(np.diff(y))  # Using the difference between amplitude values
    
    # 21. Shimmer:APQ5 (Amplitude Perturbation Quotient with a 5-point filter)
    apq5 = np.mean(np.diff(y))  # Using the difference between amplitude values
    
    # 22. MDVP:APQ (Amplitude Perturbation Quotient)
    mdvp_apq = np.mean(np.diff(y))  # Using the difference between amplitude values


    
    
    # Ensure each feature is a scalar or 1D array
    features = np.concatenate([
        np.array([mdvp_fo]),
        np.array([mdvp_fhi]),
        np.array([mdvp_flo]),
        np.array([jitter_percent]),
        np.array([jitter_abs]),
        np.array([rap]),
        np.array([ppq]),
        np.array([jitter_ddp]),
        np.array([np.mean(shimmer)]),  # Use mean if shimmer is an array
        np.array([shimmer_dB]),
          np.array([apq3]),  # New APQ3 feature
        np.array([apq5]),  # New APQ5 feature
        np.array([mdvp_apq]) , # New MDVP:APQ feature
        np.array([shimmer_dda]),
        np.array([nhr_mean]),
        np.array([hnr_mean]),
        np.array([rpde]),
        np.array([dfa]),
        np.array([spread1]),
        np.array([spread2]),
        np.array([d2]),
        np.array([ppe])
    ])
    
    # Ensure it's a 1D array with exactly 22 features
    features = features.flatten()
   
    # Check length of the feature vector
    print(f"Feature vector length: {features.shape[0]}")  # Should print 22
    return features


# Preprocess the features
def preprocess_features(features):
    features =mmsc.transform(features.reshape(1, -1))  # Reshape to 2D and normalize
    return features

# Make predictions from the .wav file
def predict_from_wav(file_path):
    features = extract_features_from_wav(file_path)  # Extract features from the .wav file
    print(features.shape)
    features = preprocess_features(features)  # Preprocess the features (scale them)
    print(features)
    
    # Predict using the trained model
    predictions = model.predict(features)
    print(f"Raw predictions: {predictions}")

    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Print the results (healthy or Parkinson's Disease)
    print(f"Prediction: {'Healthy' if predicted_class == 0 else 'Parkinsons Disease'}")
    

    class_labels = ["Healthy", "Parkinson's Disease"]
    return class_labels[predicted_class], np.max(predictions)


# Flask routes
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        
        if role == 'admin':
            flash("Admin accounts cannot be created through signup.")
            return redirect(url_for('signup'))

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already exists. Please choose a different one.")
            return redirect(url_for('signup'))

        new_user = User(username=username, password=password, role=role)
        db.session.add(new_user)
        db.session.commit()

        flash("Account created successfully. Please log in.")
        return redirect(url_for('signup'))  # Redirect to signup instead of login

    return render_template('signup1.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user_id'] = user.username
            session['role'] = user.role
            if user.role == 'doctor':
                return redirect(url_for('doctor_dashboard'))
            elif user.role == 'patient':
                return redirect(url_for('patient_dashboard'))
            elif user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
        flash("Invalid credentials.")
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == "POST":
        file = request.files.get('voice_file')
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            result, confidence = predict_from_wav(file_path)
            print(f"Result: {result}, Confidence: {confidence}")
            patient_name = session['user_id']
            audio_result = AudioResult(patient_name=patient_name, result=result)
            db.session.add(audio_result)
            db.session.commit()

            return render_template("result.html", result=result, confidence=confidence)
        flash("No file uploaded.")
        return redirect(request.url)
    return render_template("upload.html")

@app.route('/doctor_dashboard')
def doctor_dashboard():
    if 'role' in session and session['role'] == 'doctor':
        return render_template('doctor1.html')
    return redirect(url_for('login'))

@app.route('/patient_dashboard', methods=['GET', 'POST'])
def patient_dashboard():
    if 'role' in session and session['role'] == 'patient':
        if request.method == 'POST':
            file = request.files.get('file')
            if file:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)

                result, confidence = predict_from_wav(file_path)
                patient_name = session['user_id']
                audio_result = AudioResult(patient_name=patient_name, result=result)
                db.session.add(audio_result)
                db.session.commit()

                return render_template('patient1.html', result=result, confidence=confidence)
            flash("No file uploaded.")
            return redirect(request.url)
        return render_template('patient1.html')
    return redirect(url_for('login'))

@app.route('/add_recommendation/<int:result_id>', methods=['POST'])
def add_recommendation(result_id):
    if 'role' in session and session['role'] == 'doctor':
        recommendation = request.form.get('recommendation')
        audio_result = AudioResult.query.get(result_id)
        if audio_result:
            audio_result.recommendation = recommendation
            db.session.commit()
            flash("Recommendation added successfully.")
        else:
            flash("Result not found.")
        return redirect(url_for('view_patient'))
    return redirect(url_for('login'))


@app.route('/view_patient', methods=['GET', 'POST'])
def view_patient():
    if 'role' in session and session['role'] == 'doctor':
        if request.method == 'POST':
            patient_name = request.form.get('patient_name')
            results = AudioResult.query.filter_by(patient_name=patient_name).all()
            if not results:
                flash("No results found for the entered patient name.")
            return render_template('view_patient.html', results=results, patient_name=patient_name)
        return render_template('view_patient.html', results=None)
    return redirect(url_for('login'))

@app.route('/admin_dashboard', methods=['GET', 'POST'])
def admin_dashboard():
    if 'role' in session and session['role'] == 'admin':
        if request.method == 'POST':
            action = request.form.get('action')
            username = request.form.get('username')
            password = request.form.get('password')
            role = request.form.get('role')

            if action == 'add':
                existing_user = User.query.filter_by(username=username).first()
                if existing_user:
                    flash("Username already exists.")
                else:
                    new_user = User(username=username, password=password, role=role)
                    db.session.add(new_user)
                    db.session.commit()
                    flash("User added successfully.")
            elif action == 'remove':
                user = User.query.filter_by(username=username).first()
                if user:
                    db.session.delete(user)
                    db.session.commit()
                    flash("User removed successfully.")
                else:
                    flash("User not found.")
        users = User.query.all()
        return render_template('admin.html', users=users)
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Initialize database
def setup_app():
    with app.app_context():
        db.create_all()

if __name__ == "__main__":
    setup_app()
    app.run(debug=True)
