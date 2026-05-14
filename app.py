from flask import Flask, render_template, request, send_file
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['REPORT_FOLDER'] = 'static/reports/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load Pre-trained Model
model = tf.keras.models.load_model('models/pneumonia_model.h5')

def preprocess_image(image_path):
    """Preprocess uploaded X-ray image for prediction"""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_pneumonia(image_path):
    """Predict pneumonia severity from X-ray"""
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)[0]
    labels = ['Normal', 'Mild', 'Moderate', 'Severe']
    return labels[np.argmax(prediction)]

def get_remedies(severity):
    """Return remedies based on severity"""
    remedies = {
        "Mild": ["Stay hydrated", "Rest well", "Take Vitamin C", "Use a humidifier"],
        "Moderate": ["Antibiotics if prescribed", "Steam inhalation", "Avoid smoking"],
        "Severe": ["Hospitalization required!", "Oxygen therapy", "Emergency care"]
    }
    return remedies.get(severity, ["No pneumonia detected. Stay healthy!"])

def generate_pdf(image_path, severity, remedies):
    """Generate a PDF report"""
    pdf_path = os.path.join(app.config['REPORT_FOLDER'], os.path.basename(image_path).replace('.jpg', '.pdf'))
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawString(100, 750, "Pneumonia Detection Report")
    c.drawString(100, 730, f"Prediction: {severity}")

    y = 700
    c.drawString(100, y, "Recommended Remedies:")
    for remedy in remedies:
        y -= 20
        c.drawString(120, y, f"- {remedy}")

    # Save PDF
    c.save()
    return pdf_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file is in request
        if 'file' not in request.files:
            return render_template('index.html', error='No file selected')
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return render_template('index.html', error='Only image files allowed (jpg, jpeg, png, gif)')
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            severity = predict_pneumonia(file_path)
            remedies = get_remedies(severity)

            pdf_path = generate_pdf(file_path, severity, remedies)

            return render_template('results.html', severity=severity, remedies=remedies, image_path=file_path, pdf_path=pdf_path)
    
    return render_template('index.html')

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
