from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import json
from werkzeug.utils import secure_filename
from monitoring import process_monitoring

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data (no manual temp/humidity)
        moisture_sensor = request.form.get('moisture_sensor')
        moisture_sensor = float(moisture_sensor) if moisture_sensor else None

        # Handle file uploads
        drone_file = request.files.get('drone_image')
        leaf_file = request.files.get('leaf_image')
        soil_file = request.files.get('soil_image')

        drone_path = None
        if drone_file and drone_file.filename:
            filename = secure_filename(drone_file.filename)
            drone_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            drone_file.save(drone_path)

        leaf_path = None
        if leaf_file and leaf_file.filename:
            filename = secure_filename(leaf_file.filename)
            leaf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            leaf_file.save(leaf_path)

        soil_path = None
        if soil_file and soil_file.filename:
            filename = secure_filename(soil_file.filename)
            soil_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            soil_file.save(soil_path)

        # Process monitoring (no temp/humidity params)
        results = process_monitoring(drone_path, leaf_path, soil_path, moisture_sensor, app.config['OUTPUT_FOLDER'])

        return render_template('results.html', results=results)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
