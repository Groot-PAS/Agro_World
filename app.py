import os
import uuid
from flask import Flask, render_template, request, send_from_directory, flash, redirect, url_for
from werkzeug.utils import secure_filename

from functions import (
    diagnose_crop_image,
    get_crop_recommendation,
    get_fertilizer_recommendation,
    soil_types,
    Crop_types,
    crop_list
)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/crop-recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    if request.method == "POST":
        try:
            # Read each feature explicitly by name
            n = float(request.form['N'])
            p = float(request.form['P'])
            k = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form.get('ph') or request.form.get('phosphore', 7.0))
            rainfall = float(request.form['rainfall'])

            features = [n, p, k, temperature, humidity, ph, rainfall]
            result = get_crop_recommendation(features)
            return render_template("recommend_result.html", result=result.capitalize())
        except (KeyError, ValueError) as err:
            flash(f"Invalid input values: {err}")
            return redirect(url_for('crop_recommendation'))

    return render_template('crop-recommend.html')


@app.route('/fertilizer-recommendation', methods=['GET', 'POST'])
def fertilizer_recommendation():
    if request.method == "POST":
        try:
            # Read each feature explicitly by name
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            moisture = float(request.form['moisture'])
            nitrogen = float(request.form['N'])
            phosphorous = float(request.form['P'])
            potassium = float(request.form['K'])
            soil = float(request.form['soil'])
            crop = float(request.form['crop'])

            # Model expects numerical features in CSV order:
            # [Temparature, Humidity, Moisture, Nitrogen, Potassium, Phosphorous] -> N, K, P!
            num_features = [temperature, humidity, moisture, nitrogen, potassium, phosphorous]
            cat_features = [soil, crop]

            result = get_fertilizer_recommendation(num_features, cat_features)
            return render_template("recommend_result.html", result=result)
        except (KeyError, ValueError) as err:
            flash(f"Invalid input values: {err}")
            return redirect(url_for('fertilizer_recommendation'))

    return render_template(
        'fertilizer-recommend.html',
        soil_types=enumerate(soil_types),
        crop_types=enumerate(Crop_types)
    )


@app.route('/crop-disease', methods=['GET', 'POST'])
def find_crop_disease():
    if request.method == "GET":
        return render_template('crop-disease.html', crops=crop_list, file_name=None)

    # Validate crop selection against supported crop list
    crop = request.form.get("crop", "").strip().lower()
    if crop not in crop_list:
        flash("Invalid crop selected. Please choose a supported crop.")
        return redirect(url_for('find_crop_disease'))

    # Validate file presence and format
    if "file" not in request.files:
        flash("No image file provided.")
        return redirect(url_for('find_crop_disease'))

    file = request.files["file"]
    if file.filename == "":
        flash("No selected image file.")
        return redirect(url_for('find_crop_disease'))

    if not allowed_file(file.filename):
        flash("Unsupported file format. Please upload a JPG, JPEG, or PNG image.")
        return redirect(url_for('find_crop_disease'))

    # Avoid filename collisions with unique prefix
    safe_name = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)

    # Run inference with confidence score and threshold check
    disease_name, confidence_pct, is_confident = diagnose_crop_image(file_path, crop)

    return render_template(
        'disease-prediction-result.html',
        image_file_name=unique_filename,
        result=disease_name,
        confidence=confidence_pct,
        is_confident=is_confident,
        crop=crop.title()
    )


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], secure_filename(filename))


if __name__ == '__main__':
    app.run(debug=True)