from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
from datetime import datetime

app = Flask(__name__)

# Load model yang sudah dilatih
model = load_model('model/model_detection_v2.keras')  # Ganti dengan path ke model Anda

# Daftar kelas karakter
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Buat folder untuk menyimpan gambar
segmentation_folder = 'segmentation_images'
result_folder = 'result_images'

if not os.path.exists(segmentation_folder):
    os.makedirs(segmentation_folder)
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# Fungsi untuk normalisasi cahaya
def normalize_light(image):
    img = cv2.resize(image, (int(image.shape[1]*.4),int(image.shape[0]*.4)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    img_opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    img_norm = gray - img_opening
    _, img_norm_bw = cv2.threshold(img_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray, img_norm_bw

# Fungsi untuk mendeteksi plat nomor
def detect_plate(image):
    img_gray, img_norm_bw = normalize_light(image)
    
    # Simpan gambar yang dinormalisasi
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cv2.imwrite(os.path.join(segmentation_folder, f'segmentation_{timestamp}.jpg'), img_norm_bw)
    
    contours, _ = cv2.findContours(img_norm_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    index_plate_candidate = []
    index_counter = 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if w >= 100 and aspect_ratio <= 4:
            index_plate_candidate.append(index_counter)
        index_counter += 1
    
    if len(index_plate_candidate) == 0:
        return None, img_gray, None, None
    
    if len(index_plate_candidate) == 1:
        x_plate, y_plate, w_plate, h_plate = cv2.boundingRect(contours[index_plate_candidate[0]])
    else:
        x_plate, y_plate, w_plate, h_plate = cv2.boundingRect(contours[index_plate_candidate[1]])
    
    plate_img = img_gray[y_plate:y_plate+h_plate, x_plate:x_plate+w_plate]
    plate_img_bw = img_norm_bw[y_plate:y_plate+h_plate, x_plate:x_plate+w_plate]
    
    return plate_img, plate_img_bw, (x_plate, y_plate, w_plate, h_plate), image

# Fungsi untuk segmentasi dan pengenalan karakter
def recognize_characters(plate_img, plate_img_bw):
    if plate_img is None:
        return ""
    
    if plate_img_bw is None:
        _, plate_img_bw = cv2.threshold(plate_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    plate_img_bw = cv2.morphologyEx(plate_img_bw, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(plate_img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    index_chars_candidate = []
    index_counter = 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 40 <= h <= 60 and w >= 10:
            index_chars_candidate.append(index_counter)
        index_counter += 1
    
    if not index_chars_candidate:
        return ""
    
    score_chars_candidate = np.zeros(len(index_chars_candidate))
    counter_index = 0
    
    for charA in index_chars_candidate:
        xA, yA, wA, hA = cv2.boundingRect(contours[charA])
        for charB in index_chars_candidate:
            if charA == charB:
                continue
            xB, yB, wB, hB = cv2.boundingRect(contours[charB])
            y_diff = abs(yA - yB)
            if y_diff < 11:
                score_chars_candidate[counter_index] += 1
        counter_index += 1
    
    index_chars = []
    for i, score in enumerate(score_chars_candidate):
        if score == max(score_chars_candidate):
            index_chars.append(index_chars_candidate[i])
    
    x_coords = []
    for char in index_chars:
        x, _, _, _ = cv2.boundingRect(contours[char])
        x_coords.append(x)
    
    x_coords_sorted = sorted(x_coords)
    index_chars_sorted = []
    for x_coord in x_coords_sorted:
        for char in index_chars:
            x, _, _, _ = cv2.boundingRect(contours[char])
            if x == x_coord:
                index_chars_sorted.append(char)
    
    img_height, img_width = 40, 40
    num_plate = []
    
    for char in index_chars_sorted:
        x, y, w, h = cv2.boundingRect(contours[char])
        char_crop = cv2.cvtColor(plate_img_bw[y:y+h, x:x+w], cv2.COLOR_GRAY2BGR)
        char_crop = cv2.resize(char_crop, (img_width, img_height))
        
        img_array = tf.keras.preprocessing.image.img_to_array(char_crop)
        img_array = tf.expand_dims(img_array, 0)
        
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        num_plate.append(class_names[np.argmax(score)])
    
    plate_number = ''.join(num_plate)
    return plate_number

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk deteksi plat nomor dari kamera
@app.route('/detect', methods=['POST'])
# def detect_license_plate():
#     data = request.get_json()
#     if not data or 'image' not in data:
#         return jsonify({'status': 'error', 'message': 'No image provided'})

#     # Decode gambar dari base64
#     image_data = data['image']
#     image_data = image_data.split(',')[1]
#     image = Image.open(BytesIO(base64.b64decode(image_data)))
#     image = np.array(image)

#     # Konversi ke format yang sesuai untuk OpenCV (BGR)
#     if len(image.shape) == 2:
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#     elif image.shape[2] == 4:
#         image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
#     else:
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     # Deteksi plat
#     plate_img, plate_img_bw, bbox, original_image = detect_plate(image)
#     if plate_img is None:
#         return jsonify({'status': 'error', 'message': 'No plate detected'})

#     # Kenali karakter
#     plate_number = recognize_characters(plate_img, plate_img_bw)
#     if not plate_number:
#         return jsonify({'status': 'error', 'message': 'No characters recognized'})

#     # Gambar kotak dan teks nomor plat pada gambar asli
#     if bbox:
#         x_plate, y_plate, w_plate, h_plate = bbox
#         cv2.rectangle(original_image, (x_plate, y_plate), (x_plate + w_plate, y_plate + h_plate), (0, 255, 0), 5)
#         cv2.putText(original_image, plate_number, (x_plate, y_plate + h_plate + 50), cv2.FONT_ITALIC, 2.0, (0, 255, 0), 3)
        
#         # Simpan gambar hasil deteksi
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         cv2.imwrite(os.path.join(result_folder, f'result_camera_{timestamp}.jpg'), original_image)

#     return jsonify({
#         'status': 'success',
#         'plate_number': plate_number
#     })
def detect_license_plate():
    data = request.get_json()
    if not data or 'image' not in data:
        print("Error: No image provided")
        return jsonify({'status': 'error', 'message': 'No image provided'})

    # Decode gambar dari base64
    image_data = data['image']
    image_data = image_data.split(',')[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = np.array(image)

    # Logging untuk debugging
    print(f"Received image with shape: {image.shape}")

    # Konversi ke format yang sesuai untuk OpenCV (BGR)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Deteksi plat
    plate_img, plate_img_bw, bbox, original_image = detect_plate(image)
    if plate_img is None:
        print("Error: No plate detected")
        return jsonify({'status': 'error', 'message': 'No plate detected'})

    # Kenali karakter
    plate_number = recognize_characters(plate_img, plate_img_bw)
    if not plate_number:
        print("Error: No characters recognized")
        return jsonify({'status': 'error', 'message': 'No characters recognized'})

    # Gambar kotak dan teks nomor plat pada gambar asli
    if bbox:
        x_plate, y_plate, w_plate, h_plate = bbox
        cv2.rectangle(original_image, (x_plate, y_plate), (x_plate + w_plate, y_plate + h_plate), (0, 255, 0), 5)
        cv2.putText(original_image, plate_number, (x_plate, y_plate + h_plate + 50), cv2.FONT_ITALIC, 2.0, (0, 255, 0), 3)
        
        # Simpan gambar hasil deteksi
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cv2.imwrite(os.path.join(result_folder, f'result_camera_{timestamp}.jpg'), original_image)

    print(f"Detected plate number: {plate_number}")
    return jsonify({
        'status': 'success',
        'plate_number': plate_number
    })


# Route untuk deteksi plat nomor dari gambar yang diunggah
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file provided'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})

    # Baca gambar dari file
    image = Image.open(file.stream)
    image = np.array(image)
    
    # Konversi ke format yang sesuai untuk OpenCV (BGR)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Deteksi plat
    plate_img, plate_img_bw, bbox, original_image = detect_plate(image)
    if plate_img is None:
        return jsonify({'status': 'error', 'message': 'No plate detected'})

    # Kenali karakter
    plate_number = recognize_characters(plate_img, plate_img_bw)
    if not plate_number:
        return jsonify({'status': 'error', 'message': 'No characters recognized'})

    # Gambar kotak dan teks nomor plat pada gambar asli
    if bbox:
        x_plate, y_plate, w_plate, h_plate = bbox
        # cv2.rectangle(original_image, (x_plate, y_plate), (x_plate + w_plate, y_plate + h_plate), (0, 255, 0), 5)
        # cv2.putText(original_image, plate_number, (x_plate, y_plate + h_plate + 50), cv2.FONT_ITALIC, 2.0, (0, 255, 0), 3)
        cv2.putText(image, plate_number,(x_plate, y_plate + h_plate + 500), cv2.FONT_ITALIC, 2.0, (0,255,0), 3)
        
        # Simpan gambar hasil deteksi
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cv2.imwrite(os.path.join(result_folder, f'result_upload_{timestamp}.jpg'), original_image)

    return jsonify({
        'status': 'success',
        'plate_number': plate_number
    })

if __name__ == '__main__':
    app.run(debug=True)