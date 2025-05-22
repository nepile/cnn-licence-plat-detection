import cv2
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import serial

ROI_HEIGHT = 300
ROI_WIDTH = 600

try:
    ser = serial.Serial('COM8', 9600, timeout=1)  
    time.sleep(2)
    print(f"Port serial {ser.port} terbuka.")
except serial.SerialException:
    print("Gagal membuka port serial. Pastikan Arduino terhubung.")
    ser = None

# Load model deteksi karakter
model = load_model('./model/model_detection_v3.keras')

# Daftar label kelas karakter
class_names = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def normalize_light(image):
    img = cv2.resize(image, (int(image.shape[1]*.4), int(image.shape[0]*.4)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    img_opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    img_norm = gray - img_opening
    _, img_norm_bw = cv2.threshold(img_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray, img_norm_bw

def recognize_characters(plate_img):
    gray, plate_img_bw = normalize_light(plate_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    plate_img_bw = cv2.morphologyEx(plate_img_bw, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(plate_img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    index_chars_candidate = []
    index_counter = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 30 <= h <= 80 and w >= 8 and 0.2 <= aspect_ratio <= 1.0: 
            index_chars_candidate.append(index_counter)
        index_counter += 1

    if not index_chars_candidate:
        return ""

    score_chars_candidate = np.zeros(len(index_chars_candidate))
    for i, charA in enumerate(index_chars_candidate):
        xA, yA, _, _ = cv2.boundingRect(contours[charA])
        for charB in index_chars_candidate:
            if charA == charB:
                continue
            xB, yB, _, _ = cv2.boundingRect(contours[charB])
            if abs(yA - yB) < 11:
                score_chars_candidate[i] += 1

    index_chars = [index_chars_candidate[i] for i, score in enumerate(score_chars_candidate)
                   if score == max(score_chars_candidate)]

    x_coords = [(cv2.boundingRect(contours[i])[0], i) for i in index_chars]
    index_chars_sorted = [i for _, i in sorted(x_coords)]

    plate_number = ""
    for i in index_chars_sorted:
        x, y, w, h = cv2.boundingRect(contours[i])
        char_crop = gray[y:y+h, x:x+w]
        char_crop = cv2.resize(char_crop, (40, 40), interpolation=cv2.INTER_CUBIC)
        char_crop = cv2.equalizeHist(char_crop)  # kontras
        char_crop_rgb = cv2.cvtColor(char_crop, cv2.COLOR_GRAY2RGB)
        img_array = img_to_array(char_crop_rgb)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array, verbose=0)
        score = tf.nn.softmax(predictions[0])
        max_score = np.max(score)
        if max_score > 0.9:  # Ambang batas kepercayaan
            plate_number += class_names[np.argmax(score)]
    
    # Debugging
    cv2.imshow("Normalized Image", plate_img_bw)
    for i in index_chars_sorted:
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(plate_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return plate_number

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 50)
    if not cap.isOpened():
        print("Webcam tidak tersedia.")
        return

    last_detection_time = 0
    detection_delay = 2
    last_valid_plate = "Terhubung Serial"
    last_sent_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        h, w, _ = frame.shape
        center_x, center_y = w // 2, h // 2
        x1 = center_x - ROI_WIDTH // 2
        y1 = center_y - ROI_HEIGHT // 2
        x2 = center_x + ROI_WIDTH // 2
        y2 = center_y + ROI_HEIGHT // 2

        roi = frame[y1:y2, x1:x2]
        result_text = last_valid_plate

        current_time = time.time()
        if current_time - last_detection_time >= detection_delay:
            if roi.size != 0:
                plate_text = recognize_characters(roi)
                if plate_text and len(plate_text) >= 4:  # terima 4 karakter atau lebih
                    result_text = f"Plat: {plate_text}"
                    last_valid_plate = result_text
                    last_detection_time = current_time

        if result_text != last_sent_text:
            if ser is not None:
                try:
                    ser.write(result_text.encode() + b'\n')
                    ser.flush()
                    time.sleep(2)
                    print(f"Mengirim ke Arduino: {result_text}")
                except serial.SerialException:
                    print("Gagal mengirim data ke Arduino.")
            last_sent_text = result_text

        cv2.putText(frame, result_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if "Plat" in result_text else (0, 0, 255), 2)
       
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imshow("ROI & Countours", roi)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    if ser is not None:
        ser.close()

if __name__ == "__main__":
    main()