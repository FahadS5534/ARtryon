import cv2
import mediapipe as mp
from flask import Flask, request, send_file, render_template, jsonify
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Initialize MediaPipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Load ring image
ring_image = Image.open('ring.png').convert("RGBA")
ring_image_size = (50, 50)

def overlay_ring_on_image(frame, ring_image, ring_image_size):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                ring_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
                h, w, _ = frame.shape
                x = int(ring_finger_dip.x * w) - ring_image_size[0] // 2
                y = int(ring_finger_dip.y * h) - ring_image_size[1] // 2

                # Ensure the coordinates are within the frame boundaries
                x = max(0, min(x, w - ring_image_size[0]))
                y = max(0, min(y, h - ring_image_size[1]))

                ring_resized = ring_image.resize(ring_image_size)
                ring_array = np.array(ring_resized)

                ring_alpha = ring_array[:, :, 3] / 255.0
                background_alpha = 1.0 - ring_alpha

                for c in range(0, 3):
                    frame[y:y + ring_image_size[1], x:x + ring_image_size[0], c] = (
                        ring_alpha * ring_array[:, :, c] +
                        background_alpha * frame[y:y + ring_image_size[1], x:x + ring_image_size[0], c]
                    )

        return frame
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    if file:
        input_image_path = 'input_image.jpg'
        file.save(input_image_path)

        # Read the image
        frame = cv2.imread(input_image_path)

        # Process the image to overlay the ring
        processed_image = overlay_ring_on_image(frame, ring_image, ring_image_size)

        if processed_image is not None:
            # Convert processed image back to PIL format for sending
            processed_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

            # Save the processed image to a BytesIO object
            img_io = io.BytesIO()
            processed_image.save(img_io, 'JPEG')
            img_io.seek(0)

            return send_file(img_io, mimetype='image/jpeg')
        else:
            return jsonify({'status': 'error', 'message': 'Failed to process image'}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
