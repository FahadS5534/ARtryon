import cv2
import mediapipe as mp
from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__)

# Initialize MediaPipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Load ring image
ring_image = Image.open('ring.png').convert("RGBA")
ring_image_size = (50, 50)

# Function to process uploaded image and overlay the ring on the hand
def process_image(image_data):
    try:
        # Convert the image data to a PIL image
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        frame = np.array(image)

        # Convert the frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(frame_rgb)

        # If hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the coordinates of the ring finger DIP joint
                ring_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
                h, w, _ = frame.shape
                x = int(ring_finger_dip.x * w) - ring_image_size[0] // 2  # Centering the ring
                y = int(ring_finger_dip.y * h) - ring_image_size[1] // 2

                # Resize the ring image
                ring_resized = ring_image.resize(ring_image_size)

                # Convert the resized ring image to a NumPy array
                ring_array = np.array(ring_resized)

                # Define the region of interest (ROI) on the frame
                roi_height = min(ring_array.shape[0], h - y)
                roi_width = min(ring_array.shape[1], w - x)

                # Ensure ROI dimensions are valid
                if roi_height <= 0 or roi_width <= 0:
                    continue

                # Extract the alpha channel (transparency) from the ring image
                ring_alpha = ring_array[:, :, 3] / 255.0
                background_alpha = 1.0 - ring_alpha

                # Blend the ring image with the frame
                for c in range(0, 3):
                    frame[y:y + roi_height, x:x + roi_width, c] = (
                        ring_alpha * ring_array[:, :, c] +
                        background_alpha * frame[y:y + roi_height, x:x + roi_width, c]
                    )

        # Convert the frame back to PIL image for encoding
        result_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buffered = io.BytesIO()
        result_image.save(buffered, format="JPEG")
        result_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return result_image_base64

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Route to process the image uploaded from the client-side camera
@app.route('/process_image', methods=['POST'])
def process_image_route():
    data = request.get_json()
    image_data = data['image']

    # Process the image
    processed_image = process_image(image_data)

    if processed_image:
        return jsonify({'status': 'success', 'processed_image': processed_image})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to process image'})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
