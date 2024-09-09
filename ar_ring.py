import cv2
import mediapipe as mp
from flask import Flask, Response,render_template
import numpy as np
from PIL import Image

app = Flask(__name__)

# Initialize MediaPipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Capture the camera feed
cap = cv2.VideoCapture(0)
ring_image = Image.open('ring.png').convert("RGBA")
ring_image_size = (50, 50)


# Function to process the video frame and overlay the ring on the hand
def capture_video():
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                continue

            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)  # 1 for horizontal flip

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
                    roi_height = min(ring_array.shape[0], h - y)  # Ensure the ROI doesn't exceed the frame
                    roi_width = min(ring_array.shape[1], w - x)

                    # Ensure ROI dimensions are valid
                    if roi_height <= 0 or roi_width <= 0:
                        continue  # Skip if ROI is invalid

                    # Adjust the ring image size to fit the ROI if necessary
                    ring_array = ring_array[:roi_height, :roi_width]

                    # Extract the alpha channel (transparency) from the ring image
                    ring_alpha = ring_array[:, :, 3] / 255.0
                    background_alpha = 1.0 - ring_alpha

                    # Ensure the dimensions match for blending
                    if ring_array.shape[0] == roi_height and ring_array.shape[1] == roi_width:
                        # Perform the blending of the ring image with the ROI
                        for c in range(0, 3):
                            frame[y:y + roi_height, x:x + roi_width, c] = (
                                ring_alpha * ring_array[:, :, c] +
                                background_alpha * frame[y:y + roi_height, x:x + roi_width, c]
                            )

            # Encode the frame in JPEG format for display in HTML
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Return the frame as part of a multipart HTTP response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(f"Error processing frame: {e}")
            continue  # Continue with the next frame

@app.route('/')
def index():
    return render_template('index.html')

# Route to stream the video feed
@app.route('/video_feed')
def video_feed():
    return Response(capture_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        # Check if the camera is opened successfully
        if not cap.isOpened():
            print("Error: Camera not accessible")
        else:
            # Run the Flask app
            app.run(debug=True)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()