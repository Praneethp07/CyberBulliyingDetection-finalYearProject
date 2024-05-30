import cv2
import numpy as np
import math
import os
from flask import Flask, request, jsonify
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize hand detector and classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")

# Constants
offset = 20
imgSize = 300
labels = ["looser", "middlefinger", "thumbsdown"]

# Initialize label frequency count dictionary
label_counts = {label: 0 for label in labels}

# Variable to count the total number of frames processed
total_frames_processed = 0

@app.route('/process_video', methods=['POST'])
def process_video():
    global total_frames_processed  # Declare total_frames_processed as global

    try:
        # Get video file from the POST request
        file = request.files['video']

        # Save the uploaded video file to disk
        video_path = 'uploaded_video.mp4'  # Choose a location to save the file
        file.save(video_path)

        # Read video file
        cap = cv2.VideoCapture(video_path)

        while True:
            # Read frame from the video
            success, img = cap.read()
            if not success:
                break

            # Increment the total number of frames processed
            total_frames_processed += 1

            # Detect hands in the frame
            hands, img = detector.findHands(img)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                # Crop the region around the hand
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                # Check if imgCrop is not empty
                if imgCrop.size > 0:
                    # Create a white image for processing
                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                    # Resize the cropped region if aspect ratio is greater than 1
                    aspectRatio = h / w
                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize

                    # Predict the gesture using the classifier
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                    # Increment the count for the predicted label
                    label_counts[labels[index]] += 1

            # Check for 'q' key press to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Close the video capture
        cap.release()

        # Remove the saved video file
        os.remove(video_path)

        # Calculate percentage of cyberbullying gestures
        percentage_bullying = (label_counts['looser'] + label_counts['middlefinger'] + label_counts['thumbsdown']) / total_frames_processed * 100

        # Return the results as JSON
        return jsonify({
            "label_counts": label_counts,
            "total_frames_processed": total_frames_processed,
            "percentage_bullying": percentage_bullying
        })
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
