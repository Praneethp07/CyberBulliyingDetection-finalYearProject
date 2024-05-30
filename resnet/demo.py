import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize hand detector and classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")

# Constants
offset = 20
imgSize = 300
labels = ["looser", "middlefinger", "thumbsdown", "thumbsup", "peace_sign", "pointing", "waving", "ok_sign", "thumbsright", "thumbsleft", "fistbump", "call_me", "rock_on", "5_fingers", "4_fingers", "3_fingers", "2_fingers", "1_finger", "side_hand", "hands_down"]

# Initialize label frequency count dictionary
label_counts = {label: 0 for label in labels}

while True:
    # Read frame from the camera
    success, img = cap.read()
    if not success:
        break

    # Create a copy of the frame for output
    imgOutput = img.copy()

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

            # Draw rectangle and label on the output frame
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),
                          (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

            # Display the cropped and processed images
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    # Display the output frame
    cv2.imshow("Image", imgOutput)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate total frames
total_frames = sum(label_counts.values())

# Calculate total count of bullying gestures
total_bullying_gestures = sum(label_counts.values())

# Calculate percentage of cyberbullying gestures
if total_frames != 0:
    percentage_bullying = (total_bullying_gestures / total_frames) * 100
    print(f"Percentage of cyberbullying in the video: {percentage_bullying:.2f}%")
else:
    print("Error: No frames processed.")

# Print label frequency counts
print("Label Frequency Counts:")
for label, count in label_counts.items():
    print(f"{label}: {count}")

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()