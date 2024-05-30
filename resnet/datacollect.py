import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize the hand detector
detector = HandDetector(maxHands=1)

# Define the image size and offset
imgSize = 300
offset = 20

# Dictionary to map class labels to folder paths
class_folders = {
    0: "Data/middlefinger",
    1: "Data/thumbsdown",
    2: "Data/looser"
}

# Dictionary to keep track of image counts for each class
class_image_counts = {class_label: 0 for class_label in class_folders.keys()}

# Function to save image with class label
def save_image_with_label(class_label, img):
    folder = class_folders.get(class_label)
    if folder:
        # Check if the folder exists, and create it if it doesn't
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # Check if the image count for the class exceeds 500
        if class_image_counts[class_label] < 500:
            # Save the image
            cv2.imwrite(f'{folder}/Image_{class_image_counts[class_label]}.jpg', img)
            class_image_counts[class_label] += 1
            print(f"Saved {class_image_counts[class_label]} images for class {class_label}.")
            # Print total images captured
            total_images = sum(class_image_counts.values())
            print(f"Total images captured: {total_images}")
        else:
            print(f"Reached 500 images for class {class_label}. Skipping further images.")

# Function to handle user input and capture images for the selected gesture
def capture_images_with_landmarks(gesture_label):
    cap = cv2.VideoCapture(0)  # Open webcam
    counter = 0
    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            if not imgCrop.size == 0:  # Check if imgCrop is not empty
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCropShape = imgCrop.shape
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord("s"):
            counter += 1
            save_image_with_label(gesture_label, imgWhite)
            print(counter)
            if counter >= 500:
                break
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

# Function to handle user input and start capturing images for selected gestures
def capture_images():
    while True:
        print("Enter the corresponding number to capture images (0: middlefinger, 1: thumbsdown, 2: looser), or 'q' to quit:")
        choice = input()
        if choice.isdigit():
            class_label = int(choice)
            if class_label in class_folders:
                capture_images_with_landmarks(class_label)
            else:
                print("Invalid class label. Please try again.")
        elif choice.lower() == 'q':
            break
        else:
            print("Invalid input. Please enter a valid class label or 'q' to quit.")

# Main function
if __name__ == "__main__":
    capture_images()
