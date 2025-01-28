# import cv2
# import pickle
# import numpy as np
# import os

# video = cv2.VideoCapture(0)
# face_cascade_path = 'model/haarcascade_frontalface_default.xml'

# if not os.path.isfile(face_cascade_path):
#     print("Error: The Haar Cascade XML file is missing.")
#     exit(1)

# facedetect = cv2.CascadeClassifier(face_cascade_path)

# if facedetect.empty():
#     print("Error: Cascade classifier not loaded.")
#     exit(1)

# faces_data = []
# age_data = []
# crime_data = []
# gender_data = []

# i = 0

# name = input("Enter name: ")
# age = input("Enter age: ")
# crime = input("Enter the criminal activities: ")
# gender = input("Enter gender (Male/Female/Other): ")

# while True:
#     ret, frame = video.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
#     for (x, y, w, h) in faces:
#         crop_img = frame[y:y+h, x:x+w, :]
#         resized_img = cv2.resize(crop_img, (50, 50))
        
#         # Collect face data
#         if len(faces_data) < 100 and i % 10 == 0:
#             faces_data.append(resized_img)
#             age_data.append(age)
#             crime_data.append(crime)
#             gender_data.append(gender)
        
#         i += 1
        
#         cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    
#     cv2.imshow("Frame", frame)
#     k = cv2.waitKey(1)
    
#     if k == ord('q') or len(faces_data) == 100:
#         break

# video.release()

# faces_data = np.asarray(faces_data)
# faces_data = faces_data.reshape(100, -1)

# # Save name, age, crime, gender data in separate pickle files
# if 'names.pkl' not in os.listdir('data/'):
#     names = [name] * 100
#     with open('data/names.pkl', 'wb') as f:
#         pickle.dump(names, f)
# else:
#     with open('data/names.pkl', 'rb') as f:
#         names = pickle.load(f)
#     names = names + [name] * 100
#     with open('data/names.pkl', 'wb') as f:
#         pickle.dump(names, f)

# if 'ages.pkl' not in os.listdir('data/'):
#     ages = [age] * 100
#     with open('data/ages.pkl', 'wb') as f:
#         pickle.dump(ages, f)
# else:
#     with open('data/ages.pkl', 'rb') as f:
#         ages = pickle.load(f)
#     ages = ages + [age] * 100
#     with open('data/ages.pkl', 'wb') as f:
#         pickle.dump(ages, f)

# if 'crimes.pkl' not in os.listdir('data/'):
#     crimes = [crime] * 100
#     with open('data/crimes.pkl', 'wb') as f:
#         pickle.dump(crimes, f)
# else:
#     with open('data/crimes.pkl', 'rb') as f:
#         crimes = pickle.load(f)
#     crimes = crimes + [crime] * 100
#     with open('data/crimes.pkl', 'wb') as f:
#         pickle.dump(crimes, f)

# if 'genders.pkl' not in os.listdir('data/'):
#     genders = [gender] * 100
#     with open('data/genders.pkl', 'wb') as f:
#         pickle.dump(genders, f)
# else:
#     with open('data/genders.pkl', 'rb') as f:
#         genders = pickle.load(f)
#     genders = genders + [gender] * 100
#     with open('data/genders.pkl', 'wb') as f:
#         pickle.dump(genders, f)

# if 'faces_data.pkl' not in os.listdir('data/'):
#     with open('data/faces_data.pkl', 'wb') as f:
#         pickle.dump(faces_data, f)
# else:
#     with open('data/faces_data.pkl', 'rb') as f:
#         faces = pickle.load(f)
#     faces = np.append(faces, faces_data, axis=0)
#     with open('data/faces_data.pkl', 'wb') as f:
#         pickle.dump(faces, f)

import cv2
import pickle
import numpy as np
import os
from datetime import datetime

# Helper Function: Ensure Directory Exists
def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Initialize Webcam
video = cv2.VideoCapture(0)
face_cascade_path = 'model/haarcascade_frontalface_default.xml'

# Check if the Haar Cascade file exists
if not os.path.isfile(face_cascade_path):
    print("Error: The Haar Cascade XML file is missing.")
    exit(1)

facedetect = cv2.CascadeClassifier(face_cascade_path)
if facedetect.empty():
    print("Error: Failed to load the Haar Cascade classifier.")
    exit(1)

# Create or ensure the 'data/' directory exists
ensure_dir_exists('data')

# Collect User Inputs with Validation
def get_valid_input(prompt, validation_func):
    while True:
        value = input(prompt)
        if validation_func(value):
            return value
        else:
            print("Invalid input. Please try again.")

name = input("Enter name: ")
age = get_valid_input("Enter age (must be a number): ", lambda x: x.isdigit() and int(x) > 0)
crime = input("Enter the criminal activities: ")
gender = get_valid_input("Enter gender (Male/Female/Other): ", lambda x: x.lower() in ['male', 'female', 'other'])

# Let user specify how many faces to collect
face_limit = int(get_valid_input("Enter the number of face samples to collect: ", lambda x: x.isdigit() and int(x) > 0))

# Variables to Store Face and Metadata
faces_data = []
age_data = []
crime_data = []
gender_data = []

# Capture Faces
i = 0
print("Starting face capture. Press 'q' to quit early.")

while len(faces_data) < face_limit:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))

        # Collect face data
        if len(faces_data) < face_limit and i % 10 == 0:
            faces_data.append(resized_img)
            age_data.append(age)
            crime_data.append(crime)
            gender_data.append(gender)

            print(f"Captured {len(faces_data)} out of {face_limit} faces.")

        i += 1

        # Draw rectangle around detected face and display progress
        cv2.putText(frame, f"Captured: {len(faces_data)}/{face_limit}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)

    # Break if 'q' is pressed
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# Convert Faces Data to NumPy Array
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(len(faces_data), -1)

# Save Data
def save_data(file_name, new_data):
    file_path = os.path.join('data', file_name)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            existing_data = pickle.load(f)
        updated_data = np.append(existing_data, new_data, axis=0) if isinstance(new_data, np.ndarray) else existing_data + new_data
    else:
        updated_data = new_data

    with open(file_path, 'wb') as f:
        pickle.dump(updated_data, f)

# Save Metadata and Faces
save_data('names.pkl', [name] * len(faces_data))
save_data('ages.pkl', [age] * len(faces_data))
save_data('crimes.pkl', [crime] * len(faces_data))
save_data('genders.pkl', [gender] * len(faces_data))
save_data('faces_data.pkl', faces_data)

# Feedback to User
print(f"Face data collection complete. {len(faces_data)} faces saved successfully.")

