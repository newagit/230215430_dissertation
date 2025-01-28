# import cv2
# import pickle
# import numpy as np
# import os
# import csv
# import time
# from datetime import datetime
# from sklearn.neighbors import KNeighborsClassifier

# from win32com.client import Dispatch

# def speak(str1):
#     speak = Dispatch(("SAPI.SpVoice"))
#     speak.Speak(str1)

# video = cv2.VideoCapture(0)
# facedetect = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

# with open('data/names.pkl', 'rb') as w:
#     LABELS = pickle.load(w)
# with open('data/ages.pkl', 'rb') as f:
#     AGES = pickle.load(f)
# with open('data/crimes.pkl', 'rb') as f:
#     CRIMES = pickle.load(f)
# with open('data/genders.pkl', 'rb') as f:
#     GENDERS = pickle.load(f)
    
# with open('data/faces_data.pkl', 'rb') as f:
#     FACES=pickle.load(f)

# print('Shape of Faces matrix --> ', FACES.shape)

# knn=KNeighborsClassifier(n_neighbors=5)
# knn.fit(FACES, LABELS)


# imgBackground = cv2.imread("background.png")

# COL_NAMES = ['NAME', 'AGE', 'GENDER', 'CRIME', 'TIME']

# while True:
#     ret, frame = video.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)
#     for (x, y, w, h) in faces:
#         crop_img = frame[y:y+h, x:x+w, :]
#         resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
#         output = knn.predict(resized_img)
#         ts = time.time()
#         date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
#         timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
#         exist = os.path.isfile("Crime/Record_" + date + ".csv")
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
#         cv2.rectangle(frame, (x, y), (x+w, y), (50, 50, 255), -1)
#         #cv2.putText(frame, details_text, (x, y + h + 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        
#         name = str(output[0])
#         age = AGES[LABELS.index(name)]
#         crime = CRIMES[LABELS.index(name)]
#         gender = GENDERS[LABELS.index(name)]
        
#         details_text = f'Name: {name}\nAge: {age}\nGender: {gender}\nCrime: {crime}'
#         x_offset = y + h + 25
#         y_offset = y + h + 25
#         for line in details_text.split('\n'):
#             cv2.putText(frame, line, (x, y_offset), cv2.FONT_HERSHEY_COMPLEX, 1, (80, 255, 80), 1)
#             x_offset += 20
#             y_offset += 20

            
        
#         crime_data = [name, str(age), gender, str(crime), str(timestamp)]
    
#     imgBackground[162:162 + 480, 55:55 + 640] = frame
#     cv2.imshow("Frame", imgBackground)
#     k = cv2.waitKey(1)
#     if k == ord('o'):
#         speak("Identify done..")
#         time.sleep(5)
#         if exist:
#             with open("Crime/Record_" + date + ".csv", "+a") as csvfile:
#                 writer = csv.writer(csvfile)
#                 writer.writerow(crime_data)
#             csvfile.close()
#         else:
#             with open("Crime/Record_" + date + ".csv", "+a") as csvfile:
#                 writer = csv.writer(csvfile)
#                 writer.writerow(COL_NAMES)
#                 writer.writerow(crime_data)
#             csvfile.close()
#     if k == ord('q'):
#         break
# video.release()
# cv2.destroyAllWindows()



import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from win32com.client import Dispatch

# Speak Function
def speak(message):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(message)

# Initialize Video Capture and Load Model
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

# Load Data
with open('data/names.pkl', 'rb') as file:
    LABELS = pickle.load(file)
with open('data/ages.pkl', 'rb') as file:
    AGES = pickle.load(file)
with open('data/crimes.pkl', 'rb') as file:
    CRIMES = pickle.load(file)
with open('data/genders.pkl', 'rb') as file:
    GENDERS = pickle.load(file)
with open('data/faces_data.pkl', 'rb') as file:
    FACES = pickle.load(file)

print('Shape of Faces matrix --> ', FACES.shape)

# Train KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load Background Image
imgBackground = cv2.imread("background.png")

# CSV Header
COL_NAMES = ['NAME', 'AGE', 'GENDER', 'CRIME', 'TIME']

# Main Loop
while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract and Resize Face
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        # Predict with KNN
        output = knn.predict(resized_img)[0]
        name = str(output)
        age = AGES[LABELS.index(name)]
        crime = CRIMES[LABELS.index(name)]
        gender = GENDERS[LABELS.index(name)]

        # Draw Face Rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display Details
        details_text = f"Name: {name}\nAge: {age}\nGender: {gender}\nCrime: {crime}"
        y_offset = y - 10 if y - 10 > 10 else y + h + 20
        for i, line in enumerate(details_text.split('\n')):
            cv2.putText(frame, line, (x, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Prepare Data for Saving
        timestamp = datetime.now().strftime("%H:%M:%S")
        date = datetime.now().strftime("%d-%m-%Y")
        crime_data = [name, str(age), gender, str(crime), timestamp]

        # Save Crime Data on Key Press
        if cv2.waitKey(1) & 0xFF == ord('o'):
            speak("Identifying...")
            time.sleep(1)

            # Save to CSV
            record_path = f"Crime/Record_{date}.csv"
            file_exists = os.path.isfile(record_path)
            with open(record_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(COL_NAMES)
                writer.writerow(crime_data)
            print("Data saved:", crime_data)

    # Add Frame to Background
    imgBackground[162:162 + frame.shape[0], 55:55 + frame.shape[1]] = frame
    cv2.imshow("Face Recognition", imgBackground)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video.release()
cv2.destroyAllWindows()
