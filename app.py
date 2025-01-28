import streamlit as st
from pymongo import MongoClient
from bson.objectid import ObjectId
import hashlib
import cv2
import pickle
import numpy as np
import os
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

# MongoDB Setup
client = MongoClient("mongodb://localhost:27017/")
db = client['crime_branch']
users_collection = db['users']
data_collection = db['data']
faces_collection = db['faces']  # Collection for face data

DATA_PATH = "data/"  # Path to store .pkl files
os.makedirs(DATA_PATH, exist_ok=True)

# Utility functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    user = users_collection.find_one({"username": username, "password": hash_password(password)})
    return user is not None

def create_user(username, password):
    users_collection.insert_one({"username": username, "password": hash_password(password)})

def create_data_entry(data):
    data_collection.insert_one(data)

def read_data():
    data = list(data_collection.find())
    for item in data:
        item['_id'] = str(item['_id'])
        item['Face Verified'] = item.get('face_verified', 'Yes')
    return data

def update_data_entry(entry_id, updated_data):
    data_collection.update_one({"_id": ObjectId(entry_id)}, {"$set": updated_data})

def delete_data_entry(entry_id):
    data_collection.delete_one({"_id": ObjectId(entry_id)})

def save_pickle(filename, data):
    with open(os.path.join(DATA_PATH, filename), 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filename):
    filepath = os.path.join(DATA_PATH, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return []

# Face Registration
def register_face(name, age, crime, gender):
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

    faces_data = []
    st.warning("Press 'Q' to capture face.")

    while len(faces_data) < 100:
        ret, frame = video.read()
        if not ret:
            st.error("Error: Could not access camera.")
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            faces_data.append(resized_img.flatten())
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

        cv2.imshow("Registering Face", frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    if faces_data:
        existing_faces = load_pickle('faces_data.pkl')
        existing_names = load_pickle('names.pkl')
        existing_ages = load_pickle('ages.pkl')
        existing_crimes = load_pickle('crimes.pkl')
        existing_genders = load_pickle('genders.pkl')

        updated_faces = existing_faces + faces_data
        updated_names = existing_names + [name] * len(faces_data)
        updated_ages = existing_ages + [age] * len(faces_data)
        updated_crimes = existing_crimes + [crime] * len(faces_data)
        updated_genders = existing_genders + [gender] * len(faces_data)

        save_pickle('faces_data.pkl', updated_faces)
        save_pickle('names.pkl', updated_names)
        save_pickle('ages.pkl', updated_ages)
        save_pickle('crimes.pkl', updated_crimes)
        save_pickle('genders.pkl', updated_genders)

        faces_collection.insert_one({"name": name, "age": age, "crime": crime, "gender": gender, "face_verified": "Yes"})
        st.success(f"Face data for {name} has been registered.")
    else:
        st.error("No face detected. Try again.")

# Face Recognition
def recognize_face():
    st.header("Face Recognition")
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

    faces_data = load_pickle('faces_data.pkl')
    labels = load_pickle('names.pkl')
    ages = load_pickle('ages.pkl')
    crimes = load_pickle('crimes.pkl')
    genders = load_pickle('genders.pkl')

    if not faces_data or not labels:
        st.error("No registered faces found.")
        return

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(np.array(faces_data), np.array(labels))

    while True:
        ret, frame = video.read()
        if not ret:
            st.error("Error: Could not access camera.")
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)[0]
            index = labels.index(output)

            details = f"Name: {output}\nAge: {ages[index]}\nGender: {genders[index]}\nCrime: {crimes[index]}"
            st.text(details)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.putText(frame, output, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# Streamlit Web App
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state.authenticated = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials.")
else:
    st.sidebar.title("Dashboard")
    option = st.sidebar.selectbox("Select an option", ["Create", "Read", "Register Face", "Recognize Face", "Logout"])

    if option == "Create":
        st.header("Create Data Entry")
        field1 = st.text_input("Name of Officer")
        field2 = st.text_input("Department")

        if st.button("Save Data"):
            create_data_entry({"Name of Officer": field1, "Department": field2})
            st.success("Data entry created.")

    elif option == "Read":
        st.header("Read Data Entries")
        data = read_data()
        if data:
            st.table(data)
        else:
            st.info("No data available.")

    elif option == "Register Face":
        st.header("Face Registration")
        name = st.text_input("Enter Name:")
        age = st.text_input("Enter Age:")
        crime = st.text_input("Enter Criminal Activities:")
        gender = st.selectbox("Enter Gender:", ["Male", "Female", "Other"])

        if st.button("Register Face"):
            register_face(name, age, crime, gender)

    elif option == "Recognize Face":
        recognize_face()

    elif option == "Logout":
        st.session_state.authenticated = False
        st.rerun()
        st.info("Logged out successfully.")
