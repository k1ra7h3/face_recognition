import cv2
import face_recognition
import pandas as pd
from datetime import datetime

# Create a list to store student names and IDs
student_names = []
student_ids = []

# Load images of known faces and encode them
known_face_encodings = []
known_face_names = []

# Load data from Excel file (if exists)
try:
    attendance_data = pd.read_excel("attendance.xlsx")
    student_names = attendance_data["Name"].tolist()
    student_ids = attendance_data["ID"].tolist()
except:
    attendance_data = pd.DataFrame(columns=["Time", "Name", "ID"])

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and their encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in the current frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the current face encoding with the known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # Find the best match index
        match_index = None
        if True in matches:
            match_index = matches.index(True)

        # If a match was found, get the name and ID of the student
        if match_index is not None:
            name = known_face_names[match_index]
            id = student_ids[student_names.index(name)]

            # Check if the student has already been marked as present
            if id not in attendance_data["ID"].tolist():
                # Add the attendance record to the Excel file
                time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                attendance_data = attendance_data.append({"Time": time, "Name": name, "ID": id}, ignore_index=True)
                attendance_data.to_excel("attendance.xlsx", index=False)

        # If no match was found, ask the user if they want to register a new face
        else:
            cv2.imshow('Video', frame)
            key = cv2.waitKey(1)

            if key == ord('r'):
                # Ask the user to input the name and ID of the new student
                name = input("Enter student name: ")
                id = input("Enter student ID: ")

                # Add the new student to the list of known faces
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)
                student_names.append(name)
                student_ids.append(id)

                # Save the new face encoding to the image file
                cv2.imwrite("known_faces/{}.jpg".format(name), frame)

                # Add the attendance record to the Excel file
                time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                attendance_data = attendance_data.append({"Time": time, "Name": name, "ID": id}, ignore_index=True)
                attendance_data.to_excel("attendance.xlsx", index=False)

                print("New student registered:", name)

    # Display the video capture with the detected faces
    cv2.imshow('Video', frame)

    # Exit the program when the user presses
