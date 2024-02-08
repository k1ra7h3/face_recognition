import cv2
import numpy as np
import tokenizers
import face_recognition_models

face = cv2.data.haarcascades+"haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(face)

# Training data
faces = np.array([
    cv2.imread('kev.jpg', 0),
    cv2.imread('kk.jpg', 0)
])

stream = tokenizers.open('kev.jpg')  # @UndefinedVariable
contents = stream.read()
stream.close()



labels = np.array([0, 1])

# Create the recognizer and train it on the training data
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

# Initialize the video capture device
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Extract the face from the frame
        face = gray[y:y+h, x:x+w]

        # Resize the face to a standard size
        face_resized = cv2.resize(face, (100, 100))

        # Predict the label for the face
        label, confidence = recognizer.predict(face_resized)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the predicted label and confidence level
        cv2.putText(frame, f"Label: {label} Confidence: {confidence}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the resulting frame
    cv2.imshow('Video', frame)


    def execfile(cap, globals=None, locals=None):
        if globals is None:
            import sys
            globals = sys._getframe(1).f_globals
        if locals is None:
            locals = sys._getframe(1).f_locals
        with open(cap, "r") as execfile:
            exec(compile(contents + "\n", cap, 'exec'), globals, locals)


    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()