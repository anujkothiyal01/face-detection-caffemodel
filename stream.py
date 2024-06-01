import cv2
import numpy as np
import streamlit as st

st.title("Face Detection with Streamlit")

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe('./model/deploy.prototxt', './model/res10_300x300_ssd_iter_140000.caffemodel')

mean = [104, 117, 123]
scale = 1.0
in_width = 300
in_height = 300
detection_threshold = 0.5

font_style = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1

# Create a video capture object
video_cap = cv2.VideoCapture(0)

# Streamlit live video feed
frame_window = st.image([])

# Add a stop button with a unique key
stop_button = st.button("Stop", key="stop_button")

while video_cap.isOpened():
    has_frame, frame = video_cap.read()
    if not has_frame:
        st.write("No frames to display")
        break

    h = frame.shape[0]
    w = frame.shape[1]
    frame = cv2.flip(frame, 1)

    blob = cv2.dnn.blobFromImage(frame, scalefactor=scale, size=(in_width, in_height), mean=mean, swapRB=False,
                                 crop=False)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > detection_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype('int')

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'Confidence: {confidence:.4f}'
            label_size, base_line = cv2.getTextSize(label, font_style, font_scale, font_thickness)
            cv2.rectangle(frame, (x1, y1 - label_size[1]), (x1 + label_size[0], y1 + base_line), (255, 255, 255),
                          cv2.FILLED)
            cv2.putText(frame, label, (x1, y1), font_style, font_scale, (0, 0, 0))

    # Display the resulting frame in Streamlit
    frame_window.image(frame, channels='BGR')

    # Check if the stop button is clicked
    if stop_button:
        break

# Release the video capture object
video_cap.release()
