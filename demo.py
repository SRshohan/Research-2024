import cv2
from transformers import pipeline
from PIL import Image
import numpy as np
import torch

# Initialize the pipeline
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
pipe = pipeline(
    "image-classification",
    model="rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224",
    device=device
)

# Initialize video capture from webcam (use 0 for the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    # Convert the frame to RGB and PIL Image format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Perform inference
    outputs = pipe(pil_image)

    # Get the highest confidence prediction
    prediction = outputs[0]['label']
    confidence = outputs[0]['score']

    # Overlay the prediction on the frame
    text = f"{prediction}: {confidence:.2f}"
    cv2.putText(
        frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
        (0, 255, 0), 2, cv2.LINE_AA
    )

    # Display the resulting frame
    cv2.imshow('Real-Time Human Activity Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()
