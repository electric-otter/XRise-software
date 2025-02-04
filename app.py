import os
os.system("pip install cv2")
os.system("pip install numpy")
os.system("pip install pyaudiogui")
os.system("pip install time")
import cv2
import numpy as np
import pyautogui
import time

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use your camera (0 is usually the default)

# Load pre-trained model for human detection (Haar Cascade for simplicity)
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Starting positions for overlay (top-left corner of AR elements)
overlay_x, overlay_y = 50, 50

# Flag to check if we're dragging
dragging = False
start_x, start_y = 0, 0

def mouse_callback(event, x, y, flags, param):
    global overlay_x, overlay_y, dragging, start_x, start_y
    if event == cv2.EVENT_LBUTTONDOWN:
        # Start dragging
        dragging = True
        start_x, start_y = x - overlay_x, y - overlay_y
    elif event == cv2.EVENT_LBUTTONUP:
        # Stop dragging
        dragging = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            # Update overlay position
            overlay_x = x - start_x
            overlay_y = y - start_y

# Set up mouse callback for window interaction
cv2.namedWindow('XRise AR')
cv2.setMouseCallback('XRise AR', mouse_callback)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is not captured correctly, break out of the loop
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale for human detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect people in the frame
    bodies = body_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around detected bodies
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Get the current time and desktop screenshot
    current_time = time.strftime("%H:%M:%S")  # Current time in HH:MM:SS format
    desktop = pyautogui.screenshot()  # Capture desktop screenshot
    desktop = np.array(desktop)  # Convert to NumPy array for OpenCV

    # Resize desktop screenshot to fit overlay
    desktop = cv2.resize(desktop, (frame.shape[1], frame.shape[0]))

    # Overlay the desktop screenshot (transparency can be adjusted here)
    overlay = cv2.addWeighted(frame, 0.5, desktop, 0.5, 0)

    # Display the current time at the moved position
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(overlay, current_time, (overlay_x, overlay_y), font, 1, (255, 255, 255), 2)

    # Show the AR screen with the overlay
    cv2.imshow('XRise AR', overlay)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
