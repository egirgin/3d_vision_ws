import cv2
import time, os

# Define the path to save the images
save_path = './logitech/captured_images'

# Create the directory if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error opening video stream or device")
    exit()

# Set the image saving counter
image_count = 0

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame was read correctly
    if not ret:
        print("Error reading frame from camera")
        break

    # Display the frame
    cv2.imshow('frame', frame)

    # Wait for a key press
    key = cv2.waitKey(1)

    # If the spacebar is pressed, save the image
    if key == 32:
        # Create the filename
        filename = f"image_{image_count}.jpg"
        save_path_with_filename = os.path.join(save_path, filename)

        # Save the image
        cv2.imwrite(save_path_with_filename, frame)
        print(f"Image saved as {save_path_with_filename}")

        # Increment the image count
        image_count += 1

    # If the 'q' key is pressed, exit the loop
    if key == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()