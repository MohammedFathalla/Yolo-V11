from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('/home/mohamed/best.pt')  # Path to your YOLO model

# Path to the image
image_path = '/home/mohamed/tennis.png'  # Replace with your image path

# Read the image
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not read the image file.")
    exit()

# Run YOLO model on the image
results = model(image)

# Draw bounding boxes and labels on the image
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        label = box.cls  # Class index
        confidence = box.conf[0]  # Confidence score

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the label and confidence
        text = f"{model.names[int(label)]}: {confidence:.2f}"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# Save the processed image
output_path = '/home/mohamed/detected_tennis.jpg'  # Path to save the processed image
cv2.imwrite(output_path, image)

# Show the image with detections
cv2.imshow('YOLO Detection', image)
cv2.waitKey(0)  # Wait indefinitely until a key is pressed

# Close the image window
cv2.destroyAllWindows()

