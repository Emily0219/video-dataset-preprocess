from PIL import Image
from ultralytics import YOLO
import cv2
import os

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Run inference on 'bus.jpg'
image_path = 'Dataset/UCF101_n_frames/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/image_00001.jpg'
results = model(image_path)  # results list

# Open a file to save the results
with open('bounding_boxes.txt', 'w') as f:
    # Show the results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image
        im.save('results.jpg')  # save image

        # Write the bounding box data to the file
        f.write(str(r.boxes) + '\n')  # write the Boxes object containing the detection bounding boxes

# Get the full path of the folder
full_folder_path = os.path.dirname(image_path)

# Split the full path into parts
path_parts = full_folder_path.split('/')

# Find the index of the "UCF101_n_frames\ApplyEyeMakeup" part
index = path_parts.index('UCF101_n_frames') + 1

# Get the folder name after "UCF101_n_frames\ApplyEyeMakeup"
folder_name = path_parts[index]

# Print the folder name to the console
print("Action name: " + folder_name)

# Load the image
image = cv2.imread('results.jpg')

# Draw the action name on the image
text = "Action detected: " + folder_name
image = cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)

# Save the image
cv2.imwrite('results.jpg', image)