from PIL import Image
from ultralytics import YOLO
import cv2
import os

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n-pose.pt')

# Run inference on 'bus.jpg'
image_path = 'Dataset/UCF101_n_frames/Bowling/v_Bowling_g01_c01/image_00001.jpg'
results = model(image_path)  # results list

# Get the image name without extension
image_name = os.path.splitext(os.path.basename(image_path))[0]

# Create the bounding box file name
bbox_file_name = image_name + '.txt'

# Get the full path of the folder
full_folder_path = os.path.dirname(image_path)

# Replace 'UCF101_n_frames' with 'UCF-yolo' in the full folder path
new_folder_path = full_folder_path.replace('UCF101_n_frames', 'UCF-yolo')

# Create the new folder path if it doesn't exist
os.makedirs(new_folder_path, exist_ok=True)

# Create the new image and bounding box file paths
new_image_path = os.path.join(new_folder_path, image_name + '.jpg')
new_bbox_file_path = os.path.join(new_folder_path, bbox_file_name)

# Show the results
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image

    # Save the image and the bounding box file
    im.save(new_image_path)
    with open(new_bbox_file_path, 'w') as f:
        f.write(str(r.boxes) + '\n')  # write the Boxes object containing the detection bounding boxes

# Split the full path into parts
path_parts = full_folder_path.split('/')

# Find the index of the "UCF101_n_frames\ApplyEyeMakeup" part
index = path_parts.index('UCF101_n_frames') + 1

# Get the folder name after "UCF101_n_frames\ApplyEyeMakeup"
folder_name = path_parts[index]

# Print the folder name to the console
print("Action name: " + folder_name)

# Load the image
image = cv2.imread(new_image_path)

# Draw the action name on the image
text = "Action detected: " + folder_name
image = cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

# Save the image
cv2.imwrite(new_image_path, image)









# from PIL import Image
# from ultralytics import YOLO
# import cv2
# import os

# # Load a pretrained YOLOv8n model
# model = YOLO('yolov8n-pose.pt')

# # Define the root directory
# root_dir = 'Dataset/UCF101_n_frames'

# # Walk through all files in the directory
# for dirpath, dirnames, filenames in os.walk(root_dir):
#     for filename in filenames:
#         # Only process .jpg files
#         if filename.endswith('.jpg'):
#             # Get the full path of the file
#             image_path = os.path.join(dirpath, filename)

#             # Run inference on the image
#             results = model(image_path)

#             # Get the image name without extension
#             image_name = os.path.splitext(filename)[0]

#             # Create the bounding box file name
#             bbox_file_name = image_name + '.txt'

#             # Replace 'UCF101_n_frames' with 'UCF-yolo' in the full folder path
#             new_folder_path = dirpath.replace('UCF101_n_frames', 'UCF-yolo')

#             # Create the new folder path if it doesn't exist
#             os.makedirs(new_folder_path, exist_ok=True)

#             # Create the new image and bounding box file paths
#             new_image_path = os.path.join(new_folder_path, image_name + '.jpg')
#             new_bbox_file_path = os.path.join(new_folder_path, bbox_file_name)

#             # Show the results
#             for r in results:
#                 im_array = r.plot()  # plot a BGR numpy array of predictions
#                 im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image

#                 # Save the image and the bounding box file
#                 im.save(new_image_path)
#                 with open(new_bbox_file_path, 'w') as f:
#                     f.write(str(r.boxes) + '\n')  # write the Boxes object containing the detection bounding boxes

#             # Use os.path.sep to get the platform-specific directory separator
#             path_parts = dirpath.split(os.path.sep)
#             print(path_parts)

#             # Find the index of the "UCF101_n_frames" part
#             try:
#                 index = path_parts.index('UCF101_n_frames') + 1

#                 # Get the folder name after "UCF101_n_frames"
#                 folder_name = path_parts[index]

#                 # Print the folder name to the console
#                 print("Action name: " + folder_name)

#                 # Load the image
#                 image = cv2.imread(new_image_path)

#                 # Draw the action name on the image
#                 text = "Action detected: " + folder_name
#                 image = cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

#                 # Save the image
#                 cv2.imwrite(new_image_path, image)
#             except ValueError:
#                 print(f"Error: 'UCF101_n_frames' not found in path: {dirpath}")
