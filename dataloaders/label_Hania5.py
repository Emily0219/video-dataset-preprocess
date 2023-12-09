# import os
# from pathlib import Path
# import cv2
# from PIL import Image, ImageDraw
# from torchvision.transforms import functional as F
# from torch import no_grad
# from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
# from ultralytics import YOLO
# import numpy as np

# def process_video(video_path, output_dir):
#     """
#     Process a video file, generate YOLO annotations, and visualize bounding boxes.

#     Args:
#         video_path: Path to the video file.
#         output_dir: Directory to store the YOLO annotations and images with bounding boxes.
#     """
#     # Initialize models
#     action_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1).eval()
#     model = YOLO("yolov8.pt")

#     # Initialize video capture
#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened():
#         raise RuntimeError(f"Error opening video file: {video_path}")

#     # Initialize counters
#     total_labeled_frames = 0

#     # Main processing loop
#     while True:
#         # Capture frame
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert frame to PIL Image and PyTorch tensor
#         image = Image.fromarray(frame)
#         image_tensor = F.to_tensor(image)

#         # Get action and object predictions
#         with no_grad():
#             action_prediction = action_model([image_tensor])
#             object_results = model.predict(image)

#         # Process object predictions and generate YOLO annotations
#         yolo_annotations = []
#         for box, confidence, class_id in object_results.xyxy[0]:
#             x1, y1, x2, y2 = box
#             x_center = (x1 + x2) / 2
#             y_center = (y1 + y2) / 2
#             width = x2 - x1
#             height = y2 - y1
#             frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
#             yolo_annotations.append(f"{int(class_id)} {x_center} {y_center} {width} {height} {frame_number}")

#         # Save YOLO annotations
#         output_annotation_file_path = Path(output_dir) / Path(video_path).stem + ".txt"
#         with open(output_annotation_file_path, "a") as f:
#             f.write("\n".join(yolo_annotations))

#         # Draw bounding boxes for action and objects
#         draw = ImageDraw.Draw(image)
#         for box, label in zip(action_prediction[0]["boxes"], action_prediction[0]["labels"]):
#             x, y, width, height = box.tolist()
#             class_id = label.item()

#             # Convert normalized coordinates to image coordinates
#             x = x * image.width
#             y = y * image.height
#             width = width * image.width
#             height = height * image.height

#             # Draw bounding box for action (blue)
#             draw.rectangle([x, y, x + width, y + height], outline="blue", width=2)

#         for box, confidence, class_id in object_results.xyxy[0]:
#             x1, y1, x2, y2 = box
#             x = x1 * image.width
#             y = y1 * image.height
#             width = (x2 - x1) * image.width
#             height = (y2 - y1) * image.height

#             # Draw bounding box for object (red)
#             draw.rectangle([x, y, x + width, y + height], outline="red", width=2)

#         # Convert image back to OpenCV format and optionally display
#         frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#         # Convert image back to OpenCV format
#         frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#         # Save annotated image with frame number in filename
#         frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
#         output_image_file_path = Path(output_dir) / f"{Path(video_path).stem}_frame_{frame_number}.jpg"
#         cv2.imwrite(str(output_image_file_path), frame)

#         # Optionally display the annotated image
#         # cv2.imshow("Video", frame)
#         # cv2.waitKey(1)
 
#         # cv2.imshow("Video", frame)
#         # cv2.waitKey(1)

#         # Increment frame counter
#         total_labeled_frames += 1

#     # Release video capture
#     cap.release()


#     # Print completion message
#     print(f"Successfully processed video: {video_path}. Total labeled frames: {total_labeled_frames}")






# import os
# from pathlib import Path
# import cv2
# from PIL import Image, ImageDraw
# from torchvision.transforms import functional as F
# from torch import no_grad
# from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
# # Assuming your YOLOv5 directory is at the same level as "dataloaders" directory
# import sys 
# sys.path.append("./yolov5")
# from yolov5 import YOLOv5

# import numpy as np

# def process_videos(video_folder, output_dir):
#     """
#     Process all video files in a folder, generate YOLO annotations, and visualize bounding boxes.

#     Args:
#         video_folder: Path to the folder containing video files.
#         output_dir: Directory to store the YOLO annotations and images with bounding boxes.
#     """
#     # Initialize models
#     action_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1).eval()
#     object_model = YOLOv5("yolov5s6.pt", device="cuda")

#     # Process each video file in the folder
#     for video_path in Path(video_folder).glob("*.avi"):
#         process_video(video_path, output_dir, action_model, object_model)


# def process_video(video_path, output_dir, action_model, object_model):
#     """
#     Process a video file, generate YOLO annotations, and visualize bounding boxes.

#     Args:
#         video_path: Path to the video file.
#         output_dir: Directory to store the YOLO annotations and images with bounding boxes.
#         action_model: Pre-trained Faster R-CNN model for action detection.
#         object_model: Pre-trained YOLOv5 model for object detection.
#     """
#     # Initialize video capture
#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened():
#         raise RuntimeError(f"Error opening video file: {video_path}")

#     # Initialize counters
#     total_labeled_frames = 0

#     # Main processing loop
#     while True:
#         # Capture frame
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert frame to PIL Image and PyTorch tensor
#         image = Image.fromarray(frame)
#         image_tensor = F.to_tensor(image)

#         # Get action and object predictions
#         with no_grad():
#             action_prediction = action_model([image_tensor])
#             object_results = object_model.predict(image)

#         # Process object predictions and generate YOLO annotations
#         yolo_annotations = []
#         for box, confidence, class_id in object_results.xyxy[0]:
#             x1, y1, x2, y2 = box
#             x_center = (x1 + x2) / 2
#             y_center = (y1 + y2) / 2
#             width = x2 - x1
#             height = y2 - y1
#             frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
#             yolo_annotations.append(f"{int(class_id)} {x_center} {y_center} {width} {height} {frame_number}")

#         # Save YOLO annotations
#         output_annotation_file_path = Path(output_dir) / Path(video_path).stem + ".txt"
#         with open(output_annotation_file_path, "a") as f:
#             f.write("\n".join(yolo_annotations))

#         # Draw bounding boxes for action and objects
#         draw = ImageDraw.Draw(image)
#         for box, label in zip(action_prediction[0]["boxes"], action_prediction[0]["labels"]):
#             x, y, width, height = box.tolist()
#             class_id = label.item()

#             # Convert normalized coordinates to image coordinates
#             x = x * image.width
#             y = y * image.height
#             width = width * image.width
#             height = height * image.height

#             # Draw bounding box for action (blue)
#             draw.rectangle([x, y, x + width, y + height], outline="blue", width=2)

#         for box, confidence, class_id in object_results.xyxy[0]:
#             x1, y1, x2, y2 = box
#             x = x1 * image.width
#             y = y1 * image.height
#             width = (x2 - x1) * image.width
#             height = (y2 - y1) * image.height

#             # Draw bounding box for object (red)
#             draw.rectangle([x, y, x + width, y + height], outline="red", width=2)

#         # Convert image back to OpenCV format
#         frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#         # Save annotated image with frame number in filename
#         frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
#         output_image_file_path = Path(output_dir) / f"{Path(video_path).stem}_frame_{frame_number}.jpg"
#         cv2.imwrite(str(output_image_file_path), frame)

#         # Increment frame counter
#         total_labeled_frames += 1

#     # Release video capture
#     cap.release()

#     # Print completion message
#     print(f"Successfully processed video: {video_path}. Total labeled frames: {total_labeled_frames}")

# # Example Usage
# video_folder = "./Dataset/UCF-101"
# output_dir = "./Dataset/UCF-labels"

# process_videos(video_folder, output_dir)











