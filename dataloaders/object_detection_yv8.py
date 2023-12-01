from ultralytics import YOLO
import warnings
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

# Load the preprocessed dataset
processed_dataset = torch.load('processed_dataset.pth')

# Assuming that your processed dataset is an instance of UCFDataset
# Modify this line based on the actual structure of your processed dataset
test_dataloader = DataLoader(processed_dataset, batch_size=8, shuffle=True, num_workers=0)

model = YOLO("yolov8m.pt")

# Assuming you want to process only one batch from the dataloader
for i_batch, (images, targets) in enumerate(test_dataloader):
    if i_batch == 0:  # Process only the first batch for illustration
        images = images.numpy()

        # Perform inference on the images
        results = model.predict(images)

        # Assuming you want to visualize the results for the first image in the batch
        result = results[0]
        image = images[0]

        cords = result.boxes.xyxy[0].tolist()
        class_id = result.boxes.cls[0].item()
        conf = result.boxes.conf[0].item()

        print("Object type:", result.names[class_id])
        print("Coordinates:", cords)
        print("Probability:", conf)

        # Visualization on the image
        image = np.transpose(image, (1, 2, 0)) * 255  # Convert from torch format to numpy
        image = image.astype(np.uint8)
        
        start = (int(cords[0]), int(cords[1]))  # x0, y0
        end = (int(cords[2]), int(cords[3]))  # x1, y1

        cv2.rectangle(image, start, end, (0, 200, 0), thickness=2)
        cv2.putText(image, result.names[class_id], (start[0] + 15, start[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (10, 0, 10), 2)

        cv2.imshow("Object Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#%%

#copilot

# from ultralytics import YOLO
# import cv2
# import warnings

# warnings.filterwarnings('ignore')

# class CenterCrop(object):
#     # ...

#     def _init_(self, output_size=(112, 112)):
#         # ...
#         self.model = YOLO("yolov8m.pt")

#     def _call_(self, buffer):
#         # ...
#         new_buffer = self.process_buffer_with_yolo(buffer)
#         return new_buffer

#     def process_buffer_with_yolo(self, buffer):
#         new_buffer = np.zeros((buffer.shape[0], new_h, new_w, 3))
#         for i in range(buffer.shape[0]):
#             image = buffer[i, :, :, :]
#             image = image[top: top + new_h, left: left + new_w]
#             results = self.model.predict(image)
#             for box in results.boxes:
#                 cords = box.xyxy[0].tolist()
#                 class_id = box.cls[0].item()
#                 start = (int(cords[0]), int(cords[1]))  # x0, y0
#                 end = (int(cords[2]), int(cords[3]))  # x1, y1
#                 cv2.rectangle(image, start, end, (0, 200, 0), thickness=2)
#                 cv2.putText(image, results.names[class_id], (start[0] + 15, start[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 0, 10), 2)
#             new_buffer[i, :, :, :] = image
#         return new_buffer

# class RandomHorizontalFlip(object):
#     # ...
