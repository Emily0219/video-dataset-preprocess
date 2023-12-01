


#%%
import os
import pandas as pd
import numpy as np
import random
from torchvision import transforms
import cv2
from torch.utils.data import Dataset, DataLoader
import torch

#%%
from ultralytics import YOLO
import warnings


warnings.filterwarnings('ignore')

#%%

class ClipSubstractMean(object):
    def __init__(self, b=104, g=117, r=123):
        
        self.means = np.array((101.4, 97.7, 90.2))  # B=90.25, G=97.66, R=101.41

    def __call__(self, buffer):
        new_buffer = buffer - self.means
        return new_buffer


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=(112, 112)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, buffer):
        h, w = buffer.shape[1], buffer.shape[2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        new_buffer = np.zeros((buffer.shape[0], new_h, new_w, 3))
        for i in range(buffer.shape[0]):
            image = buffer[i, :, :, :]
            image = image[top: top + new_h, left: left + new_w]
            new_buffer[i, :, :, :] = image

        return new_buffer


class CenterCrop(object):
    """Crop the image in a sample at the center.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=(112, 112)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
            
        self.model = YOLO("yolov8m.pt")

    def __call__(self, buffer):
        h, w = buffer.shape[1], buffer.shape[2]
        new_h, new_w = self.output_size

        top = int(round(h - new_h) / 2.)
        left = int(round(w - new_w) / 2.)

        # new_buffer = np.zeros((buffer.shape[0], new_h, new_w, 3))
        # for i in range(buffer.shape[0]):
        #     image = buffer[i, :, :, :]
        #     image = image[top: top + new_h, left: left + new_w]
        #     new_buffer[i, :, :, :] = image
        new_buffer = self.process_buffer_with_yolo(buffer)

        return new_buffer
    
    def process_buffer_with_yolo(self, buffer):
        # h, w = buffer.shape[1], buffer.shape[2]
        new_h, new_w = self.output_size
        new_buffer = np.zeros((buffer.shape[0], new_h, new_w, 3))
        for i in range(buffer.shape[0]):
            image = buffer[i, :, :, :]
            h, w = image.shape[:2]
            top = int(round((h - new_h) / 2.))
            left = int(round((w - new_w) / 2.))
            image = image[top: top + new_h, left: left + new_w]
            results = self.model.predict(image)
            for box in results.boxes:
                cords = box.xyxy[0].tolist()
                class_id = box.cls[0].item()
                start = (int(cords[0]), int(cords[1]))  # x0, y0
                end = (int(cords[2]), int(cords[3]))  # x1, y1
                cv2.rectangle(image, start, end, (0, 200, 0), thickness=2)
                cv2.putText(image, results.names[class_id], (start[0] + 15, start[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 0, 10), 2)
            new_buffer[i, :, :, :] = image
        return new_buffer


class RandomHorizontalFlip(object):
    """Horizontally flip the given Images randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __call__(self, buffer):
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, buffer):
        # swap color axis because
        # numpy image: batch_size x H x W x C
        # torch image: batch_size x C X H X W
        img = torch.from_numpy(buffer.transpose((3, 0, 1, 2)))

        return img.float().div(255)


class UCFDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            root_dir (str): path to n_frames_jpg folders.
            info_list (str): path to annotation file.
            split(str): whether create trainset. Default='train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.

    """

    def __init__(self, root_dir, info_list, split='train', clip_len=16):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.label_class_map = {}
        with open('./Dataset/ucfTrainTestlist/classInd.txt') as f:
            for line in f:
                line=line.strip('\n').split(' ')
                self.label_class_map[line[1]] = line[0]
        self.split = split
        if split == 'train':
            self.transform = transforms.Compose(
                [ClipSubstractMean(),
                 RandomCrop(),
                 RandomHorizontalFlip(),
                 ToTensor()])
        else:
            self.transform = transforms.Compose(
                [ClipSubstractMean(),
                 CenterCrop(),
                 ToTensor()])
        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, index):
        # Loading and preprocessing.
        video_path = self.landmarks_frame.iloc[index, 0]
        # labels [0,100]
        if self.landmarks_frame.shape[1] == 2:
            labels = self.landmarks_frame.iloc[index, 1] - 1
        else:
            classes = video_path.split('/')[0]
            labels = int(self.label_class_map[classes]) - 1
        buffer = self.get_resized_frames_per_video(video_path)

        if self.transform:
            buffer = self.transform(buffer)

        return buffer, torch.from_numpy(np.array(labels))

    def get_resized_frames_per_video(self, video_path):
        slash_rows = video_path.split('.')
        dir_name = slash_rows[0]
        video_jpgs_path = os.path.join(self.root_dir, dir_name)

        # Read the number of frames from the n_frames file
        data = pd.read_csv(os.path.join(video_jpgs_path, 'n_frames'), delimiter=' ', header=None)
        frame_count = data[0][0]

        # Initialize an array to store all frames
        video_x = np.empty((self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))

        for i in range(self.clip_len):
            # Compute the frame number based on the clip length
            frame_number = (i * frame_count) // self.clip_len + 1

            # Generate the image filename based on the frame number
            s = "%05d" % frame_number
            image_name = 'image_' + s + '.jpg'
            image_path = os.path.join(video_jpgs_path, image_name)

            if os.path.exists(image_path):
                # Read and resize the image
                tmp_image = cv2.imread(image_path)
                tmp_image = cv2.resize(tmp_image, (self.resize_width, self.resize_height))
                tmp_image = np.array(tmp_image).astype(np.float64)
                tmp_image = tmp_image[:, :, ::-1]  # BGR -> RGB

                # Store the frame in the array
                video_x[i, :, :, :] = tmp_image

        return video_x
    
    




if __name__ == '__main__':
    # usage
    root_list = './Dataset/UCF101_n_frames/'
    info_list = './Dataset/ucfTrainTestlist/testlist01.txt'

    # trainUCF101 = UCFDataset(root_list, info_list,'test'
    #                           )
    # testUCF101 = UCFDataset(root_list, info_list,
    #                          transform=transforms.Compose(
    #                              [ClipSubstractMean(),
    #                               CenterCrop(),
    #                               ToTensor()]))

    # dataloader = DataLoader(trainUCF101, batch_size=8, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(UCFDataset(root_dir='./Dataset/UCF101_n_frames',
                                              info_list='./Dataset/ucfTrainTestlist/testlist01.txt',
                                              split='test',
                                              clip_len=16),
                                 batch_size=8, shuffle=True,num_workers=0)
    
    torch.save(test_dataloader, 'processed_dataset.pth')

    for i_batch, (images, targets) in enumerate(test_dataloader):
        print(i_batch, images.size(), targets.size())