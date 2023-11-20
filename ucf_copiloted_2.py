
#%%
#video2jpg_ucf101_hmdb51.py

import os
import sys
import subprocess

# 把avi视频文件解码，逐帧提取成图像文件
def class_process(dir_path, dst_dir_path, class_name):
    class_path = os.path.join(dir_path, class_name)
    if not os.path.isdir(class_path):
        return

    dst_class_path = os.path.join(dst_dir_path, class_name)
    if not os.path.exists(dst_class_path):
        os.mkdir(dst_class_path)

    for file_name in os.listdir(class_path):
        if '.avi' not in file_name:
            continue
        name, ext = os.path.splitext(file_name)
        #提取每个.avi文件的名字部分，以便在.../UCF101_n_frames/XX/创建**.avi文件的名字命名**的文件夹
        dst_directory_path = os.path.join(dst_class_path, name)

        video_file_path = os.path.join(class_path, file_name)
        try:
            if os.path.exists(dst_directory_path):
                if not os.path.exists(os.path.join(dst_directory_path, 'image_00001.jpg')):
                    subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
                    print('remove {}'.format(dst_directory_path))
                    os.mkdir(dst_directory_path)
                    #创建**.avi文件的名字命名**的文件夹
                else:
                    continue
            else:
                os.mkdir(dst_directory_path)
        except:
            print(dst_directory_path)
            continue
        cmd = 'ffmpeg -i \"{}\" -vf scale=-1:240 \"{}/image_%05d.jpg\"'.format(video_file_path, dst_directory_path)
        #视频的读取与逐帧图像保存
        print(cmd)
        subprocess.call(cmd, shell=True)
        print('\n')

if __name__=="__main__":
    # dir_path = sys.argv[1]
    # dst_dir_path = sys.argv[2]

    # dir_path = '/home/ran/mnt1/Dataset/hmdb51_org/'
    # dst_dir_path = '/home/ran/mnt1/Dataset/hmdb51_n_frames'

    dir_path = 'Dataset/UCF-101/'
    dst_dir_path = 'Dataset/UCF101_n_frames'

    for class_name in os.listdir(dir_path):
        class_process(dir_path, dst_dir_path, class_name)
        
#%%
#n_frames_ucf101_hmdb51.py

from __future__ import print_function, division
import os
import sys
import subprocess

# 统计每个视频截取的图像帧数
def class_process(dir_path, class_name):
    class_path = os.path.join(dir_path, class_name)
    if not os.path.isdir(class_path):
        return

    for file_name in os.listdir(class_path):
        video_dir_path = os.path.join(class_path, file_name)
        image_indices = []
        for image_file_name in os.listdir(video_dir_path):
            if 'image' not in image_file_name:
                continue
            image_indices.append(int(image_file_name[6:11]))

        if len(image_indices) == 0:
            print('no image files', video_dir_path)
            n_frames = 0
        else:
            image_indices.sort(reverse=True)
            n_frames = image_indices[0]
            print(video_dir_path, n_frames)
        with open(os.path.join(video_dir_path, 'n_frames'), 'w') as dst_file:
            dst_file.write(str(n_frames))


if __name__=="__main__":
    # dir_path = sys.argv[1]
    # dir_path = '/home/ran/mnt1/Dataset/hmdb51_n_frames'
    dir_path = './Dataset/UCF101_n_frames'
    for class_name in os.listdir(dir_path):
        class_process(dir_path, class_name)

#%%
#ucf_dataset.py



import os
import pandas as pd
import numpy as np
import random
from torchvision import transforms
import cv2
from torch.utils.data import Dataset, DataLoader
import torch


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

    def __call__(self, buffer):
        h, w = buffer.shape[1], buffer.shape[2]
        new_h, new_w = self.output_size

        top = int(round(h - new_h) / 2.)
        left = int(round(w - new_w) / 2.)

        new_buffer = np.zeros((buffer.shape[0], new_h, new_w, 3))
        for i in range(buffer.shape[0]):
            image = buffer[i, :, :, :]
            image = image[top: top + new_h, left: left + new_w]
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
        # get the random continuous 16 frame
        data = pd.read_csv(os.path.join(video_jpgs_path, 'n_frames'), delimiter=' ', header=None)
        frame_count = data[0][0]
        video_x = np.empty((self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        image_start = random.randint(1, abs(frame_count - self.clip_len))
        for i in range(self.clip_len):
            s = "%05d" % (i + image_start)
            image_name = 'image_' + s + '.jpg'
            image_path = os.path.join(video_jpgs_path, image_name)
            tmp_image = cv2.imread(image_path)
            tmp_image = cv2.resize(tmp_image, (self.resize_width, self.resize_height))
            tmp_image = np.array(tmp_image).astype(np.float64)
            tmp_image = tmp_image[:, :, ::-1]    # BGR -> RGB
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

    for i_batch, (images, targets) in enumerate(test_dataloader):
        print(i_batch, images.size(), targets.size())
        
        
#%%
#copy_of_video.py


# -*- coding: utf-8 -*-
"""Copy of video.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bNcwKVlxN31B-xu7Qf08Mj-QK400BA2A

##### Copyright 2022 The TensorFlow Authors.
"""

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""# Load video data

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/tutorials/load_data/video"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/load_data/video.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/load_data/video.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/load_data/video.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
</table>

This tutorial demonstrates how to load and preprocess [AVI](https://en.wikipedia.org/wiki/Audio_Video_Interleave) video data using the [UCF101 human action dataset](https://www.tensorflow.org/datasets/catalog/ucf101). Once you have preprocessed the data, it can be used for such tasks as video classification/recognition, captioning or clustering. The original dataset contains realistic action videos collected from YouTube with 101 categories, including playing cello, brushing teeth, and applying eye makeup. You will learn how to:

* Load the data from a zip file.

* Read sequences of frames out of the video files.

* Visualize the video data.

* Wrap the frame-generator [`tf.data.Dataset`](https://www.tensorflow.org/guide/data).

This video loading and preprocessing tutorial is the first part in a series of TensorFlow video tutorials. Here are the other three tutorials:

- [Build a 3D CNN model for video classification](https://www.tensorflow.org/tutorials/video/video_classification): Note that this tutorial uses a (2+1)D CNN that decomposes the spatial and temporal aspects of 3D data; if you are using volumetric data such as an MRI scan, consider using a 3D CNN instead of a (2+1)D CNN.
- [MoViNet for streaming action recognition](https://www.tensorflow.org/hub/tutorials/movinet): Get familiar with the MoViNet models that are available on TF Hub.
- [Transfer learning for video classification with MoViNet](https://www.tensorflow.org/tutorials/video/transfer_learning_with_movinet): This tutorial explains how to use a pre-trained video classification model trained on a different dataset with the UCF-101 dataset.

## Setup

Begin by installing and importing some necessary libraries, including:
[remotezip](https://github.com/gtsystem/python-remotezip) to inspect the contents of a ZIP file, [tqdm](https://github.com/tqdm/tqdm) to use a progress bar, [OpenCV](https://opencv.org/) to process video files, and [`tensorflow_docs`](https://github.com/tensorflow/docs/tree/master/tools/tensorflow_docs) for embedding data in a Jupyter notebook.
"""

# The way this tutorial uses the `TimeDistributed` layer requires TF>=2.10
!pip install -U "tensorflow>=2.10.0"

!pip install remotezip tqdm opencv-python
!pip install -q git+https://github.com/tensorflow/docs

import tqdm
import random
import pathlib
import itertools
import collections

import os
import cv2
import numpy as np
import remotezip as rz

import tensorflow as tf

# Some modules to display an animation using imageio.
import imageio
from IPython import display
from urllib import request
from tensorflow_docs.vis import embed

"""## Download a subset of the UCF101 dataset

The [UCF101 dataset](https://www.tensorflow.org/datasets/catalog/ucf101) contains 101 categories of different actions in video, primarily used in action recognition. You will use a subset of these categories in this demo.
"""

URL = 'https://storage.googleapis.com/thumos14_files/UCF101_videos.zip'

"""The above URL contains a zip file with the UCF 101 dataset. Create a function that uses the `remotezip` library to examine the contents of the zip file in that URL:"""

def list_files_from_zip_url(zip_url):
  """ List the files in each class of the dataset given a URL with the zip file.

    Args:
      zip_url: A URL from which the files can be extracted from.

    Returns:
      List of files in each of the classes.
  """
  files = []
  with rz.RemoteZip(zip_url) as zip:
    for zip_info in zip.infolist():
      files.append(zip_info.filename)
  return files

files = list_files_from_zip_url(URL)
files = [f for f in files if f.endswith('.avi')]
files[:10]

"""Begin with a few videos and a limited number of classes for training. After running the above code block, notice that the class name is included in the filename of each video.

Define the `get_class` function that retrieves the class name from a filename. Then, create a function called `get_files_per_class` which converts the list of all files (`files` above) into a dictionary listing the files for each class:
"""

def get_class(fname):
  """ Retrieve the name of the class given a filename.

    Args:
      fname: Name of the file in the UCF101 dataset.

    Returns:
      Class that the file belongs to.
  """
  return fname.split('_')[-3]

def get_files_per_class(files):
  """ Retrieve the files that belong to each class.

    Args:
      files: List of files in the dataset.

    Returns:
      Dictionary of class names (key) and files (values).
  """
  files_for_class = collections.defaultdict(list)
  for fname in files:
    class_name = get_class(fname)
    files_for_class[class_name].append(fname)
  return files_for_class

"""Once you have the list of files per class, you can choose how many classes you would like to use and how many videos you would like per class in order to create your dataset."""

NUM_CLASSES = 10
FILES_PER_CLASS = 50

files_for_class = get_files_per_class(files)
classes = list(files_for_class.keys())

print('Num classes:', len(classes))
print('Num videos for class[0]:', len(files_for_class[classes[0]]))

"""Create a new function called `select_subset_of_classes` that selects a subset of the classes present within the dataset and a particular number of files per class:"""

def select_subset_of_classes(files_for_class, classes, files_per_class):
  """ Create a dictionary with the class name and a subset of the files in that class.

    Args:
      files_for_class: Dictionary of class names (key) and files (values).
      classes: List of classes.
      files_per_class: Number of files per class of interest.

    Returns:
      Dictionary with class as key and list of specified number of video files in that class.
  """
  files_subset = dict()

  for class_name in classes:
    class_files = files_for_class[class_name]
    files_subset[class_name] = class_files[:files_per_class]

  return files_subset

files_subset = select_subset_of_classes(files_for_class, classes[:NUM_CLASSES], FILES_PER_CLASS)
list(files_subset.keys())

"""Define helper functions that split the videos into training, validation, and test sets. The videos are downloaded from a URL with the zip file, and placed into their respective subdirectiories."""

def download_from_zip(zip_url, to_dir, file_names):
  """ Download the contents of the zip file from the zip URL.

    Args:
      zip_url: A URL with a zip file containing data.
      to_dir: A directory to download data to.
      file_names: Names of files to download.
  """
  with rz.RemoteZip(zip_url) as zip:
    for fn in tqdm.tqdm(file_names):
      class_name = get_class(fn)
      zip.extract(fn, str(to_dir / class_name))
      unzipped_file = to_dir / class_name / fn

      fn = pathlib.Path(fn).parts[-1]
      output_file = to_dir / class_name / fn
      unzipped_file.rename(output_file)

"""The following function returns the remaining data that hasn't already been placed into a subset of data. It allows you to place that remaining data in the next specified subset of data."""

def split_class_lists(files_for_class, count):
  """ Returns the list of files belonging to a subset of data as well as the remainder of
    files that need to be downloaded.

    Args:
      files_for_class: Files belonging to a particular class of data.
      count: Number of files to download.

    Returns:
      Files belonging to the subset of data and dictionary of the remainder of files that need to be downloaded.
  """
  split_files = []
  remainder = {}
  for cls in files_for_class:
    split_files.extend(files_for_class[cls][:count])
    remainder[cls] = files_for_class[cls][count:]
  return split_files, remainder

"""The following `download_ucf_101_subset` function allows you to download a subset of the UCF101 dataset and split it into the training, validation, and test sets. You can specify the number of classes that you would like to use. The `splits` argument allows you to pass in a dictionary in which the key values are the name of subset (example: "train") and the number of videos you would like to have per class."""

def download_ucf_101_subset(zip_url, num_classes, splits, download_dir):
  """ Download a subset of the UCF101 dataset and split them into various parts, such as
    training, validation, and test.

    Args:
      zip_url: A URL with a ZIP file with the data.
      num_classes: Number of labels.
      splits: Dictionary specifying the training, validation, test, etc. (key) division of data
              (value is number of files per split).
      download_dir: Directory to download data to.

    Return:
      Mapping of the directories containing the subsections of data.
  """
  files = list_files_from_zip_url(zip_url)
  for f in files:
    path = os.path.normpath(f)
    tokens = path.split(os.sep)
    if len(tokens) <= 2:
      files.remove(f) # Remove that item from the list if it does not have a filename

  files_for_class = get_files_per_class(files)

  classes = list(files_for_class.keys())[:num_classes]

  for cls in classes:
    random.shuffle(files_for_class[cls])

  # Only use the number of classes you want in the dictionary
  files_for_class = {x: files_for_class[x] for x in classes}

  dirs = {}
  for split_name, split_count in splits.items():
    print(split_name, ":")
    split_dir = download_dir / split_name
    split_files, files_for_class = split_class_lists(files_for_class, split_count)
    download_from_zip(zip_url, split_dir, split_files)
    dirs[split_name] = split_dir

  return dirs

download_dir = pathlib.Path('./UCF101_subset/')
subset_paths = download_ucf_101_subset(URL,
                                       num_classes = NUM_CLASSES,
                                       splits = {"train": 30, "val": 10, "test": 10},
                                       download_dir = download_dir)

"""After downloading the data, you should now have a copy of a subset of the UCF101 dataset. Run the following code to print the total number of videos you have amongst all your subsets of data."""

video_count_train = len(list(download_dir.glob('train/*/*.avi')))
video_count_val = len(list(download_dir.glob('val/*/*.avi')))
video_count_test = len(list(download_dir.glob('test/*/*.avi')))
video_total = video_count_train + video_count_val + video_count_test
print(f"Total videos: {video_total}")

"""You can also preview the directory of data files now."""

!find ./UCF101_subset

"""## Create frames from each video file

The `frames_from_video_file` function splits the videos into frames, reads a randomly chosen span of `n_frames` out of a video file, and returns them as a NumPy `array`.
To reduce memory and computation overhead, choose a **small** number of frames. In addition, pick the **same** number of frames from each video, which makes it easier to work on batches of data.
"""

def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded.
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 15):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  need_length = 1 + (n_frames - 1) * frame_step

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  result.append(format_frames(frame, output_size))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      frame = format_frames(frame, output_size)
      result.append(frame)
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)[..., [2, 1, 0]]

  return result

"""## Visualize video data

The `frames_from_video_file` function that returns a set of frames as a NumPy array. Try using this function on a new video from [Wikimedia](https://commons.wikimedia.org/wiki/Category:Videos_of_sports){:.external} by Patrick Gillett:
"""

!curl -O https://upload.wikimedia.org/wikipedia/commons/8/86/End_of_a_jam.ogv

video_path = "End_of_a_jam.ogv"

sample_video = frames_from_video_file(video_path, n_frames = 10)
sample_video.shape

def to_gif(images):
  converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
  imageio.mimsave('./animation.gif', converted_images, fps=10)
  return embed.embed_file('./animation.gif')

to_gif(sample_video)

"""In addition to examining this video, you can also display the UCF-101 data. To do this, run the following code:"""

# docs-infra: no-execute
ucf_sample_video = frames_from_video_file(next(subset_paths['train'].glob('*/*.avi')), 50)
to_gif(ucf_sample_video)

"""Next, define the `FrameGenerator` class in order to create an iterable object that can feed data into the TensorFlow data pipeline. The generator (`__call__`) function yields the frame array produced by `frames_from_video_file` and a one-hot encoded vector of the label associated with the set of frames."""

class FrameGenerator:
  def __init__(self, path, n_frames, training = False):
    """ Returns a set of frames with their associated label.

      Args:
        path: Video file paths.
        n_frames: Number of frames.
        training: Boolean to determine if training dataset is being created.
    """
    self.path = path
    self.n_frames = n_frames
    self.training = training
    self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
    self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

  def get_files_and_class_names(self):
    video_paths = list(self.path.glob('*/*.avi'))
    classes = [p.parent.name for p in video_paths]
    return video_paths, classes

  def __call__(self):
    video_paths, classes = self.get_files_and_class_names()

    pairs = list(zip(video_paths, classes))

    if self.training:
      random.shuffle(pairs)

    for path, name in pairs:
      video_frames = frames_from_video_file(path, self.n_frames)
      label = self.class_ids_for_name[name] # Encode labels
      yield video_frames, label

"""Test out the `FrameGenerator` object before wrapping it as a TensorFlow Dataset object. Moreover, for the training dataset, ensure you enable training mode so that the data will be shuffled."""

fg = FrameGenerator(subset_paths['train'], 10, training=True)

frames, label = next(fg())

print(f"Shape: {frames.shape}")
print(f"Label: {label}")

"""Finally, create a TensorFlow data input pipeline. This pipeline that you create from the generator object allows you to feed in data to your deep learning model. In this video pipeline, each element is a single set of frames and its associated label."""

# Create the training set
output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))
train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], 10, training=True),
                                          output_signature = output_signature)

"""Check to see that the labels are shuffled."""

for frames, labels in train_ds.take(10):
  print(labels)

# Create the validation set
val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val'], 10),
                                        output_signature = output_signature)

# Print the shapes of the data
train_frames, train_labels = next(iter(train_ds))
print(f'Shape of training set of frames: {train_frames.shape}')
print(f'Shape of training labels: {train_labels.shape}')

val_frames, val_labels = next(iter(val_ds))
print(f'Shape of validation set of frames: {val_frames.shape}')
print(f'Shape of validation labels: {val_labels.shape}')

"""## Configure the dataset for performance

Use buffered prefetching such that you can yield data from the disk without having I/O become blocking. Two important functions to use while loading data are:

* `Dataset.cache`: keeps the sets of frames in memory after they're loaded off the disk during the first epoch. This function ensures that the dataset does not become a bottleneck while training your model. If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache.

* `Dataset.prefetch`: overlaps data preprocessing and model execution while training.
Refer to [Better performance with the `tf.data`](https://www.tensorflow.org/guide/data_performance) for details.
"""

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)

"""To prepare the data to be fed into the model, use batching as shown below. Notice that when working with video data, such as AVI files, the data should be shaped as a five dimensional object. These dimensions are as follows: `[batch_size, number_of_frames, height, width, channels]`. In comparison, an image would have four dimensions: `[batch_size, height, width, channels]`. The image below is an illustration of how the shape of video data is represented.

![Video data shape](https://www.tensorflow.org/images/tutorials/video/video_data_shape.png)

"""

train_ds = train_ds.batch(2)
val_ds = val_ds.batch(2)

train_frames, train_labels = next(iter(train_ds))
print(f'Shape of training set of frames: {train_frames.shape}')
print(f'Shape of training labels: {train_labels.shape}')

val_frames, val_labels = next(iter(val_ds))
print(f'Shape of validation set of frames: {val_frames.shape}')
print(f'Shape of validation labels: {val_labels.shape}')

"""## Next steps

Now that you have created a TensorFlow `Dataset` of video frames with their labels, you can use it with a deep learning model. The following classification model that uses a pre-trained [EfficientNet](https://arxiv.org/abs/1905.11946){:.external} trains to high accuracy in a few minutes:
"""

net = tf.keras.applications.EfficientNetB0(include_top = False)
net.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(scale=255),
    tf.keras.layers.TimeDistributed(net),
    tf.keras.layers.Dense(10),
    tf.keras.layers.GlobalAveragePooling3D()
])

model.compile(optimizer = 'adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics=['accuracy'])

model.fit(train_ds,
          epochs = 10,
          validation_data = val_ds,
          callbacks = tf.keras.callbacks.EarlyStopping(patience = 2, monitor = 'val_loss'))

"""To learn more about working with video data in TensorFlow, check out the following tutorials:

* [Build a 3D CNN model for video classification](https://www.tensorflow.org/tutorials/video/video_classification)
* [MoViNet for streaming action recognition](https://www.tensorflow.org/hub/tutorials/movinet)
* [Transfer learning for video classification with MoViNet](https://www.tensorflow.org/tutorials/video/transfer_learning_with_movinet)
"""