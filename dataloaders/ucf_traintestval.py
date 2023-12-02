import os
import numpy as np
from sklearn.model_selection import train_test_split

# Directory containing the UCF101 dataset
dataset_dir = './Dataset/UCF-101'

# Get a list of all video files
video_files = [os.path.join(os.path.basename(dirpath), filename)
               for dirpath, dirnames, filenames in os.walk(dataset_dir)
               for filename in filenames if filename.endswith('.avi')]

# Convert the list to a numpy array and sort it
video_files = np.array(sorted(video_files))

# Split the data into training and test sets
train_files, test_files = train_test_split(video_files, test_size=0.3, random_state=42)

# Further split the test data into validation and test sets
val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)

# Directory to store the text files
output_dir = './Dataset/UCF-TrainTestVal'

# Save the file paths to text files
np.savetxt(os.path.join(output_dir, 'train_files.txt'), train_files, fmt='%s')
np.savetxt(os.path.join(output_dir, 'val_files.txt'), val_files, fmt='%s')
np.savetxt(os.path.join(output_dir, 'test_files.txt'), test_files, fmt='%s')