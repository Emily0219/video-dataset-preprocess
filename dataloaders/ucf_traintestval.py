import os
import numpy as np
from sklearn.model_selection import train_test_split

# Directory containing the UCF101 dataset
dataset_dir = './Dataset/UCF-101'

# Get a list of all video files
video_files = [os.path.join(dirpath, filename)
               for dirpath, dirnames, filenames in os.walk(dataset_dir)
               for filename in filenames if filename.endswith('.avi')]

# Convert the list to a numpy array and sort it
video_files = np.array(sorted(video_files))

# Extract action folders from video file paths
action_folders = [os.path.basename(os.path.dirname(file)) for file in video_files]

# Directory to store the text files
output_dir = './Dataset/UCF-TrainTestVal'

# Get a list of all action folder names
all_action_folders = sorted([f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))])
# Open the output file
with open(os.path.join(output_dir, 'class.txt'), 'w') as f:
    # Write the action folder names to the file
    for i, folder in enumerate(all_action_folders, 1):
        f.write(f'{i} {folder}\n')

# Create a list of numbered action folders
numbered_action_folders = [f"{i+1} {folder}" for i, folder in enumerate(action_folders)]

# # Save the numbered action folders to a single text file
# all_action_folders_path = os.path.join(output_dir, 'all_action_folders.txt')
# np.savetxt(all_action_folders_path, numbered_action_folders, fmt='%s')

# Create a list of video file names with their respective folders
video_files_with_folders = [f"{folder}/{os.path.basename(file)}" for folder, file in zip(action_folders, video_files)]

# Save the video file names with folders to a single text file
all_video_files_path = os.path.join(output_dir, 'all_video_files.txt')
np.savetxt(all_video_files_path, video_files_with_folders, fmt='%s')

# Split the data into training and test sets
train_files, test_files = train_test_split(video_files_with_folders, test_size=0.3, random_state=42)

# Further split the test data into validation and test sets
val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)

# Save the numbered video file names with folders to individual text files
np.savetxt(os.path.join(output_dir, 'train_video_files.txt'), train_files, fmt='%s')
np.savetxt(os.path.join(output_dir, 'val_video_files.txt'), val_files, fmt='%s')
np.savetxt(os.path.join(output_dir, 'test_video_files.txt'), test_files, fmt='%s')

# Calculate the percentages
total_files = len(video_files_with_folders)
test_percentage = len(test_files) / total_files * 100
val_percentage = len(val_files) / total_files * 100
train_percentage = len(train_files) / total_files * 100

# Print the percentages
print(f"Test set percentage: {test_percentage}%")
print(f"Validation set percentage: {val_percentage}%")
print(f"Training set percentage: {train_percentage}%")




