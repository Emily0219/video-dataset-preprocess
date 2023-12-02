# import os

# # Directory containing the UCF-101 dataset (1fps images)
# dataset_dir = './Dataset/UCF101_n_frames'

# # Directory to save the YOLO-formatted annotations
# output_dir = './Dataset/UCF101_yolo'

# # List of class names
# classes = [
#     'ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching',
#     'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', 'Biking', 'Billiards', 'BlowDryHair',
#     'BlowingCandles', 'BodyWeightSquats', 'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke',
#     'BrushingTeeth', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen',
#     'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', 'FrisbeeCatch',
#     'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering', 'HammerThrow', 'HandstandPushups',
#     'HandstandWalking', 'HeadMassage', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing',
#     'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Knitting', 'LongJump',
#     'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', 'Nunchucks', 'ParallelBars', 'PizzaTossing',
#     'PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano',
#     'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch',
#     'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard',
#     'Shotput', 'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty',
#     'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', 'TaiChi', 'TennisSwing',
#     'ThrowDiscus', 'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog',
#     'WallPushups', 'WritingOnBoard', 'YoYo'
# ]

# # Create the output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# # Loop over all images in the dataset
# for filename in os.listdir(dataset_dir):
#     if filename.endswith('.jpg'):
#         # Load the corresponding annotation
#         annotation_file_path = os.path.join(dataset_dir, filename[:-4] + '.txt')
        
#         if os.path.exists(annotation_file_path):
#             with open(annotation_file_path, 'r') as f:
#                 annotations = f.readlines()

#             # Convert the annotations to YOLO format
#             yolo_annotations = []
#             for annotation in annotations:
#                 class_id, x, y, width, height = map(float, annotation.strip().split())
#                 class_id = int(class_id)  # Convert class name to index

#                 # YOLO format: class x_center y_center width height
#                 x_center = x + width / 2
#                 y_center = y + height / 2

#                 yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

#             # Save the YOLO-formatted annotations
#             output_annotation_file_path = os.path.join(output_dir, filename[:-4] + '.txt')
#             with open(output_annotation_file_path, 'w') as f:
#                 f.write('\n'.join(yolo_annotations))







# import os

# # Directory containing the UCF-101 dataset (1fps images)
# dataset_dir = './Dataset/UCF101_n_frames/'

# # Directory to save the YOLO-formatted annotations
# output_dir = './Dataset/UCF101_yolo'

# # Create the output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# # Loop over all images in the dataset
# for filename in os.listdir(dataset_dir):
#     if filename.endswith('.jpg'):
#         # Load the corresponding annotation
#         annotation_file_path = os.path.join(dataset_dir, filename[:-4] + '.txt')
        
#         if os.path.exists(annotation_file_path):
#             with open(annotation_file_path, 'r') as f:
#                 annotations = f.readlines()

#             # Convert the annotations to YOLO format
#             yolo_annotations = []
#             for annotation in annotations:
#                 try:
#                     class_id, x, y, width, height = map(float, annotation.strip().split())
#                     class_id = int(class_id)  # Convert class name to index

#                     # YOLO format: class x_center y_center width height
#                     x_center = x + width / 2
#                     y_center = y + height / 2

#                     yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
#                 except Exception as e:
#                     print(f"Error processing annotation '{annotation}': {e}")

#             # Save the YOLO-formatted annotations
#             output_annotation_file_path = os.path.join(output_dir, filename[:-4] + '.txt')
#             with open(output_annotation_file_path, 'w') as f:
#                 f.write('\n'.join(yolo_annotations))
#         else:
#             print(f"Annotation file does not exist: {annotation_file_path}")
#     else:
#         print(f"Not a .jpg file: {filename}")
















import os

# Directory containing the UCF-101 dataset (1fps images)
dataset_dir = './Dataset/UCF101_n_frames'

# Directory to save the YOLO-formatted annotations
output_dir = './Dataset/UCF101_yolo'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop over all subdirectories in the dataset
for dirpath, dirnames, filenames in os.walk(dataset_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            # Load the corresponding annotation
            annotation_file_path = os.path.join(dirpath, filename[:-4] + '.txt')
            
            if os.path.exists(annotation_file_path):
                with open(annotation_file_path, 'r') as f:
                    annotations = f.readlines()

                # Convert the annotations to YOLO format
                yolo_annotations = []
                for annotation in annotations:
                    try:
                        class_id, x, y, width, height = map(float, annotation.strip().split())
                        class_id = int(class_id)  # Convert class name to index

                        # YOLO format: class x_center y_center width height
                        x_center = x + width / 2
                        y_center = y + height / 2

                        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
                    except Exception as e:
                        print(f"Error processing annotation '{annotation}': {e}")

                # Save the YOLO-formatted annotations
                output_annotation_file_path = os.path.join(output_dir, dirpath[len(dataset_dir):], filename[:-4] + '.txt')
                os.makedirs(os.path.dirname(output_annotation_file_path), exist_ok=True)
                with open(output_annotation_file_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
            else:
                print(f"Annotation file does not exist: {annotation_file_path}")
        else:
            print(f"Not a .jpg file: {filename}")
