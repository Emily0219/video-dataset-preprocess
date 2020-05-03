import os
import glob
import numpy as np
import sys


# 生成HMDB51训练集与测试集的标注文件，包含视频名与分类标签，标签从0开始
def traintest_annotion(root_dir):
    split_data={}
    for file in sorted(glob.glob(root_dir+'*')):
        with open(file) as f:
            for line in f:
                split_data[line.split(' ')[0].split('.')[0]] = int(line.split(' ')[1])
    return split_data


def split_traintest(split_data, frames_dir, outdir):
    train_labels = []
    test_labels = []
    val_labels = []
    clasess = []
    labels = {}

    for foldername in sorted(os.listdir(frames_dir)):
        clasess.append(foldername)

    # for foldername in sorted(os.listdir(location)):
    #     clasess.append([foldername, str(i)])
    #     i+=1
    # print(clasess)
    for i in range(len(clasess)):
        labels[clasess[i]] = i

    # f = open('mylabel.txt', 'w')
    # for item in clasess:
    #     item=' '.join(item)
    #     f.write(item)
    #     f.write('\n')
    # f.close()

    # import pandas as pd
    # landmarks_frame = pd.read_csv('hmdb_labels.txt', delimiter=' ', header=None)
    # for idx in range(51):
    #     video_label = landmarks_frame.iloc[idx, 1]

    for video in clasess:
        for videoname in sorted(glob.glob(frames_dir+video+'/*')):
            video_name = videoname.split('/')[-2]+'/'+videoname.split('/')[-1]
            print(videoname.split('/')[-2]+'/'+videoname.split('/')[-1])
            print('='*50)
            if split_data[videoname.split('/')[-1]] == 1:  # 1 - train, 2 - test, 0 - not train or test
                train_labels.append([video_name, str(labels[video])])
            elif split_data[videoname.split('/')[-1]] == 2:
                test_labels.append([video_name, str(labels[video])])
            elif split_data[videoname.split('/')[-1]] == 0:
                val_labels.append([video_name, str(labels[video])])

    f = open(outdir+'hmdb51_train.txt', 'w')
    for item in train_labels:
        item = ' '.join(item)
        f.write(item)
        f.write('\n')
    f.close()

    f = open(outdir+'hmdb51_test.txt', 'w')
    for item in test_labels:
        item = ' '.join(item)
        f.write(item)
        f.write('\n')
    f.close()

    f = open(outdir+'hmdb51_val.txt', 'w')
    for item in val_labels:
        item = ' '.join(item)
        f.write(item)
        f.write('\n')
    f.close()


if __name__ == '__main__':
    root_dir = sys.argv[1]
    frames_dir = sys.argv[2]
    out_dir = sys.argv[3]

    # frames_dir = '/home/ran/mnt1/Dataset/hmdb51_n_frames/'
    # root_dir = '/home/ran/mnt1/Dataset/hmdb51_TrainTestlist/'

    split_data = traintest_annotion(root_dir)
    split_traintest(split_data, frames_dir, out_dir)