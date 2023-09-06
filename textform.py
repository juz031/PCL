import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import numpy as np
import faiss
from PIL import Image
from scipy.stats import pearsonr
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import pcl.loader
import pcl.builder

parser = argparse.ArgumentParser(description='Data sampling')
parser.add_argument('checkpoint', metavar='DIR',
                    help='path to checkpoint')

arch = 'resnet50'
low_dim = 128
pcl_r = 256
moco_m = 0.999
temperature = 0.2
mlp = True

def main():
    args = parser.parse_args()

    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    model = pcl.builder.MoCo(
            models.__dict__[arch],
            low_dim, pcl_r, moco_m, temperature, mlp)

    checkpoint_path = args.checkpoint
    print(checkpoint_path)
    # checkpoint_path = '/home/junruz/PCL/experiment_pcl/checkpoint_0199.pth.tar'
    # checkpoint_path = '/home/junruz/PCL/experiment_pcl_shape/checkpoint_0199.pth.tar'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict_unwrapped'])

    model.eval()

    transform_texform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # This line ensures all images have 3 channels
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Define the main folders
    main_folders = ['/user_data/junruz/Texform_eval_hard_original', '/user_data/junruz/Texform_eval_hard_texform1', '/user_data/junruz/Texform_eval_hard_texform2']
    # main_folders = ['/user_data/ziqiwen/imagenette2-160/val/', '/user_data/ziqiwen/imagenette2-160_textureform_seed_pool4/val', '/user_data/ziqiwen/imagenette2-160_textureform_seed1_pool4/val']
    # main_folders = ['/user_data/ziqiwen/imagenette2-160/val/', '/user_data/ziqiwen/imagenette2-160_textureform/val', '/user_data/ziqiwen/imagenette2-160_textureform_seed/val']
    # main_folders = ['/user_data/ziqiwen/imagenette2-160/val/', '/user_data/ziqiwen/imagenette2-160_shuffle/val/', '/user_data/ziqiwen/imagenette2-160_shuffle2/val/']

    # Define the image extensions
    image_extensions = ['.JPEG', '.jpg', '.jpg']

    # Get the sub-folders from the first main folder
    # sub_folders = os.listdir(main_folders[0])

    correctCount = 0
    total = 0
    fir=0

    failed_images=[]

    image_files = os.listdir(main_folders[1])
    for image_file in image_files:
        features = []
        original_image_name=""
        for main_folder, image_extension in zip(main_folders, image_extensions):
            # Construct the image path
            image_path = os.path.join(main_folder, image_file.replace('.jpg', image_extension))
            if image_extension == ".JPEG":
                original_image_name=image_path

            image = transform_texform(Image.open(image_path)).unsqueeze(0)

            # Extract the features from the last convolutional layer
            with torch.no_grad():
                feature = model(image, is_eval=True).cpu().numpy().flatten()

            mean = feature.mean()
            std = feature.std()
            # Normalize the tensor
            feature_normalized = (feature - mean) / std

            # Append the feature vector to the list
            features.append(feature_normalized)

        # Calculate the Pearson distances
        pearson_distance_12 = 1 - pearsonr(features[0], features[1])[0]
        pearson_distance_13 = 1 - pearsonr(features[0], features[2])[0]
        pearson_distance_23 = 1 - pearsonr(features[1], features[2])[0]

        # Calculate the dissimilarities
        dissimilarity1 = (pearson_distance_12 + pearson_distance_13) / 2
        dissimilarity2 = (pearson_distance_12 + pearson_distance_23) / 2
        dissimilarity3 = (pearson_distance_13 + pearson_distance_23) / 2

        # Pass the dissimilarities into a softmax function
        probabilities = softmax([dissimilarity1, dissimilarity2, dissimilarity3])
        if fir==0:
            print(probabilities)
            fir=1
        prediction = np.argmax(probabilities)
        if prediction==0:
            correctCount+=1
        else:
            failed_images.append(original_image_name)
        total+=1

        # print(f'Prediction for {image_file}:', prediction)

    # with open("failed_images.txt", "w") as f:
    #     for image in failed_images:
    #         f.write(str(image) + "\n")  # add a newline character for each new item

    acc_texform=correctCount/total*1.0
    print(acc_texform)

if __name__ == '__main__':
    main()