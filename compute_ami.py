import torch
import os
import numpy as np
from sklearn.metrics.cluster import adjusted_mutual_info_score as ami
import matplotlib.pyplot as plt


data_dir = "/user_data/junruz/imagenette2/train"
dataset = os.walk(data_dir)

labels = np.zeros(9469)
for _, dir_list, __ in dataset:
    idx = 0
    value = 0
    for dir_name in dir_list:
        img_dir = os.path.join(data_dir, dir_name)
        imgs = os.walk(img_dir)
        for ___, ____, img_list in imgs:
            cnt = len(img_list)
            labels[idx:idx + cnt] = value
        idx += cnt
        value += 1

# cluster_dir = '/home/junruz/PCL/experiment_pcl_200/clusters'
# all_clusters = os.listdir(cluster_dir)
# all_clusters_sorted = sorted([item for item in all_clusters if os.path.isfile(os.path.join(cluster_dir, item))])

# ami_scores = {'400':[0], '500':[0], '600':[0]}

# for file_name in all_clusters_sorted:
#     print(file_name)
#     clusters = torch.load(os.path.join(cluster_dir, file_name))
#     assignments_400 = clusters['im2cluster'][0].cpu().detach().numpy()
#     assignments_500 = clusters['im2cluster'][1].cpu().detach().numpy()
#     assignments_600 = clusters['im2cluster'][2].cpu().detach().numpy()
#     ami_scores['400'].append(ami(assignments_400, labels))
#     ami_scores['500'].append(ami(assignments_500, labels))
#     ami_scores['600'].append(ami(assignments_600, labels))

# epochs = np.linspace(20, 200, 10)

# plt.plot(epochs, ami_scores['400'], label='num_clusters=200')
# plt.plot(epochs, ami_scores['500'], label='num_clusters=250')
# plt.plot(epochs, ami_scores['600'], label='num_clusters=300')

# plt.legend()

# plt.savefig('/home/junruz/PCL/figures/pcl_200_ami_scores.png')


file_path0 = '/home/junruz/PCL/experiment_pcl_16_shape/clusters_29'
# file_path1 = r"D:\Download\ShapeProto_clusters_199"
a0 = torch.load(file_path0)
# a1 = torch.load(file_path1)
assignments_400_0 = a0['im2cluster'][0].cpu().detach().numpy()
# assignments_400_1 = a1['im2cluster'][0].cpu().detach().numpy()

print(ami(assignments_400_0, labels))
# print(ami(assignments_400_1, labels))