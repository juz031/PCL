import torch
import pcl.loader
import torchvision.transforms as transforms
import numpy as np
import cv2

evaldir = '/user_data/junruz/imagenette_masks/train'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

eval_augmentation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
        ])

eval_dataset = pcl.loader.ImageFolderInstance(
        evaldir,
        eval_augmentation)

images = eval_dataset.imgs

cluster_path = '/home/junruz/PCL/experiment_pcl_shape/clusters_199'
cluster = torch.load(cluster_path,map_location=torch.device('cpu'))
assignments = [a.numpy() for a in cluster['im2cluster']]
centroids = [c.numpy() for c in cluster['centroids']]

feature_path = '/home/junruz/PCL/experiment_pcl_shape/features_199'
features = torch.load(feature_path)
features = features.cpu().detach().numpy()

k = 3
top_k = {'imgs':[], 'distances':[]}
for cluster_id in range(20):
        points = np.argwhere(assignments[0] == cluster_id)
        print(points.shape[0])
        distances = []
        for point in points:
                distances.append(np.linalg.norm(features[point].squeeze()-centroids[0][cluster_id]))
        idx = np.argpartition(distances, k)[:k]
        top_k['distances'].append(np.array(distances)[idx])
        top_k['imgs'].append(points[idx])

for cluster_id in range(20):
        for img_id in range(3):
                img = cv2.imread(images[top_k['imgs'][cluster_id][img_id][0]][0])
                class_id = images[top_k['imgs'][cluster_id][img_id][0]][1]
                img_name = '/home/junruz/PCL/experiment_pcl_shape/top_3/{}_{}_{}.png'.format(cluster_id, img_id, class_id)
                cv2.imwrite(img_name, img)