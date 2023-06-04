import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from sklearn import metrics
import time
from sklearn import preprocessing
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn import manifold
from skimage.segmentation import slic,mark_boundaries
import cv2
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

k=50

def accuracy_eval1(label_tr, label_pred):
    overall_accuracy = metrics.accuracy_score(label_tr, label_pred)
    avarage_accuracy = np.mean(metrics.precision_score(label_tr, label_pred, average = None))
    kappa = metrics.cohen_kappa_score(label_tr, label_pred)
    cm = metrics.confusion_matrix(label_tr, label_pred)
    return overall_accuracy, avarage_accuracy, kappa,cm


def LSC_superpixel(I, nseg):
    superpixelNum = nseg
    ratio = 0.075
    size = int(math.sqrt(((I.shape[0] * I.shape[1]) / nseg)))
    superpixelLSC = cv2.ximgproc.createSuperpixelLSC(
        I,
        region_size=size,
        ratio=0.005)
    superpixelLSC.iterate()
    superpixelLSC.enforceLabelConnectivity(min_element_size=25)
    segments = superpixelLSC.getLabels()
    return np.array(segments, np.int64)


def SEEDS_superpixel(I, nseg):
    I = np.array(I[:, :, 0:3], np.float32).copy()
    I_new = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    # I_new =np.array( I[:,:,0:3],np.float32).copy()
    height, width, channels = I_new.shape

    superpixelNum = nseg
    seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, int(superpixelNum), num_levels=2, prior=1,
                                               histogram_bins=5)
    seeds.iterate(I_new, 4)
    segments = seeds.getLabels()
  
    return segments


def SegmentsLabelProcess(labels):
    
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))

    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i

    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels

class SLIC(object):
    def __init__(self, HSI, labels, n_segments=1000, compactness=20, max_iter=20, sigma=0, min_size_factor=0.3,
                 max_size_factor=2):
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_iter = max_iter
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma

        height, width, bands = HSI.shape  
        data = np.reshape(HSI, [height * width, bands])
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        self.data = np.reshape(data, [height, width, bands])
        self.labels = labels

    



def Eu_dis(x):

    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    dist_mat = aa + aa.T - (2 * x * x.T)
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    print(dist_mat.shape)
    
    return dist_mat

def feature_concat(*F_list, normal_col=False):
   
    features = None
    for f in F_list:
        if f is not None and type(f) != list:
            # deal with the dimension that more than two #f != []:
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    H = None
    for h in H_list:
        if h is not None and type(h) != list:  
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H










device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
samples_type = ['ratio', 'same_num'][0]

def DrawResult(labels, imageID):
    # ID=1:Pavia University
    # ID=2:Indian Pines
    # ID=6:KSC
    num_class = int(labels.max() + 1)
    if imageID == 2:
        row = 610
        col = 340
        palette = np.array([[0, 0, 0],
                            [0, 255, 0],
                            [0, 255, 255],
                            [45, 138, 86],
                            [255, 0, 255],
                            [255, 165, 0],
                            [159, 31, 239],
                            [255, 0, 0],
                            [255, 255, 0],
                            [255, 255, 255]])
        palette = palette * 1.0 / 255
    elif imageID == 1:
        row = 145
        col = 145
        palette = np.array([[0, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 48, 96],
                            [46, 139, 87],
                            [160, 32, 240],
                            [255, 127, 80],
                            [127, 255, 212],
                            [218, 112, 214],
                            [160, 82, 45],
                            [127, 255, 0],
                            [216, 191, 216],
                            [238, 0, 0],
                            [255, 255, 255]])
        palette = palette * 1.0 / 255
    elif imageID == 3:
        row = 512
        col = 217
        palette = np.array([[0, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 48, 96],
                            [46, 139, 87],
                            [160, 32, 240],
                            [255, 127, 80],
                            [127, 255, 212],
                            [218, 112, 214],
                            [160, 82, 45],
                            [127, 255, 0],
                            [216, 191, 216],
                            [238, 0, 0],
                            [255, 255, 255]])
        palette = palette * 1.0 / 255
    elif imageID == 4:
        row = 512
        col = 614
        palette = np.array([[0, 0, 0],
                            [255, 0, 255],
                            [217, 115, 0],
                            [179, 30, 0],
                            [0, 52, 0],
                            [72, 0, 0],
                            [0, 255, 255],
                            [145, 132, 135],
                            [255, 255, 172],
                            [255, 197, 80],
                            [60, 201, 255],
                            [11, 63, 124],
                            [0, 0, 255],
                            [255, 255, 255]])
        palette = palette * 1.0 / 255
    elif imageID == 5:
        row = 1476
        col = 256
        palette = np.array([[0, 0, 0],
                            [255, 0, 255],
                            [217, 115, 0],
                            [179, 30, 0],
                            [0, 52, 0],
                            [72, 0, 0],
                            [0, 255, 255],
                            [145, 132, 135],
                            [255, 255, 172],
                            [255, 197, 80],
                            [60, 201, 255],
                            [11, 63, 124],
                            [0, 0, 255],
                            [127, 255, 0],
                            [255,255,255]])
        palette = palette * 1.0 / 255
    elif imageID == 6:
        row = 349
        col = 1905
        palette = np.array([[0, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 48, 96],
                            [46, 139, 87],
                            [160, 32, 240],
                            [255, 127, 80],
                            [127, 255, 212],
                            [218, 112, 214],
                            [160, 82, 45],
                            [127, 255, 0],
                            [216, 191, 216],
                            [238, 0, 0]])
        palette = palette * 1.0 / 255
    X_result = np.zeros((labels.shape[0], 3))
    for i in range(0, num_class):
        X_result[np.where(labels == i), 0] = palette[i, 0]
        X_result[np.where(labels == i), 1] = palette[i, 1]
        X_result[np.where(labels == i), 2] = palette[i, 2]

    X_result = np.reshape(X_result, (row, col, 3))
    plt.axis("off")
    plt.imshow(X_result)
    return X_result

for (FLAG, curr_train_ratio, Scale) in [(1, 0.05,150)]:
 
    Seed_List = [0]
    samples_type = 'ratio'
    if FLAG == 1:
        data_mat = sio.loadmat('/DATA/HyperImage_data/Indian_pines_corrected.mat')
        data = data_mat['indian_pines_corrected']
        gt_mat = sio.loadmat('/DATA/HyperImage_data/Indian_pines_gt.mat')
        gt = gt_mat['indian_pines_gt']
     
        val_ratio = 0.01  
        class_count = 16  
        learning_rate = 5e-4  
        max_epoch = 600 
        dataset_name = "_IP"  
        # superpixel_scale=100
        pass
    if FLAG == 2:
        data_mat = sio.loadmat('/DATA/HyperImage_data/PaviaU.mat')
        data = data_mat['paviaU']
        gt_mat = sio.loadmat('/DATA/HyperImage_data/PaviaU_gt.mat')
        gt = gt_mat['paviaU_gt']

        # 参数预设
        # train_ratio = 0.01  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.01  
        class_count = 9 
        learning_rate = 5e-3 
        max_epoch = 600 
        dataset_name = "_UP"  
        # superpixel_scale = 100
        pass
    if FLAG == 3:
        data_mat = sio.loadmat('/DATA/HyperImage_data/Salinas_corrected.mat')
        data = data_mat['salinas_corrected']
        gt_mat = sio.loadmat('/DATA/HyperImage_data/Salinas_gt.mat')
        gt = gt_mat['salinas_gt']

        # 参数预设
        # train_ratio = 0.01  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.01 
        class_count = 16 
        learning_rate = 5e-4 
        max_epoch = 600  
        dataset_name = "_SA" 
        # superpixel_scale = 100
        pass
    if FLAG == 4:
        data_mat = sio.loadmat('/DATA/HyperImage_data/KSC.mat')
        data = data_mat['KSC']
        gt_mat = sio.loadmat('/DATA/HyperImage_data/KSC_gt.mat')
        gt = gt_mat['KSC_gt']

        # 参数预设
        # train_ratio = 0.05  
        val_ratio = 0.01 
        class_count = 13  
        learning_rate = 5e-4  
        max_epoch = 600 
        dataset_name = "_KSC" 
        # superpixel_scale = 200
        pass
    if FLAG == 5:
        data_mat = sio.loadmat('/DATA/HyperImage_data/Botswana.mat')
        data = data_mat['Botswana']
        gt_mat = sio.loadmat('/DATA/HyperImage_data/Botswana_gt.mat')
        gt = gt_mat['Botswana_gt']

        # 参数预设
        # train_ratio = 0.05  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.01  
        class_count = 14  
        learning_rate = 5e-4  
        max_epoch = 600  
        dataset_name = "_BS"  
    if FLAG == 6:
        data_mat = sio.loadmat('/DATA/HyperImage_data/contest2013.mat')
        data = data_mat['contest2013']
        gt_mat = sio.loadmat('/DATA/HyperImage_data/contest2013_gt.mat')
        gt = gt_mat['contest2013_gt']

        # 参数预设
        # train_ratio = 0.05  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.01  
        class_count = 15 
        learning_rate = 5e-4
        max_epoch = 600 
        dataset_name = "_HS" 
        # superpixel_scale = 200
        pass
    ###########
   

