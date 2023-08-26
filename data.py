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

    def get_Q_and_S_and_Segments(self):
      
        img = self.data
        (h, w, d) = img.shape
        img = np.double(img)
        
        segments = slic(img, n_segments=self.n_segments, compactness=self.compactness, max_iter=self.max_iter,
                        convert2lab=False, sigma=self.sigma, enforce_connectivity=True,
                        min_size_factor=self.min_size_factor, max_size_factor=self.max_size_factor, slic_zero=False)
       
        if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))): segments = SegmentsLabelProcess(
            segments)
        self.segments = segments
        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count
        print("superpixel_count", superpixel_count)
       
        out = mark_boundaries(img[:, :, [0, 1, 2]], segments)
        plt.figure()
        plt.imshow(out)
        #plt.show()
        # out = (img[:, :, [0, 1, 2]]-np.min(img[:, :, [0, 1, 2]]))/(np.max(img[:, :, [0, 1, 2]])-np.min(img[:, :, [0, 1, 2]]))
        segments = np.reshape(segments, [-1])
        S = np.zeros([superpixel_count, d], dtype=np.float32)
        Q = np.zeros([w * h, superpixel_count], dtype=np.float32)
        x = np.reshape(img, [-1, d])
        for i in range(superpixel_count):
            idx = np.where(segments == i)[0]
            count = len(idx)
            pixels = x[idx]
            superpixel = np.sum(pixels, 0) / count
            S[i] = superpixel
            Q[idx, i] = 1
        self.S = S
        self.Q = Q

        return Q, S, self.segments
        
class LDA_SLIC(object):
    def __init__(self, data, labels, n_component):
        self.data = data
        self.init_labels = labels
        self.curr_data = data
        self.n_component = n_component
        self.height, self.width, self.bands = data.shape
        self.x_flatt = np.reshape(data, [self.width * self.height, self.bands])
        self.y_flatt = np.reshape(labels, [self.height * self.width])
        self.labes = labels

    def SLIC_Process(self, img, scale=25):
        n_segments_init = self.height * self.width / scale
        print("n_segments_init", n_segments_init)
        myslic = SLIC(img, n_segments=n_segments_init, labels=self.labes, compactness=1, sigma=1, min_size_factor=0.1,
                      max_size_factor=2)
        Q, S, Segments = myslic.get_Q_and_S_and_Segments()

        return Q, S, Segments

    def simple_superpixel_PCA(self, scale):
        pca = PCA(n_components=k)
        X=np.reshape(self.data, [self.height*self.width,self.bands])
        X = pca.fit(X).transform(X)
        X=np.reshape(X,[self.height, self.width, -1])
        Q, S, Seg = self.SLIC_Process(X, scale=scale)
        return Q, S, Seg
    
    def simple_superpixel_LDA(self, scale):
        LDA=LinearDiscriminantAnalysis(n_components=16)
        X=np.reshape(self.data, [self.height*self.width,self.bands])
        X = LDA.fit(X,self.y_flatt).transform(X)
        X=np.reshape(X,[self.height, self.width, -1])
        Q, S, Seg = self.SLIC_Process(X, scale=scale)
        return Q, S, Seg
    
    def simple_superpixel_tSNE(self, scale):
        tsne = manifold.TSNE(n_components=15, init='pca', random_state=0)
        X=np.reshape(self.data, [self.height*self.width,self.bands])
        X = tsne.fit_transform(X)
        X=np.reshape(X,[self.height, self.width, -1])
        Q, S, Seg = self.SLIC_Process(X, scale=scale)
        return Q, S, Seg
        
    def simple_superpixel_no_LDA(self, scale):
        Q, S, Seg = self.SLIC_Process(self.data, scale=scale)
        return Q, S, Seg

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
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    """
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

def generate_G_from_H(H, variable_weight=False):
   
    if type(H) != list:
        
        return _generate_G_from_H(H)
        
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H))
        return G
def _generate_G_from_H(H):  
    H = np.array(H)
    n_edge = H.shape[1]
    W = np.ones(n_edge)
    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)
    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    DV1 = np.mat(np.diag(np.power(DV, -1)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T
    G= DV1 * H * W * invDE * HT 
    return G

def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=False, m_prob=1):
 
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        dis_vec=dis_vec.T
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H

def construct_H_with_KNN(X, K_neigs, split_diff_scale=False, is_probH=True, m_prob=1):

    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    dis_mat = Eu_dis(X)

    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H
    
def load_feature_construct_H(cnn_ft,
        m_prob=1,
        K_neigs=[8],
        is_probH=True,
        split_diff_scale=False, ):
    mvcnn_ft = cnn_ft
    fts = None
    fts = feature_concat(fts, mvcnn_ft)
    H = None
    tmp = construct_H_with_KNN(fts, K_neigs=K_neigs,
                               split_diff_scale=split_diff_scale,
                               is_probH=is_probH, m_prob=m_prob)
    H = hyperedge_concat(H, tmp)
    return fts, H


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)

        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, Q, dropout):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.n_class = n_class
        self.Q = Q
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)
        self.linear = nn.Linear(in_ch, n_class)
    def forward(self, x, G):
        x = self.hgc1(x, G)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
        x = torch.matmul(self.Q, x)
        x = self.linear(x)
        
        return x

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
    superpixel_scale = Scale  #########################
    m,n,d=data.shape
    orig_data = data
    height, width, bands = data.shape  # 原始高光谱数据的三个维度
    print('data',data.shape)
    data = np.reshape(data, [height * width, bands])
    
    
    
    
    data = np.reshape(data, [height, width, bands])
    data1 = np.reshape(data, [height * width, bands])
    gt_reshape = np.reshape(gt, [-1])
    
    train_rand_idx = []



    if samples_type == 'ratio':
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            rand_list = [i for i in range(samplesCount)]
            train_sum = np.ceil(samplesCount * curr_train_ratio).astype('int32')
            if (train_sum >50):
                train_sum =50
                rand_idx = random.sample(rand_list, train_sum)
            #else:
            rand_idx = random.sample(rand_list, train_sum)
            rand_real_idx_per_class = idx[rand_idx]
            train_num = len(rand_real_idx_per_class)
            print('class:%d  sample_sum:%d  train_num:%d  test_num:%d' % (
                i + 1, samplesCount, train_num, samplesCount - train_num))
            train_rand_idx.append(rand_real_idx_per_class)
        train_rand_idx = np.array(train_rand_idx)
        train_data_index = []
        for c in range(train_rand_idx.shape[0]):
            a = train_rand_idx[c]
            for j in range(a.shape[0]):
                train_data_index.append(a[j])
        train_data_index = np.array(train_data_index)

        train_data_index = set(train_data_index)
        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)

        background_idx = np.where(gt_reshape == 0)[-1]
        background_idx = set(background_idx)
        test_data_index = all_data_index - train_data_index - background_idx
        data_idx=all_data_index - background_idx
        background_idx = list(background_idx)
        data_idx = list(data_idx)
        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)

    train_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(train_data_index)):
        train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
        pass

 
    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_data_index)):
        test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
        pass
    train_samples_gt = np.reshape(train_samples_gt, [height, width])
    test_samples_gt = np.reshape(test_samples_gt, [height, width])

    ls = LDA_SLIC(data, np.reshape(train_samples_gt, [height, width]), class_count-1)
    tic0 = time.time()
    Q, S, Seg = ls.simple_superpixel_LDA(scale=superpixel_scale)
    Q = torch.from_numpy(Q).to(device)
    idx_test = test_data_index
    idx_test = np.array(idx_test)
    idx_test = torch.from_numpy(idx_test).type(torch.long)
    idx_train = train_data_index
    idx_train = np.array(idx_train)
    idx_train = torch.from_numpy(idx_train).type(torch.long)
    background_idx = np.array(background_idx)
    background_idx = torch.from_numpy(background_idx).type(torch.long)
    data_idx = np.array(data_idx)
    data_idx = torch.from_numpy(data_idx).type(torch.long)
    train_num_all=len(idx_train)



    fts, H = load_feature_construct_H(S,m_prob=1, K_neigs=[8], is_probH=True, split_diff_scale=False)
    G = generate_G_from_H(H)
    G = torch.Tensor(G).to(device)
    fts = torch.Tensor(fts).to(device)
    labels = torch.from_numpy(gt_reshape.astype('int32'))
    labels=labels.long()
    labels = labels - 1
    labels1 = labels[idx_test].numpy()

    torch.cuda.empty_cache()
    for curr_seed in Seed_List:
        lr = 0.001
        nepochs=300
        print(fts.shape)
        print(G.shape)
        model =HGNN(fts.shape[1],class_count,32,Q=Q,dropout=0)
        model = model.to(device)
       
  
        adam = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer =adam
       
        schedular =  torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100], gamma=0.9)
        print("Model Summary:")
        print(model)
        print('----------------------')
     
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        for i in range(nepochs):
            schedular.step()
            model.train()
            optimizer.zero_grad()
            output= model(fts,G)
            _, preds = torch.max(output[idx_train], 1)
            loss = criterion(output[idx_train], labels[idx_train])
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.eval()
                OUTPUT = model(fts, G)
                _, pred = torch.max(OUTPUT, 1)
                preds=pred[idx_train]
                trainloss = criterion(output[idx_train], labels[idx_train])
                trainOA,_,_,_ =accuracy_eval1(labels[idx_train], preds)  
                print("{}\tloss={}\t train OA={}".format(str(i + 1), trainloss, trainOA))
                torch.save(model.state_dict(), "model\\best_model.pt")
            torch.cuda.empty_cache()
        
        testOA,testAA,testKappa,testcm =accuracy_eval1(labels[idx_train], preds) 
        cm1=np.diag(testcm)/testcm.sum(axis=0)#OA
        file_name = 'classification'+dataset_name+'-150.txt'
        with open(file_name, 'a') as x_file:
            x_file.write('\n') 
            #x_file.write('{}'.format(cm))
            x_file.write('\n')
            x_file.write('{} Overall accuracy (%)'.format(testOA*100))
            x_file.write('\n')
            x_file.write('{} Average accuracy (%)'.format(testAA*100))
            x_file.write('\n')
            x_file.write('{} Kappa accuracy (%)'.format(testKappa*100))
            x_file.write('\n')      
            x_file.write('\n')
            x_file.write('{}'.format(cm1*100))
        pred=pred+1
        pred[background_idx]=0
        y=DrawResult(pred,FLAG)
        #plt.imsave('HGCN'+dataset_name+ repr(int(testOA*10000))+'.png', y)
        torch.cuda.empty_cache()
        del model
