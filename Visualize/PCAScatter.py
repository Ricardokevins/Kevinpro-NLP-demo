import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
#X为样本特征，Y为样本簇类别， 共1000个样本，每个样本3个特征，共4个簇
X, y = make_blobs(n_samples=10000, n_features=3, centers=[[3,3, 3], [0,0,0], [1,1,1], [2,2,2]], cluster_std=[0.2, 0.1, 0.2, 0.2], 
                  random_state =9)
fig = plt.figure()

#先不降维，只对数据进行投影，看看投影后的三个维度的方差分布
from sklearn.decomposition import PCA
#进行降维，从三维降到2维
pca1 = PCA(n_components=2)
pca1.fit(X)

'''通过对比，因为上面三个投影后的特征维度的方差分别为：
[ 3.78483785  0.03272285  0.03201892]，投影到二维后选择的肯定是前两个特征，而抛弃第三个特征'''
#将降维后的2维数据进行可视化
X_new = pca1.transform(X)
print(X_new[:, 0].reshape(-1).shape)
print(X_new[:, 1].shape)
plt.scatter(X_new[:, 0].tolist(), X_new[:, 1].tolist(),marker='o')
plt.show()