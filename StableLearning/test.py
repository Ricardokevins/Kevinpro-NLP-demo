from sklearn import datasets
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


digits = datasets.load_digits()
# print(digits.data)  
# print(type(digits.data))
# print(digits.data.shape)
# print(type(digits.target))
# print(digits.target)


def convert2fake_label(i):
    return i
    #return 0 if i > 4 else 1


def count_distribution(data):
    count = {}
    for i in data.target:
        if i not in count:
            count[i] = 1
        else:
            count[i] += 1
    print("Original Distribution", count)
    return count

def balance_train_sample(count):
    for i in count:
        count[i] = int(count[i] * 0.5)
    print("AfterSample Distribution",count)
    return count

def unbalance_train_sample(count):
    for i in count:
        count[i] = int(count[i] * 0.95) if i % 2 == 1 else int(count[i] * 0.05)
    print("AfterSample Distribution",count)
    return count

def generate_split(data):
    # Over sample data with ood 1,3,5,7,9 90% 
    count = count_distribution(data)
    total_sample_number = sum(count[i] for i in count)
    count = unbalance_train_sample(count)
    train_sample_number = sum(count[i] for i in count)
    print(total_sample_number,train_sample_number,train_sample_number/total_sample_number)
    train_data = []
    test_data = []
    train_label = []
    origin_label = [ ]
    test_label = []
    for d,l in zip(data.data,data.target):
        if count[l] > 0: #continue sample
            train_data.append(d)
            train_label.append(convert2fake_label(l))
            count[l] -= 1
            origin_label.append(l)
        else:
            test_data.append(d)
            test_label.append(convert2fake_label(l))

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    # print(train_data.shape)
    # print(test_data.shape)
    return train_data,train_label,test_data,test_label,origin_label

train_data,train_label,test_data,test_label,origin_label = generate_split(digits)




def column_wise_resampling(x, replacement = False, random_state = 0, **options):
    """
    Perform column-wise random resampling to break the joint distribution of p(x).
    In practice, we can perform resampling without replacement (a.k.a. permutation) to retain all the data points of feature x_j. 
    Moreover, if the practitioner has some priors on which features should be permuted,
    it can be passed through options by specifying 'sensitive_variables', by default it contains all the features
    """
    rng = np.random.RandomState(random_state)
    n, p = x.shape
    if 'sensitive_variables' in options:
        sensitive_variables = options['sensitive_variables']
    else:
        sensitive_variables = [i for i in range(p)] 
    x_decorrelation = np.zeros([n, p])
    for i in sensitive_variables:
        var = x[:, i]
        # print(var.shape)
        # print(x.shape)
        # exit()
        if replacement: # sampling with replacement
            x_decorrelation[:, i] = np.array([var[rng.randint(0, n)] for j in range(n)])
        else: # permutation     
            x_decorrelation[:, i] = var[rng.permutation(n)]
    return x_decorrelation

def decorrelation(x, solver = 'adam', hidden_layer_sizes = (5,5), max_iter = 500, random_state = 0):
    """
    Calcualte new sample weights by density ratio estimation
           q(x)   P(x belongs to q(x) | x) 
    w(x) = ---- = ------------------------ 
           p(x)   P(x belongs to p(x) | x)
    """
    n, p = x.shape
    x_decorrelation = column_wise_resampling(x, random_state = random_state)
    P = pd.DataFrame(x)
    Q = pd.DataFrame(x_decorrelation)
    P['src'] = 1 # 1 means source distribution
    Q['src'] = 0 # 0 means target distribution
    Z = pd.concat([P, Q], ignore_index=True, axis=0)
    labels = Z['src'].values
    Z = Z.drop('src', axis=1).values
    P, Q = P.values, Q.values
    # train a multi-layer perceptron to classify the source and target distribution
    # clf = MLPClassifier(solver=solver, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state)
    clf = MLPClassifier()
    clf.fit(Z, labels)
    proba = clf.predict_proba(Z)[:len(P), 1]

    # y_pred_model = clf.predict(Z)
    # right = sum(i == j for i, j in zip(y_pred_model,labels))
    # print(f"TEst Total {len(y_pred_model)} Right {right} Acc Ratio {right / len(y_pred_model) * 100}%")
    
    # weights = (1 - proba) - 1. # calculate sample weights by density ratio
    # weights /= np.mean(weights) # normalize the weights to get average 1

    weights = (1./proba) - 1. # calculate sample weights by density ratio
    weights /= np.mean(weights) # normalize the weights to get average 1
    weights = np.reshape(weights, [n,1])
    return weights

def similarity_weight(x):
    weights = []

    from scipy.spatial.distance import cosine
    for i in x:
        similar = [cosine(i,j) for j in x]
        # print(similar)
        # exit()
        weights.append(sum(similar)/len(similar))
    weights = np.array(weights)
    weights /= np.mean(weights)
    return weights


simi_weight = similarity_weight(train_data)

weights = decorrelation(train_data)
weights = weights.reshape(-1)
weights_sum = {i: [] for i in range(10)}
for i,j in zip(weights,origin_label):
    weights_sum[j].append(i)
for i in range(10):
    print(len(weights_sum[i]))
    weights_sum[i] = sum(weights_sum[i])/len(weights_sum[i])
    
print(weights_sum)
svc = svm.SVC()  # 支持向量机，SVM
dt = DecisionTreeClassifier()
mlp = MLPClassifier()

manual_weight = []
for i,j in zip(weights,origin_label):
    if j % 2 == 0:
        manual_weight.append(0.8)
    else:
        manual_weight.append(0.2)

manual_weight = np.array(manual_weight)
model = mlp
#model = svc
model = LogisticRegression()
model.fit(train_data, train_label)
print('begin to predict……')
y_pred_model = model.predict(test_data)
right = sum(i == j for i, j in zip(y_pred_model,test_label))
print(f"Baseline Total {len(test_label)} Right {right} Acc Ratio {right / len(test_label) * 100}%")

model = LogisticRegression()
model.fit(train_data, train_label,sample_weight=weights)
print('DeWeight begin to predict……')
y_pred_model = model.predict(test_data)
right = sum(i == j for i, j in zip(y_pred_model,test_label))
print(f"Total {len(test_label)} Right {right} Acc Ratio {right / len(test_label) * 100}%")

model = LogisticRegression()
model.fit(train_data, train_label,sample_weight=simi_weight)
print('SimiWeight begin to predict……')
y_pred_model = model.predict(test_data)
right = sum(i == j for i, j in zip(y_pred_model,test_label))
print(f"Total {len(test_label)} Right {right} Acc Ratio {right / len(test_label) * 100}%")

model = LogisticRegression()
model.fit(train_data, train_label,sample_weight=manual_weight)
print('Manual begin to predict……')
y_pred_model = model.predict(test_data)
right = sum(i == j for i, j in zip(y_pred_model,test_label))
print(f"Total {len(test_label)} Right {right} Acc Ratio {right / len(test_label) * 100}%")

def visual(train_data,train_label,test_data,test_label):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    fig = plt.figure()
    #先不降维，只对数据进行投影，看看投影后的三个维度的方差分布
    from sklearn.decomposition import PCA
    #进行降维，从三维降到2维
    pca1 = PCA(n_components=2)
    X = np.concatenate((train_data,test_data),axis=0)
    Y = np.concatenate((train_label,test_label+2),axis=0)
    pca1.fit(X)
    X_new = pca1.transform(X)
    print(X_new[:, 0].reshape(-1).shape)
    print(X_new[:, 1].shape)
    colors = ['red','yellow','green','blue']
    labels = ['train_0','train_1','test_0','test_1']
    for i in range(4):
        plt.scatter(X_new[Y==i, 0].tolist(), X_new[Y==i, 1].tolist(),marker='o',c = colors[i],label = labels[i])
    plt.legend()
    plt.show()

    labels = ['train','test']
    for i in range(4):
        if i < 2:
            plt.scatter(X_new[Y==i, 0].tolist(), X_new[Y==i, 1].tolist(),marker='o',c = colors[0],label = labels[0])
        else:
            plt.scatter(X_new[Y==i, 0].tolist(), X_new[Y==i, 1].tolist(),marker='o',c = colors[2],label = labels[1])
    plt.legend()
    plt.show()
#visual(train_data,train_label,test_data,test_label)
# SVM
# 80.60200668896321% Vs 96.00443951165371%
# 59.75473801560758% VS 89.45615982241954% Weighted




# MLP
# 74.91638795986621% VS 93.00776914539401%

# 78.14938684503902%

# 89.40914158305463% 97.2253052164262%