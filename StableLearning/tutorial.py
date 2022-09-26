#importing all the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

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
    clf = MLPClassifier(solver=solver, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state)
    print(Z.shape)
    clf.fit(Z, labels)
    proba = clf.predict_proba(Z)[:len(P), 1]
    y_pred_model = clf.predict(Z)
    right = sum(i == j for i, j in zip(y_pred_model,labels))
    print(f"TEst Total {len(y_pred_model)} Right {right} Acc Ratio {right / len(y_pred_model) * 100}%")
    weights = (1./proba) - 1. # calculate sample weights by density ratio
    weights /= np.mean(weights) # normalize the weights to get average 1
    weights = np.reshape(weights, [n,1])
    return weights

dataset = pd.read_csv('kc_house_data.csv')
dataset = dataset.drop(['id','date'], axis = 1)
print(dataset.head)
#print(dataset)
train_data =  []
def split_data(start,end):
    temp = dataset[dataset['yr_built']<=end]
    temp = temp[temp['yr_built'] >= start]
    return temp

d1900 = split_data(1900,1919)
d1920 = split_data(1920, 1939)
d1940= split_data(1940, 1959)
d1950 = split_data(1960, 1979)
d1980 = split_data(1980, 1999)
d1999 = split_data(1999, 2015)

from sklearn.linear_model import LogisticRegression

from math import sqrt
def RMSE(y_pred,y_test):
    loss = [(i-j)**2 for i,j in zip(y_pred,y_test)]
    loss = sqrt(sum(loss))/len(loss)
    return loss

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

def test(dataset,regressor):
    X = dataset.iloc[:,1:].values
    #X = normalize(X, axis=0, norm='max')
    y = dataset.iloc[:,0].values
    y_pred = regressor.predict(X)
    print(RMSE(y_pred,y))

def train(dataset):
    # print(dataset.iloc[:,1:].columns.values.tolist())
    # exit()
    X = dataset.iloc[:,1:].values
    y = dataset.iloc[:,0].values
    #splitting dataset into training and testing dataset
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
    
    regressor = LinearRegression()
    
    #X = normalize(X, axis=0, norm='max')
    data = np.array(X)
    weight = decorrelation(data)
    weight = weight.reshape(-1)
    #regressor.fit(X, y,sample_weight = weight)
    regressor.fit(X, y)
    # y_pred = regressor.predict(X_test)
    # print("TRAINSET",RMSE(y_pred,y_test))
    y_pred = regressor.predict(X)
    print("TRAINSET",RMSE(y_pred,y))
    return regressor

model = train(d1900)
test(d1900,model)
test(d1920,model)
test(d1940,model)
test(d1950,model)
test(d1980,model)
test(d1999,model)





