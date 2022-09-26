import sys

sys.path.append("")
import unittest
from pickle import load

import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)
from causallearn.search.FCMBased import lingam
import io
import unittest
from itertools import product
from causallearn.search.FCMBased.lingam.CAMUV import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from causallearn.graph.Dag import Dag
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.Granger.Granger import Granger
from causallearn.utils.cit import fisherz
from causallearn.utils.DAG2PAG import dag2pag
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.TimeseriesVisualization import plot_time_series

class TestDirectLiNGAM(unittest.TestCase):

    def test_DirectLiNGAM(self):
        np.set_printoptions(precision=3, suppress=True)
        np.random.seed(100)
        x3 = np.random.uniform(size=1000)
        x0 = 3.0 * x3 + np.random.uniform(size=1000)
        x2 = 6.0 * x3 + np.random.uniform(size=1000)
        x1 = 3.0 * x0 + 2.0 * x2 + np.random.uniform(size=1000)
        x5 = 4.0 * x0 + np.random.uniform(size=1000)
        x4 = 8.0 * x0 - 1.0 * x2 + np.random.uniform(size=1000)
        X = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T, columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])

        model = lingam.DirectLiNGAM()
        model.fit(X)

        print(model.causal_order_)
        print(model.adjacency_matrix_)
    
    def test(self):
        dataset = pd.read_csv('kc_house_data.csv')
        X = dataset.drop(['id','date'], axis = 1)
        from sklearn.preprocessing import MinMaxScaler
        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(X)
        newX = minmax_scaler.transform(X)
        X = pd.DataFrame(newX, columns=X.columns)

        model = lingam.DirectLiNGAM()
        model.fit(X)
        # print(model.causal_order_)
        # #print(model.adjacency_matrix_)
        # print(model.adjacency_matrix_.shape)
        print(X.columns.values.tolist())
        names = X.columns.values.tolist()
        print(len(names))
        # print(np.array(X).shape)
        # exit()
        P,U = execute(np.array(X[:500]), 0.01, 18)
        for i, result in enumerate(P):
            if not len(result) == 0:
                print("child: " + str(names[i]) + ",  parents: "+ str([names[j] for j in result]))
        for result in U:
            print(result)
        #print(model.adjacency_matrix_[2])
        #print(model.adjacency_matrix_[:,2])
        #print(dataset.head)
        # child: bedrooms,  parents: ['sqft_living15', 'sqft_living']
        # child: bathrooms,  parents: ['sqft_above', 'floors', 'lat']
        # child: sqft_living,  parents: ['price', 'sqft_basement', 'sqft_lot', 'floors']
        # child: view,  parents: ['waterfront']
        # child: grade,  parents: ['price', 'sqft_basement']
        # child: sqft_above,  parents: ['price', 'condition', 'sqft_basement', 'lat']
        # child: yr_built,  parents: ['price', 'bedrooms', 'bathrooms', 'floors', 'condition', 'yr_renovated']
        # child: zipcode,  parents: ['sqft_lot15', 'sqft_lot', 'yr_built']
        # child: lat,  parents: ['view']
        # child: long,  parents: ['bathrooms', 'sqft_basement']
        # child: sqft_living15,  parents: ['price', 'grade', 'sqft_above', 'sqft_basement']
        # child: sqft_lot15,  parents: ['condition']
        for i in range(len(names)):
            print(names[i],round(model.adjacency_matrix_[0,i],0))


        meandata = np.mean(model.adjacency_matrix_)
        print(meandata)
        nodes = [GraphNode(names[i]) for i in range(len(names))]
        dag1 = Dag(nodes)
        index_j = 0
        for index_i, i in enumerate(model.adjacency_matrix_):
            for j in i:
                if j> 0.4:
                    #print(j)
                    dag1.add_directed_edge(nodes[index_j], nodes[index_i])
                index_j += 1
            index_j = 0

        pyd = GraphUtils.to_pydot(dag1)
        pyd.write_png('dag.png')

def test2():
    import numpy as np
    import random
    import causallearn.search.FCMBased.lingam.CAMUV
    P,U = CAMUV.execute(data,0.01,3)
t = TestDirectLiNGAM()
t.test()
