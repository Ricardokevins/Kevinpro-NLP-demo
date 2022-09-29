import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sbs

sbs.set_style('darkgrid')

import collections
import numpy as np
loss_function = torch.nn.MSELoss()
import warnings

warnings.filterwarnings('ignore')

device = 'cuda:0'
def get_sub_task_data(a,b):
    # a = np.random.uniform(0.1, 5.0)
    # b = np.random.uniform(0, 2*np.pi)
    
    train_x = (torch.rand(10,1)-0.5) * 10
    train_x, indices = torch.sort(train_x,dim=0)
    train_y = a * torch.sin(train_x + b)
    # print(train_x)
    # exit()
    #train_y= torch.sin(train_x) * a + b
    #train_y = train_x * a +b
    test_x = (torch.rand(50,1)-0.5) * 10
    test_x, indices = torch.sort(test_x,dim=0)
    test_y = a * torch.sin(test_x + b)
    #test_y= torch.sin(test_x) * a + b
    #test_y = test_x * a +b
    return [train_x.to(device),train_y.to(device)],[test_x.to(device),test_y.to(device)]


#tasks = [[2,3]]
tasks = []

for _ in range(5000):
    a = np.random.uniform(0.1, 5.0)
    b = np.random.uniform(0, 2 * np.pi)
    tasks.append([a,b])

def meta_data():
    return [get_sub_task_data(i[0], i[1]) for i in tasks]

tasks_datas = meta_data()

new_task = [2,3]
new_task_data = get_sub_task_data(new_task[0],new_task[1])
#new_datase

class LinearRegression(torch.nn.Module):
    """
    Linear Regressoin Module, the input features and output 
    features are defaults both 1
    """
    def __init__(self):
        super().__init__()
        #self.linear1 = torch.nn.Linear(1,1)
        self.net=nn.Sequential(
            nn.Linear(1,40),nn.ReLU(),
            nn.Linear(40,40),nn.ReLU(),
            nn.Linear(40,1)
        )
        
    def forward(self,x):
        #out = self.linear1(x)
        return self.net(x)

def test(test_data,model):
    test_x,test_y = test_data
    with torch.no_grad():
        predict = model(test_x)
        loss = loss_function(predict,test_y)
    return loss


    
    # print(model.state_dict())

def fit(net, traind, optim=None, get_test_loss=False, create_graph=False, force_new=False):

    net.train()

    if optim is not None:
        optim.zero_grad()

    train_x,train_y = traind
    predict = net(train_x)
    loss = loss_function(predict,train_y)

    loss.backward(create_graph=create_graph, retain_graph=True)

    if optim is not None:
        optim.step()
    return loss.data.cpu().numpy()#[0]

def eval_sine_test(net, task, fits=(0, 1), lr=0.001):
    model = LinearRegression().to(device)
    model.load_state_dict(net.state_dict())
    train_data,test_data = task
    train_x,train_y = train_data
    model.train()

    optim = torch.optim.SGD(model.parameters(), lr)

    fit_res = []
    if 0 in fits:
        predict = model(train_x)
        loss = loss_function(predict,train_y)
        fit_res.append((0, predict, loss))

    for i in range(np.max(fits)):
        fit(model, train_data, optim)
        if i + 1 in fits:
            predict = model(train_x)
            loss = loss_function(predict,train_y)
            fit_res.append((i+1, predict, loss))

    return fit_res

def New_Tasks_Test(model_name,task, fits=(0, 1), lr=0.01):
    fig = plt.figure(figsize=(10,10))
    for current_fig, m_n in enumerate(model_name, start=1):
        model = LinearRegression().to(device)
        model.load_state_dict(torch.load(m_n))
        train_data,test_data = task
        train_x,train_y = train_data
        test_x,test_y = test_data
        fit_res = eval_sine_test(model, task, fits, lr)
        plt.subplot(1, 2, current_fig)
        train, = plt.plot(train_x.cpu().numpy(), train_y.cpu().numpy(), '^')
        ground_truth, = plt.plot(test_x.cpu().numpy(), test_y.cpu().numpy())
        plots = [train, ground_truth]
        legend = ['Training Points', 'True Function']
        for n, res, loss in fit_res:
            cur, = plt.plot(train_x.cpu().numpy(), res.cpu().data.numpy(), '--')
            plots.append(cur)
            legend.append(f'After {n} Steps')
    plt.legend(plots, legend)
    #plt.show()
    plt.savefig("result.jpg")

def pretrain_learning(data,save=True):
    model = LinearRegression().to(device)
    model.train()
    train_data,test_data = data
    train_x,train_y = train_data
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    patient = 3
    best_loss = 100000000000000000
    for epoch, _ in enumerate(range(500), start=1):
        print("EPOCH: ",epoch,patient)
        predict = model(train_x)
        loss = loss_function(predict,train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        test_loss = test(test_data,model)
        if test_loss < best_loss:
            best_loss = test_loss
            if save:
                torch.save(model.state_dict(), "pretrain.pth")
            patient = 3 
        else:
            patient -= 1
            if patient < 0:
                break

def meta_learning(tasks_data):
    model = LinearRegression().to(device)
    meta_gradient = model.state_dict()
    meta_optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    best_loss = 10000000000000000
    patient = 3
    for epoch in range(100):
        print("EPOCH: ",epoch)
        for k in meta_gradient:
            meta_gradient[k] = torch.zeros_like(meta_gradient[k])
        avg_loss = 0
        for i in tasks_data:
            traind,testd = i

            train_x,train_y = traind
            test_x,test_y = testd

            tasks_model = LinearRegression().to(device)
            tasks_model.load_state_dict(model.state_dict())
            # Train
            fit(tasks_model,traind,create_graph = True)
            
            state_dict = model.state_dict()
            task_state = tasks_model.state_dict()
            for name,param in tasks_model.named_parameters():
                grad = param.grad
                state_dict[name] = task_state[name] - 0.01 * grad
            tasks_model.load_state_dict(state_dict)

            loss = fit(tasks_model,testd)
            avg_loss += loss

            meta_optimizer.step()
            meta_optimizer.zero_grad()
        avg_loss = avg_loss/len(tasks_data)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "meta.pth")
            patient = 3
        else:
            patient -= 1
            if patient < 0:
                break


    

def combine2pretrain(tasks_datas):
    trainX = []
    trainY = []
    testX = []
    testY = []
    for i in tasks_datas:
        traind,testd = i
        train_x,train_y = traind
        test_x,test_y = testd
        trainX.append(train_x)
        trainY.append(train_y)
        testX.append(test_x)
        testY.append(test_y)
    trainX = torch.cat(trainX)
    print(trainX.shape) 
    trainY = torch.cat(trainY)
    testX = torch.cat(testX)
    testY = torch.cat(testY)
    return trainX,trainY,testX,testY

trainX,trainY,testX,testY = combine2pretrain(tasks_datas)
pretrain_learning([[trainX,trainY],[testX,testY]])
meta_learning(tasks_datas)
New_Tasks_Test(['meta.pth','pretrain.pth'],new_task_data,(0,1,10,100))
#New_Tasks_Test(,new_task_data,(0,1,10,100))
# linear = Linear_Model()
# #linear.test({'x':testX,'y':testY},'pretrain.pth')
# linear.test({'x':testX,'y':testY},'meta.pth')


def adapt2New(new_task_data,models,names):
    data = {'model': [], 'fits': [], 'loss': [], 'set': []}
    new_task_data
    for name, models in smodels:

        if not isinstance(models, list):

            models = [models]

        for n_model, model in enumerate(models):

            for n_test, test in enumerate(SINE_TEST):
                n_test = n_model * len(SINE_TEST) + n_test
                fit_res = eval_sine_test(model, test, fits, lr)

                for n, _, loss in fit_res:

                    data['model'].append(name)

                    data['fits'].append(n)

                    data['loss'].append(loss)

                    data['set'].append(n_test)

    
    print(len(data['loss']))
    ax = sbs.tsplot(
        pd.DataFrame(data), condition='model', value='loss',time='fits', unit='set', marker=marker, linestyle=linestyle)

    plt.show()
    