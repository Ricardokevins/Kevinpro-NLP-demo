import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import collections
loss_function = torch.nn.MSELoss()

def get_sub_task_data(a,b):
    train_x = (torch.rand(10000,1)-0.5) * 10
    train_y= torch.sin(train_x) * a + b
    #train_y = train_x * a +b
    test_x = (torch.rand(2000,1)-0.5) * 10
    test_y= torch.sin(test_x) * a + b
    #test_y = test_x * a +b
    return [train_x,train_y],[test_x,test_y]

#tasks = [[2,3],[-3,1],[7,4],[-9,2],[1,-20],[4,5.5]]
tasks = [[2,3]]
def meta_data():
    return [get_sub_task_data(i[0], i[1]) for i in tasks]

tasks_datas = meta_data()


class LinearRegression(torch.nn.Module):
    """
    Linear Regressoin Module, the input features and output 
    features are defaults both 1
    """
    def __init__(self):
        super().__init__()
        #self.linear1 = torch.nn.Linear(1,1)
        self.net=nn.Sequential(
            nn.Linear(in_features=1,out_features=10),nn.ReLU(),
            nn.Linear(10,100),nn.ReLU(),
            nn.Linear(100,10),nn.ReLU(),
            nn.Linear(10,1)
        )
        
    def forward(self,x):
        out = self.net(x)
        #out = self.linear1(x)
        return out

class Linear_Model():
    def __init__(self):
        """
        Initialize the Linear Model
        """
        self.learning_rate = 0.001
        self.epoches = 10000
        self.loss_function = torch.nn.MSELoss()
        self.create_model()

    def create_model(self):
        self.model = LinearRegression()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
    
    def train(self, data, model_save_path="model.pth"):
        """
        Train the model and save the parameters
        Args:
            model_save_path: saved name of model
            data: (x, y) = data, and y = kx + b
        Returns: 
            None
        """
        x = data["x"]
        y = data["y"]
        for epoch in range(self.epoches):
            prediction = self.model(x)
            loss = self.loss_function(prediction, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 500 == 0:
                print("epoch: {}, loss is: {}".format(epoch, loss.item()))
        torch.save(self.model.state_dict(), "linear.pth")
      
        
    def test(self, data, model_path="linear.pth"):
        """
        Reload and test the model, plot the prediction
        Args:
            model_path: the model's path and name
            data: (x, y) = data, and y = kx + b
        Returns:
            None
        """
        x = data["x"]
        y = data["y"]
        self.model.load_state_dict(torch.load(model_path))
        prediction = self.model(x)
        
        plt.scatter(x.numpy(), y.numpy(), c=x.numpy())
        plt.scatter(x.numpy(), prediction.detach().numpy(), color="r")
        #plt.plot(x.numpy(), prediction.detach().numpy(), color="r")
        plt.show()
        
    def compare_epoches(self, data):
        x = data["x"]
        y = data["y"]
        
        num_pictures = 16
        fig = plt.figure(figsize=(10,10))
        current_fig = 0
        for epoch in range(self.epoches):
            prediction = self.model(x)
            loss = self.loss_function(prediction, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if epoch % (self.epoches/num_pictures) == 0:
                current_fig += 1
                plt.subplot(4, 4, current_fig)
                plt.scatter(x.numpy(), y.numpy(), c=x.numpy())
                plt.plot(x.numpy(), prediction.detach().numpy(), color="r")
        plt.show()

def test(test_data,model):
    test_x,test_y = test_data
    with torch.no_grad():
        predict = model(test_x)
        loss = loss_function(predict,test_y)
    return loss


    
    # print(model.state_dict())

def pretrain_learning(data):
    model = LinearRegression()
    model.train()
    train_data,test_data = data
    train_x,train_y = train_data
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
    best_loss = 100000000000000000
    for epoch, _ in enumerate(range(100), start=1):
        print("EPOCH: ",epoch)
        predict = model(train_x)
        loss = loss_function(predict,train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        test_loss = test(test_data,model)
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), "pretrain.pth")

# worker_state_dict = [x.state_dict() for x in models]
# weight_keys = list(worker_state_dict[0].keys())
# fed_state_dict = collections.OrderedDict()
# for key in weight_keys:
#     key_sum = 0
#     for i in range(len(models)):
#         key_sum = key_sum + worker_state_dict[i][key]
#     fed_state_dict[key] = key_sum / len(models)
# #### update fed weights to fl model
# fl_model.load_state_dict(fed_state_dict)

def meta_learning(tasks_data):
    model = LinearRegression()
    for epoch in range(100):
        print("EPOCH: ",epoch)
        for i in tasks_data:
            traind,testd = i

            train_x,train_y = traind
            test_x,test_y = testd

            tasks_model = LinearRegression()
            tasks_model.net.load_state_dict(model.net.state_dict())
            # Train

            optimizer = torch.optim.SGD(tasks_model.parameters(), lr = 0.01)
            predict = tasks_model(train_x)
            loss = loss_function(predict,train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(tasks_model.state_dict(),loss)
            
            

            # Test
            optimizer = torch.optim.SGD(tasks_model.parameters(), lr = 0.001)
            predict = tasks_model(test_x)
            loss = loss_function(predict,test_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(tasks_model.state_dict(),loss)
            # print(model.state_dict())

            worker_state_dict = tasks_model.state_dict()
            # for i in worker_state_dict:
            #     print(i,worker_state_dict[i])
            # print(worker_state_dict)
            # #weight_keys = list(worker_state_dict[0].keys())
            # print(weight_keys)
            # print(worker_state_dict)
            # exit()
            #print(model.state_dict())
            fed_state_dict = collections.OrderedDict()
            for i in worker_state_dict:
                fed_state_dict[i] = model.state_dict()[i] - 0.01 * worker_state_dict[i]
            model.load_state_dict(fed_state_dict)
            #print(model.state_dict())
        #print(model.state_dict())
        #model.linear1.data = model.linear1.data-0.01*model.linear1.data
    torch.save(model.state_dict(), "meta.pth")

#model.linear1.state_dict() 
    # ori_theta = ori_theta - beta * meta_gradient / tasksmodel.linear1.state_dict() 

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

# train_data1,test_data1 = task1_data
# train_data2,test_data2 = task2_data

# train_x1,train_y1 = train_data1
# test_x1,test_y1 = test_data1
# train_x2,train_y2 = train_data2
# test_x2,test_y2 = test_data2
# train_x = torch.cat((train_x1,train_x2))
# train_y = torch.cat((train_y1,train_y2))
# test_x = torch.cat((test_x1,test_x2))
# test_y = torch.cat((test_y1,test_y2))



# # linear.train(data)
# # linear.test(data)
# data = {'x':trainX,'y':trainY}
# linear.compare_epoches(data)

pretrain_learning([[trainX,trainY],[testX,testY]])
meta_learning(tasks_datas)
    

linear = Linear_Model()
linear.test({'x':testX,'y':testY},'pretrain.pth')
linear.test({'x':testX,'y':testY},'meta.pth')