import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from lr_utils import load_dataset
import timeit
from sklearn.linear_model import LogisticRegression
import dataset_utils.open_libsvm
from dataset_utils.open_libsvm import open_libsvm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def projection(u):
    # return np.maximum(-1, np.minimum(u, 1))
    return torch.clamp(torch.clamp(u, max=1), min=-1)

class MyLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, y, lam):
        y_hat = torch.sigmoid(x.mv(weight) + bias)
        Pw = projection(weight)
        ctx.save_for_backward(x, Pw, y_hat, y)
        ctx.lam = lam
        # loss = -torch.mean(y*torch.log(y_hat) + (1-y) * torch.log(1-y_hat))+lam* torch.norm(weight-Pw,1)
        loss = torch.mean((y_hat-y)**2)+lam* torch.norm(weight-Pw,1)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x, Pw, y_hat, y = ctx.saved_tensors
        # print('m =', x.shape[0])
        grad_weight = (1/x.shape[0]) * (x.t().mv(y_hat - y) + ctx.lam * Pw)
        # print('grad_weight =', grad_weight[0])
        grad_bias = torch.mean(y_hat - y)
        grad_bias.unsqueeze_(0)
        return None, grad_weight, grad_bias, None, None

class LogisticRegressionNet(nn.Module):
    def __init__(self, in_dim, out_dim, labels, lam=0.1, device='cpu'):
        super().__init__()
        self.input_features = in_dim
        self.output_features = out_dim
        self.lam = lam
        self.weight = nn.Parameter(torch.zeros(in_dim, device=device))
        self.bias = nn.Parameter(torch.zeros(out_dim,device=device))
        # self.weight.data.uniform_(-0.0001, 0.0001)
        # self.bias.data.uniform_(-0.0001, 0.0001)
        self.y = labels
    
    def forward(self, x):
        # out = self.l1(x)
        # out = F.sigmoid(out)
        return MyLoss.apply(x, self.weight, self.bias, self.y, self.lam)
    
    def predict(self, x):
        w = self.weight
        # w = u - projection(u)
        a = x.mv(w) + self.bias
        return (torch.sigmoid(a) > 0.5).float()

if __name__ == "__main__":
    train_set_x, train_set_y = open_libsvm('.\datasets\ionosphere_scale')
    test_set_x, test_set_y = open_libsvm('.\datasets\ionosphere_scale')

    torch.random.manual_seed(0)
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    
    N = train_set_x.shape[0]       # batch size
    in_dim = train_set_x.shape[1]  # input dimension
    out_dim = 1  # output dimension

    # preprocessing
    # min_max_scaler = preprocessing.MinMaxScaler()
    # train_set_x = min_max_scaler.fit_transform(train_set_x)

    # min_max_scaler = preprocessing.MinMaxScaler()
    # test_set_x = min_max_scaler.fit_transform(test_set_x)

    # np -> tensor
    x = torch.from_numpy(train_set_x).float().to(device)
    y = torch.from_numpy(train_set_y).float().to(device).squeeze_()

    x_test = torch.from_numpy(test_set_x).float().to(device)
    y_test = torch.from_numpy(test_set_y).float().to(device).squeeze_()
    # Construct our model by instantiating the class defined above.
    model = LogisticRegressionNet(in_dim, out_dim, y,lam=0.0,device=device)

    learning_rate = 1e-4
    tolerance = 1e-9
    max_itr = 10000
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=max_itr, max_eval=None, tolerance_grad=1e-07, tolerance_change=tolerance, history_size=100, line_search_fn=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    start = timeit.default_timer()
    loss_prev = float('inf')
    for t in range(max_itr+1):
        # Forward pass: Compute predicted y by passing x to the model
        loss = model(x)
        # if (t % 100 == 0):
        #     print(t, loss.item())
        
        # Zero gradients, perform a backward pass, and update the weights.
        if(abs(loss - loss_prev) <= tolerance):
            break
        loss_prev = loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    stop = timeit.default_timer()
    time_our_method = stop - start
    print('Number of itr : ', t)
    print('Time (our method): ', time_our_method)  
    y_predict = model.predict(x_test)
    acc = 100 - torch.mean(abs(y_predict - y_test))*100
    print ("accurency (our method): ", acc.item())

    #SKLearn Logistic Regression
    #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    logreg = LogisticRegression(C=1.0, penalty='l1', tol=tolerance, max_iter=max_itr)
    start = timeit.default_timer()
    logreg.fit(train_set_x, train_set_y.squeeze())
    stop = timeit.default_timer()
    time_lr_sklearn = stop - start
    print('Time (sklearn LR): ', time_lr_sklearn)  
    y_pred = logreg.predict(test_set_x)
    print ("accurency (sklearn LR): ", 100 - np.mean(abs(y_pred - test_set_y))*100)