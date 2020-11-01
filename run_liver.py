import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import timeit
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file


def projection(u):
    return torch.clamp(torch.clamp(u, max=1), min=-1)

class MyLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, y, lam):
        y_hat = torch.sigmoid(x.mv(weight) + bias)
        Pw = projection(weight)
        ctx.save_for_backward(x, Pw, y_hat, y)
        ctx.lam = lam
        loss = -torch.mean(y*torch.log(y_hat) + (1-y) * torch.log(1-y_hat))+lam* torch.norm(weight-Pw,1)
        # loss = torch.mean((y_hat-y)**2)+lam* torch.norm(weight-Pw,1)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x, Pw, y_hat, y = ctx.saved_tensors
        grad_weight = (1/x.shape[0]) * (x.t().mv(y_hat - y) + ctx.lam * Pw)
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
        self.y = labels
    
    def forward(self, x):
        return MyLoss.apply(x, self.weight, self.bias, self.y, self.lam)
    
    def predict(self, x):
        w = self.weight
        a = x.mv(w) + self.bias
        return (torch.sigmoid(a) > 0.5).float()

if __name__ == "__main__":
    train_path = '.\datasets\\liver-disorders'
    test_path = '.\datasets\\liver-disorders.t'

    train_set_x, train_set_y = load_svmlight_file(train_path)
    train_set_x = train_set_x.todense()
    train_set_y[train_set_y==-1] = 0

    torch.random.manual_seed(0)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    
    N = train_set_x.shape[0]       # batch size
    in_dim = train_set_x.shape[1]  # input dimension
    out_dim = 1 # output dimension
    learning_rate = 0.01
    tolerance = 1e-7
    max_itr = 100

    # preprocessing
    scaler = preprocessing.StandardScaler()
    train_set_x = scaler.fit_transform(train_set_x)

    # np -> tensor
    x = torch.from_numpy(train_set_x).float().to(device)
    y = torch.from_numpy(train_set_y).float().to(device).squeeze_()

    # Construct our model by instantiating the class defined above.
    model = LogisticRegressionNet(in_dim, out_dim, y,lam=1.0,device=device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    loss_prev = float('inf')
    start = timeit.default_timer()
    for t in range(max_itr+1):
        loss = model(x)
        # if (t % 5 == 0):
        #     print(t, loss.item())
        
        if(abs(loss - loss_prev) <= tolerance):
            break
        
        # Zero gradients, perform a backward pass, and update the weights.
        loss_prev = loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    stop = timeit.default_timer()
    time_our_method = stop - start
    print('Number of itr : ', t)
    print('Time (our method): ', time_our_method)  

    # testing
    test_set_x, test_set_y = load_svmlight_file(test_path)
    test_set_x = test_set_x.todense()
    scaler = preprocessing.StandardScaler()
    test_set_x = scaler.fit_transform(test_set_x)
    test_set_y[test_set_y==-1] = 0
    x_test = torch.from_numpy(test_set_x).float().to(device)
    y_test = torch.from_numpy(test_set_y).float().to(device).squeeze_()
    y_predict = model.predict(x_test)
    acc = 100 - torch.mean(abs(y_predict - y_test))*100
    print ("accurency (our method): ", acc.item())

    #SKLearn Logistic Regression
    #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    logreg = LogisticRegression(C=1.0, penalty='l1', tol=tolerance, max_iter=max_itr, solver='liblinear')
    start = timeit.default_timer()
    logreg.fit(train_set_x, train_set_y.squeeze())
    stop = timeit.default_timer()
    time_lr_sklearn = stop - start
    print('Time (sklearn LR): ', time_lr_sklearn)  
    y_pred = logreg.predict(test_set_x)
    print ("accurency (sklearn LR): ", 100 - np.mean(abs(y_pred - test_set_y))*100)