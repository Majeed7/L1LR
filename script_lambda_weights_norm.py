import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import timeit
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt

def projection(u):
    return torch.clamp(torch.clamp(u, max=1), min=-1)

class MyLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, y, lam):
        Pw = projection(weight)
        y_hat = torch.sigmoid(x.mv(weight) + bias)
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
    train_path = '.\datasets\\madelon'

    train_set_x, train_set_y = load_svmlight_file(train_path)
    train_set_x = train_set_x.todense()
    train_set_y[train_set_y==-1] = 0

    torch.random.manual_seed(0)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    
    N = train_set_x.shape[0]       # batch size
    in_dim = train_set_x.shape[1]  # input dimension
    out_dim = 1 # output dimension
    learning_rate = 1e-2
    max_itr = 100

    # preprocessing
    scaler = preprocessing.StandardScaler()
    train_set_x = scaler.fit_transform(train_set_x)

    # np -> tensor
    x = torch.from_numpy(train_set_x).float().to(device)
    y = torch.from_numpy(train_set_y).float().to(device)
    
    lam_list = np.arange(0, 10.1, 0.1)
    weights_norm1 = np.zeros(len(lam_list))
    
    #run model with different lambdas
    for ind, lam in enumerate(lam_list):
        # Construct our model by instantiating the class defined above.
        model = LogisticRegressionNet(in_dim, out_dim, y,lam=lam,device=device)

        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        for t in range(max_itr):
            loss = model(x)
            loss.item()
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
        weights_norm1[ind] = torch.norm(model.weight, 1).item()


    # plot
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['figure.dpi'] = 300
    plt.figure(figsize=(9,6))
    plt.rcParams.update({'font.size': 18})
    plt.plot(lam_list, weights_norm1)
    plt.xlabel(r'$\lambda$')
    plt.xticks(np.arange(0, 10.1, 0.5),fontsize=12)
    plt.tick_params(labelrotation=90)
    plt.ylabel(r'${||w||}_1$')
    plt.title(r'The $l_1$-norm of the weights with $\lambda$ $\in$ [0,10]')
    plt.savefig('./chart_norm_vs_lambda.eps', format='eps', bbox_inches='tight')
    plt.savefig('./chart_norm_vs_lambda.png', format='png', bbox_inches='tight')
    plt.show()