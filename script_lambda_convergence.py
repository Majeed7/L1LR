import torch

import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt

from model_LR_NN_PR import LogisticRegressionNet

if __name__ == "__main__":
    train_path = './datasets//madelon'

    train_set_x, train_set_y = load_svmlight_file(train_path)
    train_set_x = train_set_x.todense()
    train_set_y[train_set_y==-1] = 0

    torch.random.manual_seed(0)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    
    N = train_set_x.shape[0]       # batch size
    in_dim = train_set_x.shape[1]  # input dimension
    out_dim = 1 # output dimension
    learning_rate = 4e-3
    max_itr = 100

    # preprocessing
    scaler = preprocessing.StandardScaler()
    train_set_x = scaler.fit_transform(train_set_x)

    # np -> tensor
    x = torch.from_numpy(train_set_x).float().to(device)
    y = torch.from_numpy(train_set_y).float().to(device)
    
    lam_list = [1, 10, 50, 100]
    costs = np.zeros((len(lam_list), max_itr))
    
    #run model with different lambdas
    for ind, lam in enumerate(lam_list):
        # Construct our model by instantiating the class defined above.
        model = LogisticRegressionNet(in_dim, out_dim, y,lam=lam,device=device)

        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        for t in range(max_itr):
            loss = model(x)
            loss.item()
            costs[ind][t] = loss.item()
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    # plot
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['figure.dpi'] = 300
    plt.figure(figsize=(9,6))
    linestyles = ['-', '--', '-.', ':']
    for i in range(costs.shape[0]):
        plt.plot(costs[i], label=f'$\lambda$ = {lam_list[i]}', linestyle=linestyles[i % len(linestyles)])
    plt.xlabel('Iterations')
    plt.ylabel('Objective function value')
    plt.title(r'Objective function values with $\lambda$ $\in$ {1, 10, 50, 100}')
    plt.legend()
    plt.savefig('./chart_obj_vs_lambda.eps', format='eps', bbox_inches='tight')
    plt.savefig('./chart_obj_vs_lambda.png', format='png', bbox_inches='tight')
    plt.show()