import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file

from model_LR_NN_PR import LogisticRegressionNet

if __name__ == "__main__":
    dataset_name = "ijcnn1"
    train_path = f'.\datasets\\{dataset_name}'

    train_set_x, train_set_y = load_svmlight_file(train_path)
    train_set_x = train_set_x.todense()
    train_set_y[train_set_y==-1] = 0

    torch.random.manual_seed(0)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    
    N = train_set_x.shape[0]       # batch size
    in_dim = train_set_x.shape[1]  # input dimension
    out_dim = 1 # output dimension
    learning_rate = 1e-3
    max_itr = 50

    # preprocessing
    scaler = preprocessing.StandardScaler()
    train_set_x = scaler.fit_transform(train_set_x)

    # np -> tensor
    x = torch.from_numpy(train_set_x).float().to(device)
    y = torch.from_numpy(train_set_y).float().to(device)
    
    lam_list = [10**i for i in range(0, 7, 1)]
    lam_list.extend([5, 20])
    lam_list.sort()
    weights = np.zeros((len(lam_list), in_dim))
    
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
 
        weights[ind] = model.weight.detach().numpy()


    # plot
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['figure.dpi'] = 300
    # plt.figure(figsize=(15,8))
    plt.yticks(np.arange(len(lam_list)), lam_list,fontsize=8)
    plt.ylabel(r"$\lambda$", fontsize=22)
    plt.xlabel("Weights vector", fontsize=22)
    plt.xticks([])
    # plt.imshow(weights, aspect='auto')
    plt.pcolormesh(weights,shading='flat', edgecolors='k', linewidths=0.25)
    plt.colorbar()
    plt.savefig(f'./chart_w_heatmap_{dataset_name}.png', format='png', bbox_inches='tight')
    plt.savefig(f'./chart_w_heatmap_{dataset_name}.eps', format='eps', bbox_inches='tight')
    plt.tight_layout()
    plt.show()
