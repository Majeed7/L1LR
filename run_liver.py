import timeit

import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression

from model_LR_NN_PR import LogisticRegressionNet

if __name__ == "__main__":
    dataset_name = "liver-disorders"
    train_path = f'./datasets//{dataset_name}'
    test_path = f'./datasets//{dataset_name}.t'

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


    # save results for draw ROC
    y_prob = logreg.predict_proba(test_set_x)
    y_score = model.predict_proba(x_test).detach().numpy()
    sio.savemat(f"./results/{dataset_name}_ours.mat", {'score_our': y_score, 'score_sk':y_prob[:, 1]})