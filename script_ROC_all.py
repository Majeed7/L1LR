import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import auc, roc_curve

plt.rcParams['figure.dpi'] = 300
plt.rcParams["font.family"] = "Sans Serif"
plt.rcParams['font.size'] = 24
mpl.rcParams['xtick.labelsize'] = 14 
mpl.rcParams['ytick.labelsize'] = 14 

if __name__ == "__main__":
    method_names = ['Gauss-Seidel','Shooting','Gauss-Southwell','Grafting','SubGradient','Max-K SubGradient','epsL1','Log-Barrier','SmoothL1 (short-cut)','SmoothL1 (continuation)','EM','SQP','ProjectionL1','InteriorPoint','Orthant-Wise','Pattern-Search','Projected SubGradient', 'sklearn', 'Proposed Method']
    selected_methods = ['Gauss-Seidel','Shooting','Log-Barrier','ProjectionL1','InteriorPoint', 'sklearn', 'Proposed Method']
    ## matlab results
    score_a1a  = 1.0 - sio.loadmat("./results/a1a.mat")["score"]
    score_a9a  = 1.0 - sio.loadmat("./results/a9a.mat")["score"]
    score_splice  = 1.0 - sio.loadmat("./results/splice.mat")["score"]
    score_ijcnn1  = 1.0 - sio.loadmat("./results/ijcnn1.mat")["score"]
    score_liver  = 1.0 - sio.loadmat("./results/liver-disorders.mat")["score"]
    score_madelon  = 1.0 - sio.loadmat("./results/madelon.mat")["score"]

    print(score_a1a.shape)
    print(score_a9a.shape)
    print(score_splice.shape)
    print(score_ijcnn1.shape)
    print(score_liver.shape)
    print(score_madelon.shape)

    ## add our python results
    score1  = sio.loadmat("./results/ijcnn1_ours.mat")["score_sk"]
    score2  = sio.loadmat("./results/ijcnn1_ours.mat")["score_our"]
    score_ijcnn1 = np.vstack([score_ijcnn1, score1, score2])
    print(score_ijcnn1.shape)

    score1  = sio.loadmat("./results/a1a_ours.mat")["score_sk"]
    score2  = sio.loadmat("./results/a1a_ours.mat")["score_our"]
    score_a1a = np.vstack([score_a1a, score1, score2])
    print(score_ijcnn1.shape)
    
    score1  = sio.loadmat("./results/a9a_ours.mat")["score_sk"]
    score2  = sio.loadmat("./results/a9a_ours.mat")["score_our"]
    score_a9a = np.vstack([score_a9a, score1, score2])
    print(score_a9a.shape)
    
    score1  = sio.loadmat("./results/madelon_ours.mat")["score_sk"]
    score2  = sio.loadmat("./results/madelon_ours.mat")["score_our"]
    score_madelon = np.vstack([score_madelon, score1, score2])
    print(score_madelon.shape)

    score1  = sio.loadmat("./results/splice_ours.mat")["score_sk"]
    score2  = sio.loadmat("./results/splice_ours.mat")["score_our"]
    score_splice = np.vstack([score_splice, score1, score2])
    print(score_splice.shape)

    score1  = sio.loadmat("./results/liver-disorders_ours.mat")["score_sk"]
    score2  = sio.loadmat("./results/liver-disorders_ours.mat")["score_our"]
    score_liver = np.vstack([score_liver, score1, score2])
    print(score_liver.shape)

    
    fig, ax = plt.subplots(2, 3,sharex=True,sharey=True,figsize=(24, 16))
    fig.tight_layout()
    lw=3.0
    line_styles = ['-.','--','-',':']
    _, ytrue = load_svmlight_file('./datasets/a9a.t')
    ytrue[ytrue==-1] = 0

    for k, name in enumerate(method_names):
        if name not in selected_methods:
            continue
        fpr, tpr, _ = roc_curve(ytrue, score_a9a[k])
        roc_auc = auc(fpr, tpr)
        ax[0,0].plot(fpr, tpr, lw=lw, label=f'{name} (area = {roc_auc:.3f})', linestyle=line_styles[k % len(line_styles)])
    ax[0,0].set_title('a9a')
    ax[0,0].legend(loc="lower right", prop={'size': 16})
    ax[0,0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    _, ytrue = load_svmlight_file('./datasets/a1a.t')
    ytrue[ytrue==-1] = 0
    for k, name in enumerate(method_names):
        if name not in selected_methods:
            continue
        fpr, tpr, _ = roc_curve(ytrue, score_a1a[k])
        roc_auc = auc(fpr, tpr)
        ax[0,1].plot(fpr, tpr, lw=lw, label=f'{name} (area = {roc_auc:.3f})', linestyle=line_styles[k % len(line_styles)])
    ax[0,1].set_title('a1a')
    ax[0,1].legend(loc="lower right", prop={'size': 16})
    ax[0,1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    _, ytrue = load_svmlight_file('./datasets/madelon.t')
    ytrue[ytrue==-1] = 0
    for k, name in enumerate(method_names):
        if name not in selected_methods:
            continue
        fpr, tpr, _ = roc_curve(ytrue, score_madelon[k])
        roc_auc = auc(fpr, tpr)
        ax[0,2].plot(fpr, tpr, lw=lw, label=f'{name} (area = {roc_auc:.3f})', linestyle=line_styles[k % len(line_styles)])
    ax[0,2].set_title('madelon')
    ax[0,2].legend(loc="lower right", prop={'size': 16})
    ax[0,2].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    _, ytrue = load_svmlight_file('./datasets/splice.t')
    ytrue[ytrue==-1] = 0
    for k, name in enumerate(method_names):
        if name not in selected_methods:
            continue
        fpr, tpr, _ = roc_curve(ytrue, score_splice[k])
        roc_auc = auc(fpr, tpr)
        ax[1,0].plot(fpr, tpr, lw=lw, label=f'{name} (area = {roc_auc:.3f})', linestyle=line_styles[k % len(line_styles)])
    ax[1,0].set_title('splice')
    ax[1,0].legend(loc="lower right", prop={'size': 16})
    ax[1,0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    _, ytrue = load_svmlight_file('./datasets/ijcnn1.t')
    ytrue[ytrue==-1] = 0
    for k, name in enumerate(method_names):
        if name not in selected_methods:
            continue
        fpr, tpr, _ = roc_curve(ytrue, score_ijcnn1[k])
        roc_auc = auc(fpr, tpr)
        ax[1,1].plot(fpr, tpr, lw=lw, label=f'{name} (area = {roc_auc:.3f})', linestyle=line_styles[k % len(line_styles)])
    ax[1,1].set_title('ijcnn1')
    ax[1,1].legend(loc="lower right", prop={'size': 16})
    ax[1,1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    _, ytrue = load_svmlight_file('./datasets/liver-disorders.t')
    ytrue[ytrue==-1] = 0
    for k, name in enumerate(method_names):
        if name not in selected_methods:
            continue
        fpr, tpr, _ = roc_curve(ytrue, score_liver[k])
        roc_auc = auc(fpr, tpr)
        ax[1,2].plot(fpr, tpr, lw=lw, label=f'{name} (area = {roc_auc:.3f})', linestyle=line_styles[k % len(line_styles)])
    ax[1,2].set_title('liver-disorders')
    ax[1,2].legend(loc="lower right", prop={'size': 16})
    ax[1,2].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    
    fig.text(0.5, 0.001, 'False Positive Rate', ha='center',fontsize=28)
    fig.text(0.001, 0.5, 'True Positive Rate', va='center', rotation='vertical',fontsize=28)
    plt.setp(ax, xlim=[0.0, 1.0], ylim=[0.0, 1.05])
    plt.savefig(f'./Chart_ROC_all.png', format='png', bbox_inches='tight')
    plt.savefig(f'./Chart_ROC_all.eps', format='eps', bbox_inches='tight')
    plt.show()



