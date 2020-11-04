import torch
import torch.nn as nn

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
        loss = -torch.mean(y*torch.log(y_hat) + (1-y) * torch.log(1-y_hat))+lam* torch.norm(weight-Pw,1)
        # loss = torch.mean((y_hat-y)**2)+lam* torch.norm(weight-Pw,1)
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
        return MyLoss.apply(x, self.weight, self.bias, self.y, self.lam)
    
    def predict(self, x):
        w = self.weight
        a = x.mv(w) + self.bias
        return (torch.sigmoid(a) > 0.5).float()

    def predict_proba(self, x):
        w = self.weight
        a = x.mv(w) + self.bias
        return torch.sigmoid(a)