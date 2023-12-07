import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from methods.meta_template import MetaTemplate


class DifferentiableSVM(nn.Module):
    def __init__(self, num_features, num_classes):
        super(DifferentiableSVM, self).__init__()
        # Initialize weights and biases for the SVM
        self.weights = nn.Parameter(torch.randn(num_classes, num_features))
        self.bias = nn.Parameter(torch.randn(num_classes))

    def forward(self, x):
        # Linear decision function: Wx + b
        return torch.matmul(x, self.weights.t()) + self.bias

    def hinge_loss(self, outputs, labels):
        # Implement hinge loss function for SVM
        # Note: labels should be +1 or -1
        hinge_loss = torch.mean(torch.clamp(1 - outputs.t() * labels, min=0))
        return hinge_loss

    def regularization_loss(self):
        # L2 regularization loss (optional)
        reg_loss = torch.norm(self.weights, p=2)
        return reg_loss

class MetaOptNet(MetaTemplate):
    def __init__(self, backbone, n_way, n_support, num_classes, num_features):
        super(MetaOptNet, self).__init__(backbone, n_way, n_support)
        self.classifier = DifferentiableSVM(num_classes=num_classes, num_features=num_features) 

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        scores_support = self.classifier(z_proto)
        scores_query = self.classifier(z_query)
        scores = -euclidean_dist(scores_query, scores_support)
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)
        return self.loss_fn(scores, y_query)
    

def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
