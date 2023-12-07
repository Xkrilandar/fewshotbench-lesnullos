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

        # Directly use the support and query features in SVM
        scores = self.classifier(z_support, z_query)
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)
        return self.loss_fn(scores, y_query)
