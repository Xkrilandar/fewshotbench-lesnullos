import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from methods.meta_template import MetaTemplate
from methods.helpers import solve_qp

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
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_reg = 50.0


    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)

        # z_support = z_support.contiguous()
        # z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        # z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # scores_support = self.classifier(z_proto)
        # scores = self.classifier(z_query)
        # scores = -euclidean_dist(scores_query, scores_support)
        print(x)
        print("Je fais des testssssssss")

        tasks_per_batch = z_query.size(0)
        n_support = z_support.size(1)
        n_query = z_query.size(1)

        assert(z_query.dim() == 3)
        assert(z_support.dim() == 3)
        assert(z_query.size(0) == z_support.size(0) and z_query.size(2) == z_support.size(2))
        assert(n_support == self.n_way * self.n_shot)      # n_support must equal to n_way * n_shot

        #Here we solve the dual problem:
        #Note that the classes are indexed by m & samples are indexed by i.
        #min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i

        #where w_m(\alpha) = \sum_i \alpha^m_i x_i,
        
        #\alpha is an (n_support, n_way) matrix
        kernel_matrix = computeGramMatrix(z_support, z_support)
        kernel_matrix += self.lambda_reg * torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()

        block_kernel_matrix = kernel_matrix.repeat(self.n_way, 1, 1) #(n_way * tasks_per_batch, n_support, n_support)
        
        support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), self.n_way) # (tasks_per_batch * n_support, n_way)
        support_labels_one_hot = support_labels_one_hot.transpose(0, 1) # (n_way, tasks_per_batch * n_support)
        support_labels_one_hot = support_labels_one_hot.reshape(self.n_way * tasks_per_batch, n_support)     # (n_way*tasks_per_batch, n_support)
        
        G = block_kernel_matrix
        e = -2.0 * support_labels_one_hot
        
        #This is a fake inequlity constraint as qpth does not support QP without an inequality constraint.
        id_matrix_1 = torch.zeros(tasks_per_batch*self.n_way, n_support, n_support)
        C = Variable(id_matrix_1)
        h = Variable(torch.zeros((tasks_per_batch*self.n_way, n_support)))
        dummy = Variable(torch.Tensor()).cuda()      # We want to ignore the equality constraint.

        
        G, e, C, h = [x.float().cuda() for x in [G, e, C, h]]

        # Solve the following QP to fit SVM:
        #        \hat z =   argmin_z 1/2 z^T G z + e^T z
        #                 subject to Cz <= h
        # We use detach() to prevent backpropagation to fixed variables.
        #qp_sol = QPFunction(verbose=False)(G, e.detach(), C.detach(), h.detach(), dummy.detach(), dummy.detach())
        qp_sol = solve_qp(G, e.detach(), C.detach(), h.detach(), dummy.detach(), dummy.detach())
        #qp_sol = QPFunction(verbose=False)(G, e.detach(), dummy.detach(), dummy.detach(), dummy.detach(), dummy.detach())

        #qp_sol (n_way*tasks_per_batch, n_support)
        qp_sol = qp_sol.reshape(self.n_way, tasks_per_batch, n_support)
        #qp_sol (n_way, tasks_per_batch, n_support)
        qp_sol = qp_sol.permute(1, 2, 0)
        #qp_sol (tasks_per_batch, n_support, n_way)
        
        # Compute the classification score.
        compatibility = computeGramMatrix(z_support, z_query)
        compatibility = compatibility.float()
        compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_query, self.n_way)
        qp_sol = qp_sol.reshape(tasks_per_batch, n_support, self.n_way)
        logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, self.n_way)
        logits = logits * compatibility
        logits = torch.sum(logits, 1)

        return logits

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

def computeGramMatrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.
    
    Parameters:
      A:  a (n_batch, n, d) Tensor.
      B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """
    
    assert(A.dim() == 3)
    assert(B.dim() == 3)
    assert(A.size(0) == B.size(0) and A.size(2) == B.size(2))

    return torch.bmm(A, B.transpose(1,2))


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies


