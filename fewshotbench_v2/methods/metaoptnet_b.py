import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from methods.meta_template import MetaTemplate
import cvxpy as cp
import wandb
from qpth.qp import QPFunction
import torch.nn.functional as F
import sys

class MetaOptNet(MetaTemplate):
    def __init__(self, backbone, n_way, n_support, num_classes, num_features):
        super(MetaOptNet, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.C_reg = 0.01


    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)

        tasks_per_batch = z_query.size(0)
        n_support = z_support.size(1)
        n_query = z_query.size(1)

        # assert(z_query.dim() == 3)
        # assert(z_support.dim() == 3)
        # assert(z_query.size(0) == z_support.size(0) and z_query.size(2) == z_support.size(2))
        #assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot

        #Here we solve the dual problem:
        #Note that the classes are indexed by m & samples are indexed by i.
        #min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
        #s.t.  \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i

        #where w_m(\alpha) = \sum_i \alpha^m_i x_i,
        #and C^m_i = C if m  = y_i,
        #C^m_i = 0 if m != y_i.
        #This borrows the notation of liblinear.
        
        #\alpha is an (n_support, n_way) matrix
        # z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        # z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        print(z_support.size()) # 25 64
        print(z_query.size()) # 75 64

        # kernel_matrix = torch.bmm(z_support.unsqueeze(1), z_support.unsqueeze(2)).squeeze()
        
        kernel_matrix = computeGramMatrix(z_support, z_support)
        print(kernel_matrix.size()) # 25


        

        id_matrix_0 = torch.eye(self.n_way).expand(self.n_way, self.n_way).cuda()
        print(id_matrix_0.size()) # 5 5 5 

        block_kernel_matrix = batched_kronecker(kernel_matrix, id_matrix_0)
        print(block_kernel_matrix.size()) # 
        #This seems to help avoid PSD error from the QP solver.
        #block_kernel_matrix += 1.0 * torch.eye(self.n_way*n_support).expand(tasks_per_batch, self.n_way*n_support, self.n_way*n_support).cuda()
        
        #print("y_support", y_support.size())
        support_labels = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda())

        # original_labels = y_support.reshape(tasks_per_batch * n_support) # ??? OU PAS)
        #print("original_labels", original_labels.size())
        #print("y_query", y_query.size())
        # y_query = y_query.reshape(tasks_per_batch * n_query)
        #print("y_query", y_query.size())

        # label_mapping = {label: i for i, label in enumerate(set(torch.unique(original_labels).tolist()))}
        # support_labels = torch.tensor([label_mapping[label.item()] for label in original_labels]).to('cuda')
        # query_labels = torch.tensor([label_mapping[label.item()] for label in y_query]).to('cuda')
        support_labels_one_hot = one_hot(support_labels, self.n_way) # (tasks_per_batch * n_support, n_support)
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, self.n_way)
        support_labels_one_hot = support_labels_one_hot.reshape(tasks_per_batch, n_support * self.n_way)
        

        G = block_kernel_matrix
        e = -1.0 * support_labels_one_hot

        #print (G.size())
        #This part is for the inequality constraints:
        #\alpha^m_i <= C^m_i \forall m,i
        #where C^m_i = C if m  = y_i,
        #C^m_i = 0 if m != y_i.
        id_matrix_1 = torch.eye(self.n_way * n_support).expand(tasks_per_batch, self.n_way * n_support, self.n_way * n_support)

        
        C = Variable(id_matrix_1)
        h = Variable(self.C_reg * support_labels_one_hot)

        #print (C.size(), h.size())
        #This part is for the equality constraints:
        #\sum_m \alpha^m_i=0 \forall i
        id_matrix_2 = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()

        A = Variable(batched_kronecker(id_matrix_2, torch.ones(tasks_per_batch, 1, self.n_way).cuda()))
        b = Variable(torch.zeros(tasks_per_batch, n_support))
        #print (A.size(), b.size())
        G, e, C, h, A, b = [x.float().cuda() for x in [G, e, C, h, A, b]]

        # Solve the following QP to fit SVM:
        #        \hat z =   argmin_z 1/2 z^T G z + e^T z
        #                 subject to Cz <= h
        # We use detach() to prevent backpropagation to fixed variables.
        maxIter = 15
        qp_sol = QPFunction(verbose=False, maxIter=maxIter)(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach())
        #qp_sol = solve_qp(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach(), n_support)
        print("qp_sol", qp_sol, qp_sol.size())
        # Compute the classification score.
        print("z_query", z_query)

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
        print(scores.size()) # this should be 75 5, is 5 15 5 currently
        sys.exit()
        return self.loss_fn(scores, y_query)
    
    


    





def computeGramMatrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.
    
    Parameters:
      A:  a (n_batch, n, d) Tensor.
      B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """
    
    # assert(A.dim() == 3)
    # assert(B.dim() == 3)
    # assert(A.size(0) == B.size(0) and A.size(2) == B.size(2))

    return torch.bmm(A, B.transpose(1,2))


def batched_kronecker(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
    matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
    return torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1)).reshape([matrix1.size()[0]] + list(matrix1.size()[1:]) + list(matrix2.size()[1:])).permute([0, 1, 3, 2, 4]).reshape(matrix1.size(0), matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))


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