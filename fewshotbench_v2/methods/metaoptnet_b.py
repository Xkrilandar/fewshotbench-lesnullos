import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from methods.meta_template import MetaTemplate
from utils.data_utils import one_hot
from qpth.qp import QPFunction
import wandb


class MetaOptNet(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(MetaOptNet, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.C_reg = 0.1

    # Forward pass for the model. Extracts features and computes class logits.
    def set_forward(self, x, is_feature=False):
        #Parses the x features into query and support, also calls the backbone and embeds the features
        z_support, z_query = self.parse_feature(x, is_feature)

        tasks_per_batch = z_query.size(0)

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = Variable(y_support.cuda())

        #reshapes the support slabels
        y_support = y_support.reshape(tasks_per_batch * self.n_support)
        #maps labels to values between [0, n_way] so that we can train on n way
        y_support = torch.tensor(map_labels(y_support, self.n_way))
        #converts support labels to one-hot encodings
        y_support_one_hot = one_hot(y_support, self.n_way).to('cuda')
        y_support_one_hot = y_support_one_hot.view(tasks_per_batch, self.n_support, self.n_way)
        y_support_one_hot = y_support_one_hot.reshape(tasks_per_batch, self.n_support * self.n_way)
        
        # Solve the quadratic programming problem for SVM
        qp_sol = self.qp_solve(y_support_one_hot, z_support, self.n_support, tasks_per_batch)
        #compute the compatibility between the support and query features
        compatibility = gram_matrix(z_support, z_query)
        compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, self.n_support, self.n_query, self.n_way)
        #calculate the class logits with the compatibility and the quadratic solution
        logits = qp_sol * compatibility
        logits = torch.sum(logits, 1)

        logits = logits.view(-1, self.n_way)
        return logits

    #compute the loss for the forward pass
    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())
        #calculates the logits for the forward pass
        scores = self.set_forward(x) 
        #calls the loss function
        
        return self.loss_fn(scores, y_query)
    
    # Solve the quadratic programming problem for SVM
    def qp_solve(self, y_support_one_hot, z_support, n_support, tasks_per_batch):
        #compute the kernel matrix
        kernel_matrix = gram_matrix(z_support, z_support)
         # Prepare the block matrix for the QP solver
        id_matrix_0 = torch.eye(self.n_way).expand(tasks_per_batch, self.n_way, self.n_way).cuda()
        block_kernel_matrix = batched_kronecker(kernel_matrix, id_matrix_0)
        block_kernel_matrix += 1.0 * torch.eye(self.n_way*n_support).expand(tasks_per_batch, self.n_way*n_support, self.n_way*n_support).cuda()
        #prepare the block matrix for the QP solver
        G = block_kernel_matrix #Qudratic term 
        e = -1.0 * y_support_one_hot # Linear term
        #Inequality constraints
        id_matrix_1 = torch.eye(self.n_way * n_support).expand(tasks_per_batch, self.n_way * n_support, self.n_way * n_support)
        C = Variable(id_matrix_1)
        h = Variable(self.C_reg * y_support_one_hot)
        #Equality constraints
        id_matrix_2 = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()
        A = Variable(batched_kronecker(id_matrix_2, torch.ones(tasks_per_batch, 1, self.n_way).cuda()))
        b = Variable(torch.zeros(tasks_per_batch, n_support))
        
        #Put the tensors on GPU
        G, e, C, h, A, b = [x.float().cuda() for x in [G, e, C, h, A, b]]
        #solve the QP problem
        maxIter = 5
        qp_sol = QPFunction(verbose=False, maxIter=maxIter)(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach())
        qp_sol = qp_sol.reshape(tasks_per_batch, self.n_support, self.n_way)
        return qp_sol.float().unsqueeze(2).expand(tasks_per_batch, self.n_support, self.n_query, self.n_way)
        
    


# Map labels from 0 to n_way
def map_labels(labels, n_way):
    unique_labels = torch.unique(labels)
    if len(unique_labels) > n_way:
        raise ValueError(f"Number of unique labels exceeds {n_way}")

    label_mapping = {label: i for i, label in enumerate(sorted(unique_labels.tolist()))}
    return [label_mapping[label.item()] for label in labels]


# Compute the Gram matrix, which is the matrix of dot products between all pairs of vectors from A and B.
def gram_matrix(A, B):
    return torch.bmm(A, B.transpose(1,2)).float()

# Perform a batched Kronecker product between two matrices.
# The Kronecker product is a generalization of the outer product from vectors to matrices.
# This is used for constructing large block matrices in optimization problems, such as in QP for SVMs.
def batched_kronecker(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
    matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
    return torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1)).reshape([matrix1.size()[0]] + list(matrix1.size()[1:]) + list(matrix2.size()[1:])).permute([0, 1, 3, 2, 4]).reshape(matrix1.size(0), matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))


