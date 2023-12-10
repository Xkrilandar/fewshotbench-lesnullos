import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from methods.meta_template import MetaTemplate
from qpth.qp import QPFunction
import cvxpy as cp
import wandb

# class DifferentiableSVM(nn.Module):
#     def __init__(self, num_features, num_classes):
#         super(DifferentiableSVM, self).__init__()
#         # Initialize weights and biases for the SVM
#         self.weights = nn.Parameter(torch.randn(num_classes, num_features))
#         self.bias = nn.Parameter(torch.randn(num_classes))

#     def forward(self, x):
#         # Linear decision function: Wx + b
#         return torch.matmul(x, self.weights.t()) + self.bias

#     def hinge_loss(self, outputs, labels):
#         # Implement hinge loss function for SVM
#         # Note: labels should be +1 or -1
#         hinge_loss = torch.mean(torch.clamp(1 - outputs.t() * labels, min=0))
#         return hinge_loss

#     def regularization_loss(self):
#         # L2 regularization loss (optional)
#         reg_loss = torch.norm(self.weights, p=2)
#         return reg_loss

class MetaOptNet(MetaTemplate):
    def __init__(self, backbone, n_way, n_support, num_classes, num_features):
        super(MetaOptNet, self).__init__(backbone, n_way, n_support)
        #self.classifier = DifferentiableSVM(num_classes=num_classes, num_features=num_features) 
        self.loss_fn = nn.CrossEntropyLoss()
        self.C_reg = 0.01


    def set_forward(self, x, y, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        y_support, y_query = self.parse_feature(y, True)
        # z_support = z_support.contiguous()
        # z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        # z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # scores_support = self.classifier(z_proto)
        # scores = self.classifier(z_query)
        # scores = -euclidean_dist(scores_query, scores_support)

        tasks_per_batch = z_query.size(0)
        n_support = z_support.size(1)
        n_query = z_query.size(1)

        assert(z_query.dim() == 3)
        assert(z_support.dim() == 3)
        assert(z_query.size(0) == z_support.size(0) and z_query.size(2) == z_support.size(2))
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
        kernel_matrix = computeGramMatrix(z_support, z_support)

        id_matrix_0 = torch.eye(self.n_way).expand(tasks_per_batch, self.n_way, self.n_way).cuda()
        block_kernel_matrix = batched_kronecker(kernel_matrix, id_matrix_0)
        #This seems to help avoid PSD error from the QP solver.
        block_kernel_matrix += 1.0 * torch.eye(self.n_way*n_support).expand(tasks_per_batch, self.n_way*n_support, self.n_way*n_support).cuda()
        original_labels = y_support.reshape(tasks_per_batch * n_support) # ??? OU PAS)
        label_mapping = {label: i for i, label in enumerate(set(torch.unique(original_labels).tolist()))}
        back_mapping = {i: label for i, label in enumerate(set(torch.unique(original_labels).tolist()))}
        support_labels = torch.tensor([label_mapping[label.item()] for label in original_labels]).to('cuda')
        support_labels_one_hot = one_hot(support_labels, self.n_way) # (tasks_per_batch * n_support, n_support)
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, self.n_way)
        support_labels_one_hot = support_labels_one_hot.reshape(tasks_per_batch, n_support * self.n_way)
        
        G = block_kernel_matrix
        e = -1.0 * support_labels_one_hot
        dummy = Variable(torch.Tensor()).cuda()      # We want to ignore the equality constraint.
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

        # Compute the classification score.
        compatibility = computeGramMatrix(z_support, z_query)
        compatibility = compatibility.float()
        compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_query, self.n_way)
        qp_sol = qp_sol.reshape(tasks_per_batch, n_support, self.n_way)
        logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, self.n_way)
        logits = logits * compatibility
        logits = torch.sum(logits, 1)

        # Reshape logits to the desired shape
        logits = logits.view(-1, self.n_way)

        return logits

    def set_forward_loss(self, x, y):
        _, y_query = self.parse_feature(y, True)
        scores = self.set_forward(x, y)
        y_query = y_query.reshape(-1)
        label_mapping = {label: i for i, label in enumerate(set(torch.unique(y_query).tolist()))}
        y_query = torch.tensor([label_mapping[label.item()] for label in y_query]).to('cuda')
        ret = self.loss_fn(scores, y_query)
        return ret
    
    def train_loop(self, epoch, train_loader, optimizer):  # overwrite parrent function
        print_freq = 10
        avg_loss = 0
        task_count = 0
        loss_all = []
        optimizer.zero_grad()

        # train
        for i, (x, y) in enumerate(train_loader):
            if isinstance(x, list):
                self.n_query = x[0].size(1) - self.n_support
                if self.change_way:
                    self.n_way = x[0].size(0)
                # assert self.n_way == x[0].size(
                #     0), f"MAML do not support way change, n_way is {self.n_way} but x.size(0) is {x.size(0)}"
            else:
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)  
                # assert self.n_way == x.size(
                #     0), f"MAML do not support way change, n_way is {self.n_way} but x.size(0) is {x.size(0)}"

            # Labels are assigned later if classification task
            # if self.type == "classification":
            #     y = None

            loss = self.set_forward_loss(x, y)
            # loss_all.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()

            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
                wandb.log({'loss/train': avg_loss / float(i + 1)})

    def test_loop(self, test_loader, record=None, return_std=False):
        correct = 0
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, y) in enumerate(test_loader):
            if isinstance(x, list):
                self.n_query = x[0].size(1) - self.n_support
                if self.change_way:
                    self.n_way = x[0].size(0)
            else: 
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)
            logits, y_query = self.set_forward(x, y)

            # smoothed_one_hot = one_hot(y_query.reshape(-1), self.n_way)
            # eps = 0
            # smoothed_one_hot = smoothed_one_hot * (1 - eps) + (1 - smoothed_one_hot) * eps / (self.n_way - 1)

            # log_prb = F.log_softmax(logits.reshape(-1, self.n_way), dim=1)
            # loss = -(smoothed_one_hot * log_prb).sum(dim=1)
            # loss = loss.mean()
            acc = self.count_accuracy(logits.reshape(-1, self.n_way), y_query.reshape(-1))

            # correct_this, count_this = self.correct(x, y)
            acc_all.append(acc.cpu())

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean
    





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

def solve_qp(Q, c, G, h, A, b, n_support):
    # Create a variable to optimize
    # x = cp.Variable(len(c))
    x = cp.Variable(n_support)

    # Define the objective function
    objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + x)

    # Define the constraints
    # constraints = [G @ x <= h, A @ x == b]
    constraints = [A @ x == b]

    # Define the problem and solve it
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return x.value

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
