import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from methods.meta_template import MetaTemplate
from qpth.qp import QPFunction
import wandb
import sys
from utils.data_utils import one_hot


class DifferentiableSVM(nn.Module):
    def __init__(self, num_classes, num_features):
        super(DifferentiableSVM, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features

        # Initialize SVM parameters (weights and biases)
        self.weights = nn.Parameter(torch.randn(num_classes, num_features))
        self.bias = nn.Parameter(torch.zeros(num_classes))

class MetaOptNet(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(MetaOptNet, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.C_reg = 0.01
        self.classifier = DifferentiableSVM(num_classes=n_way, num_features=64)


    def set_forward(self, x, y, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        
        
        logits = ???

        return logits

    def set_forward_loss(self, x, y):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        logits = self.set_forward(x, y)
        
        ret = self.loss_fn(logits, y_query)
        return ret
    
    def train_loop(self, epoch, train_loader, optimizer):  # overwrite parrent function
        print_freq = 10
        avg_loss = 0
        task_count = 0
        loss_all = []
        optimizer.zero_grad()

        # train
        for i, (x, y) in enumerate(train_loader):
            print("x0", x.size(1))
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

    def correct(self, x, y):
        scores = self.set_forward(x, y)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        # print("topk_inddddddd", topk_ind[:, 0])
        # print("y_queryyyyyyyyyyy", y_query)
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        #print("y_query", y_query, "topk_labels", topk_labels)
        
        return float(top1_correct), len(y_query)
    
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
            correct_this, count_this = self.correct(x, y)
            acc_all.append(correct_this / count_this * 100)

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


# def one_hot(indices, depth):
#     """
#     Returns a one-hot tensor.
#     This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
#     Parameters:
#       indices:  a (n_batch, m) Tensor or (m) Tensor.
#       depth: a scalar. Represents the depth of the one hot dimension.
#     Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
#     """

#     encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
#     index = indices.view(indices.size()+torch.Size([1]))
#     encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
#     return encoded_indicies
