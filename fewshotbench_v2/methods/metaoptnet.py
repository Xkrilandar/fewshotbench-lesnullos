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
    def set_forward(self, x, y_support, is_feature=False):
        #Parses the x features into query and support, also calls the backbone and embeds the features
        z_support, z_query = self.parse_feature(x, is_feature=False)

        tasks_per_batch = z_query.size(0)

        #reshapes the support slabels
        original_labels = y_support.reshape(tasks_per_batch * self.n_support)
        #maps labels to values between [0, n_way] so that we can call one_hot
        support_labels = torch.tensor(map_labels(original_labels, self.n_way))
        #converts support labels to one-hot encodings
        support_labels_one_hot = one_hot(support_labels, self.n_way).to('cuda')
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, self.n_support, self.n_way)
        support_labels_one_hot = support_labels_one_hot.reshape(tasks_per_batch, self.n_support * self.n_way)
        
        # Solve the quadratic programming problem for SVM
        qp_sol = self.qp_solve(support_labels_one_hot, z_support, self.n_support, tasks_per_batch)
        #compute the compatibility between the support and query features
        compatibility = computeGramMatrix(z_support, z_query)
        compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, self.n_support, self.n_query, self.n_way)
        #calculate the class logits with the compatibility and the quadratic solution
        logits = qp_sol * compatibility
        logits = torch.sum(logits, 1)

        logits = logits.view(-1, self.n_way)
        return logits

    #compute the loss for the forward pass
    def set_forward_loss(self, x, y):
        y_support, y_query = self.parse_feature(y, is_feature=True)
        #calculates the logits for the forward pass
        scores = self.set_forward(x,y_support) 
        y_query = y_query.reshape(-1)
        y_query = torch.tensor(map_labels(y_query, self.n_way)).to('cuda')
        #calls the loss function
        return self.loss_fn(scores, y_query)
    
    #same training loop as the parrent, just gives the labels to set_forward_loss
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
            else:
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)  


            loss = self.set_forward_loss(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()

            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
                wandb.log({'loss/train': avg_loss / float(i + 1)})


    # Compute the number of correct predictions in a batch
    def correct(self, x, y): # overwrite parrent function
        y_support, y_query = self.parse_feature(y, is_feature=True)
        #computes the class scores
        scores = self.set_forward(x, y_support) 
        y_query = y_query.reshape(-1)
        #maps labels to values between [0, n_way]
        y_query = map_labels(y_query, self.n_way)
        #outputs top labels
        _, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        #sums number of correct predicted labels
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        loss = self.loss_fn(scores, torch.tensor(y_query).to('cuda')).cpu().detach().numpy()
        return float(top1_correct), len(y_query), loss
    
    #same as parrent loop just gives the labels to the correct function
    def test_loop(self, test_loader, record=None, return_std=False): # overwrite parrent function
        correct = 0
        count = 0
        acc_all = []
        losses = []
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
            correct_this, count_this, loss = self.correct(x, y)
            acc_all.append(correct_this / count_this * 100)
            losses.append(loss)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        loss_mean = np.mean(np.asarray(losses))
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        print(f"mean test loss {loss_mean}")

        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean

    # Solve the quadratic programming problem for SVM
    def qp_solve(self, support_labels_one_hot, z_support, n_support, tasks_per_batch):
        #compute the kernel matrix
        kernel_matrix = computeGramMatrix(z_support, z_support)
         # Prepare the block matrix for the QP solver
        id_matrix_0 = torch.eye(self.n_way).expand(tasks_per_batch, self.n_way, self.n_way).cuda()
        block_kernel_matrix = batched_kronecker(kernel_matrix, id_matrix_0)
        block_kernel_matrix += 1.0 * torch.eye(self.n_way*n_support).expand(tasks_per_batch, self.n_way*n_support, self.n_way*n_support).cuda()
        #prepare the block matrix for the QP solver
        G = block_kernel_matrix #Qudratic term 
        e = -1.0 * support_labels_one_hot # Linear term
        #Inequality constraints
        id_matrix_1 = torch.eye(self.n_way * n_support).expand(tasks_per_batch, self.n_way * n_support, self.n_way * n_support)
        C = Variable(id_matrix_1)
        h = Variable(self.C_reg * support_labels_one_hot)
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
def computeGramMatrix(A, B):
    return torch.bmm(A, B.transpose(1,2)).float()

# Perform a batched Kronecker product between two matrices.
# The Kronecker product is a generalization of the outer product from vectors to matrices.
# This is used for constructing large block matrices in optimization problems, such as in QP for SVMs.
def batched_kronecker(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
    matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
    return torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1)).reshape([matrix1.size()[0]] + list(matrix1.size()[1:]) + list(matrix2.size()[1:])).permute([0, 1, 3, 2, 4]).reshape(matrix1.size(0), matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))


