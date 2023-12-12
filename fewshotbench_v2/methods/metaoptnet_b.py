import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from methods.meta_template import MetaTemplate
from utils.data_utils import one_hot
from qpth.qp import QPFunction
import cvxpy as cp
import wandb


class MetaOptNet(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(MetaOptNet, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.C_reg = 0.1
        print(self.n_way)

    def set_forward(self, x, y_support, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature=False)

        tasks_per_batch = z_query.size(0)
        n_support = z_support.size(1)
        n_query = z_query.size(1)

        original_labels = y_support.reshape(tasks_per_batch * n_support) # ??? OU PAS)
        support_labels = torch.tensor(map_labels(original_labels))
        support_labels_one_hot = one_hot(support_labels, self.n_way).to('cuda') # (tasks_per_batch * n_support, n_support)
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, self.n_way)
        support_labels_one_hot = support_labels_one_hot.reshape(tasks_per_batch, n_support * self.n_way)
        
        qp_sol = self.qp_solve(support_labels_one_hot, z_support, n_support, tasks_per_batch)
        compatibility = computeGramMatrix(z_query, z_query)
        compatibility = compatibility.float()
        compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_query, n_query, self.n_way)
        qp_sol = qp_sol.reshape(tasks_per_batch, n_support, self.n_way)
        logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_query, n_query, self.n_way)
        logits = logits * compatibility
        logits = torch.sum(logits, 1)

        logits = logits.view(-1, self.n_way)
        return logits

    def set_forward_loss(self, x, y):
        y_support, y_query = self.parse_feature(y, is_feature=True)
        #qp_sol = solve_qp(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach(), n_support)
        scores = self.set_forward(x,y_support)
        #self.y_query = torch.tensor(y_query.reshape(-1).tolist()).to('cuda')
        y_query = y_query.reshape(-1)
        y_query = torch.tensor(map_labels(y_query)).to('cuda')
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
        y_support, y_query = self.parse_feature(y, is_feature=True)
        scores = self.set_forward(x, y_support)
        #y_query = np.repeat(range(self.n_way), self.n_query))
        y_query = y_query.reshape(-1)
        y_query = map_labels(y_query)
        _, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        loss = self.loss_fn(scores, torch.tensor(y_query).to('cuda')).cpu().detach().numpy()
        return float(top1_correct), len(y_query), loss
    
    def test_loop(self, test_loader, record=None, return_std=False):
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

    def qp_solve(self, support_labels_one_hot, z_support, n_support, tasks_per_batch):
        kernel_matrix = computeGramMatrix(z_support, z_support)

        id_matrix_0 = torch.eye(self.n_way).expand(tasks_per_batch, self.n_way, self.n_way).cuda()
        block_kernel_matrix = batched_kronecker(kernel_matrix, id_matrix_0)

        block_kernel_matrix += 1.0 * torch.eye(self.n_way*n_support).expand(tasks_per_batch, self.n_way*n_support, self.n_way*n_support).cuda()
        G = block_kernel_matrix
        e = -1.0 * support_labels_one_hot
        
        id_matrix_1 = torch.eye(self.n_way * n_support).expand(tasks_per_batch, self.n_way * n_support, self.n_way * n_support)
        C = Variable(id_matrix_1)
        h = Variable(self.C_reg * support_labels_one_hot)

        id_matrix_2 = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()

        A = Variable(batched_kronecker(id_matrix_2, torch.ones(tasks_per_batch, 1, self.n_way).cuda()))
        b = Variable(torch.zeros(tasks_per_batch, n_support))
        G, e, C, h, A, b = [x.float().cuda() for x in [G, e, C, h, A, b]]

        maxIter = 1
        return QPFunction(verbose=False, maxIter=maxIter)(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach())
    
def map_labels(labels):
    label_mapping = {label: i for i, label in enumerate(sorted(set(torch.unique(labels).tolist())))}
    return [label_mapping[label.item()] for label in labels]

def computeGramMatrix(A, B):
    return torch.bmm(A, B.transpose(1,2))

def batched_kronecker(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
    matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
    return torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1)).reshape([matrix1.size()[0]] + list(matrix1.size()[1:]) + list(matrix2.size()[1:])).permute([0, 1, 3, 2, 4]).reshape(matrix1.size(0), matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))


