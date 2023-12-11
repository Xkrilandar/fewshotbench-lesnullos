import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from methods.meta_template import MetaTemplate

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class MetaOptNet(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(MetaOptNet, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.clf = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1))


    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        support_labels = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda())

        support_labels_cpu = support_labels.cpu().numpy()
        z_support_cpu = z_support.cpu().detach().numpy()
        z_query_cpu = z_query.cpu().detach().numpy()

        # print("support_labels", support_labels_cpu)
        # print("z_support", z_support_cpu)
        self.clf.fit(z_support_cpu, support_labels_cpu)
        
        scores = self.clf.decision_function(z_query_cpu)
        # print("scores", scores)
        scores_torch = Variable(torch.from_numpy(scores).cuda(), requires_grad=True)
        return scores_torch


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query )
