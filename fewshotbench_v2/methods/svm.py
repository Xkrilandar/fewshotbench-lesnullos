import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from methods.meta_template import MetaTemplate

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import sys

# STANDARD SVM

# average acc for this method is 60 % 
# this method is not meta learning since the standard SVM is not differentiable
# this means that the model doesnt learn to adapt its embeddings for the method
# as we can see in the evaluation where the accuracy and loss are constant

class MetaOptNet(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(MetaOptNet, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()

        # putting the SVM init in the method init or in the forward function doesnt change the result
        # since the fit function overwrites the support vectors for the svm
        self.clf = make_pipeline(StandardScaler(), SVC(kernel='linear', C=0.5))


    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        y_support = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda())

        # to numpy since svm doesnt allow tensors
        z_support_cpu = z_support.cpu().detach().numpy()
        z_query_cpu = z_query.cpu().detach().numpy()
        y_support_cpu = y_support.cpu().detach().numpy()

        # fit the svm to this specific batch x of data
        self.clf.fit(z_support_cpu, y_support_cpu)
        
        # get the logits from the svm for the query
        scores = self.clf.decision_function(z_query_cpu)

        # get the logits back to tensor
        scores_torch = Variable(torch.from_numpy(scores).cuda(), requires_grad=True)
        return scores_torch


    def set_forward_loss(self, x):
        # creating labels for a few shot learning task
        y_query = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query)
