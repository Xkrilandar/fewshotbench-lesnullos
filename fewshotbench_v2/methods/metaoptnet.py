import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.autograd import Variable

from methods.meta_template import MetaTemplate

class MetaOptNet(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(MetaOptNet, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()

    # def set_forward(self, x, is_feature=False):
        # z_support, z_query = self.parse_feature(x, is_feature)

        # z_support = z_support.contiguous()
        # z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        # z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # dists = euclidean_dist(z_query, z_proto)
        # scores = -dists
        # return scores


    # def set_forward_loss(self, x):
    #     y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
    #     y_query = Variable(y_query.cuda())

    #     scores = self.set_forward(x)

    #     return self.loss_fn(scores, y_query )
