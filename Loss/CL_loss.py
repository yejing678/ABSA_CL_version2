import math
import os, sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import numpy as np
from Loss.TripletLoss import HardTripletLoss

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定


class CL_loss_arrange(nn.Module):
    def __init__(self, opt, margin=0.2, hardest=False, squared=False):
        super(CL_loss_arrange, self).__init__()
        self.opt = opt
        self.margin = margin
        self.hardest = hardest
        self.squared = squared
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.tl_loss = HardTripletLoss(opt=self.opt, margin=self.margin, hardest=self.hardest)

    def forward(self, logic, features, sentiment_labels, implicit_labels, i_epoch, stages):
        """
        t: i_epoch
        T: stage num
        """

        CE_Loss = self.ce_loss(logic, sentiment_labels)
        if have_different_sample(sentiment_labels):
            features = F.normalize(features, p=2, dim=1)
            TL_Loss = self.tl_loss(features, sentiment_labels)
        else:
            TL_Loss = 0

        implicit_weight = min(1, 1e-5 + (i_epoch/stages))
        explicit_weight = max(1, 2-(i_epoch/stages))
        # ones = torch.ones_like(implicit_labels).to(self.opt.device)
        sample_weight = implicit_labels * implicit_weight + (1-implicit_labels) * explicit_weight
        CE_Loss = (CE_Loss * sample_weight / sample_weight.sum()).sum()
        loss = CE_Loss + TL_Loss
        print('sample_weight:{}\n,ce_loss:{}, tl_loss:{}'.format(sample_weight, CE_Loss, TL_Loss))
        return loss


def have_different_sample(targets):
    flag = 0
    target1 = targets[0]
    for i in range(len(targets)):
        a = targets[i]
        if a != target1:
            flag = 1
            break
    return flag
