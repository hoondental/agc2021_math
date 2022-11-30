import os
import pickle
import numpy as np
import time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .models.util import to_onehot
from .trainer import Trainer, Hyper


    
class Trainer_QuestionClassifier(Trainer):
    def __init__(self, models, ds_trains, ds_evals=None, hp=Hyper, log_dir=None, optimizers=None, schedulers=None): 
        super().__init__(models, ds_trains, ds_evals, hp=hp, log_dir=log_dir, optimizers=optimizers, schedulers=schedulers)
        
        
    def get_loss(self, batch, loss_key=None, backward=False):
        hp = self.hp
        model = list(self.models.values())[0]            

        text = batch['text']
        n_text = batch['n_text']
        pid = batch['pid']
        idx = batch['idx']
        bsize = text.shape[0]
        pid_onehot = to_onehot(pid, num_cls=hp.num_problems)
        
        logit = model(text, n_text)
        log_prob = logit.log_softmax(dim=1)
        prob = logit.softmax(dim=1)
        
        loss_ce = -(pid_onehot * log_prob).sum() / bsize
        loss_total = loss_ce       
        
        pred = prob.argmax(dim=1)
        correct = (pid == pred)
        accuracy = correct.float().mean()
        
        losses, outputs, images = {}, {}, {}
        losses['total'] = loss_total
        losses['ce'] = loss_ce
        outputs['text'] = text
        outputs['n_text'] = n_text
        outputs['pid'] = pid
        outputs['idx'] = idx
        losses['accuracy'] = accuracy
        
        if backward:            
            loss_total.backward()
        return losses, outputs, images
            
    