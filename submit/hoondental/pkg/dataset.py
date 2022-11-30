import os
import sys
import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from shutil import copyfile

import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio as ta
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import random
import time


def read_questions(path_question):
    with open(path_question, 'r') as f:
        lines = f.readlines()
    meta = []
    for line in lines:
        idx, pname, question = line.strip().split('|')
        meta.append([int(idx), pname, question])
    return meta

def write_questions(meta, path_question):
    with open(path_question, 'w') as f:
        lines = []
        for _meta in meta:
            idx, pname, question = _meta
            lines.append('|'.join([str(idx), pname, question]))
        f.write('\n'.join(lines))

        
    
# Problem class 에서 문제 직접 생성

class ProblemDataset(Dataset):
    def __init__(self, problems, shuffle=True, batch_size=None, gec=None):
        self.problems = problems
        self.batch_size = batch_size
        self.multiple = 1 if batch_size is None else batch_size
        self.shuffle = shuffle
        self.gec = gec        
        self.idxes = list(range(len(problems) * self.multiple))
        if shuffle:
            np.random.shuffle(self.idxes)
        
    def __len__(self):
        return len(self.problems)
    
    def __getitem__(self, i):
        if self.batch_size is None:
            return self._getitem(self.idxes[i])
        else:
            idxes = self.idxes[i * self.batch_size:(i + 1) * self.batch_size]
            samples = []
            for i in idxes:
                samples.append(self._getitem(i))
            return self.collate(samples)
        
        
    def _getitem(self, i):
        j = self.idxes[i]
        k = j % len(self.problems)
        values, question = self.problems[k].random_question()
        pname = self.problems[k].__name__
        return k, pname, question
    
    def collate(self, samples):
        bsize = len(samples)
        idxes = []
        texts = []
        n_texts = []
        pnames = []
        for s in samples:
            idxes.append(s[0])
            pnames.append(s[1])
            texts.append(s[2])
            n_texts.append(len(s[2]))
        if self.gec is not None:
            texts = self.gec(texts)
            for i, t in enumerate(texts):
                n_texts[i] = len(t)
        return {'idx':idxes, 'text':texts, 'n_text':n_texts, 'pname':pnames}

        
# 생성되어 저장된 문제 데이터셋

class QuestionDataset(Dataset):
    def __init__(self, meta, vocab, shuffle=True, batch_size=None, 
                 add_bos=False, add_eos=False):
        self.meta = meta
        self.vocab = vocab
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.add_bos = add_bos
        self.add_eos = add_eos          
        
        self.pids = []
        self.pnames = []
        self.texts = []
        self.stexts = []
        self.itexts = []
        tk_bos = self.vocab.bos_token()
        tk_eos = self.vocab.eos_token()
        for pid, pname, question in self.meta:
            self.pids.append(pid)
            self.pnames.append(pname)
            self.texts.append(question)
            stext = self.vocab.encode_as_tokens(question)
            if self.add_bos and tk_bos:
                stext = [tk_bos] + stext
            if self.add_eos and tk_eos:
                stext = stext + [tk_eos]
            itext = self.vocab.tokens_to_ids(stext)
            self.stexts.append(stext)
            self.itexts.append(itext)            
        
        self.idxes = list(range(len(self.meta))) * (batch_size or 1)
        if shuffle:
            np.random.shuffle(self.idxes)                          
        
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, i):
        k = self.idxes[i]
        if self.batch_size is None:
            return {'idx':k, 'pid':self.pids[k], 'text':self.itexts[k], 'n_text':len(self.itexts[k])}
        else:
            idxes = self.idxes[i * self.batch_size:(i + 1) * self.batch_size]
            samples = []
            for k in idxes:
                samples.append({'idx':k, 'pid':self.pids[k], 'text':self.itexts[k], 'n_text':len(self.itexts[k])})
            return self.collate(samples)

    
    def collate(self, samples):
        bsize = len(samples)
        idxes = []
        pids = []
        texts = []
        n_texts = []
        for s in samples:
            idxes.append(s['idx'])
            pids.append(s['pid'])
            texts.append(s['text'])
            n_texts.append(s['n_text'])
        max_len = max(n_texts)
        texts = [text + [self.vocab.pad_id()] * (max_len - n_texts[i]) for i, text in enumerate(texts)]
        batch = {'idx': torch.tensor(idxes, dtype=torch.int64),
                 'pid': torch.tensor(pids, dtype=torch.int64),
                 'text': torch.tensor(texts, dtype=torch.int64), 
                 'n_text': torch.tensor(n_texts, dtype=torch.int32)}
        return batch
    
    
    def get_loader(self, batch_size=None, shuffle=True, num_workers=4):
        _collate = None if batch_size is None else self.collate
        _shuffle = None if batch_size is None else shuffle
        if batch_size is None and shuffle:
            random.shuffle(self.idxes)
        return DataLoader(self, batch_size, collate_fn=_collate, num_workers=num_workers, shuffle=_shuffle)  