import os
import re
import math
import time
import pickle
import json
import torch, torch.nn as nn
import numpy as np


class Solver:
    def __init__(self, qc, vocab, problems, parser, device='cpu'):
        self.qc = qc.eval().to(device)
        self.vocab = vocab
        self.problems = problems
        self.parser = parser
        self.device = device
    
    def normalize_text(self, text):
        text = text.replace('?', '')
        text = text.replace('  ', ' ')
        text = text.strip()
        return text   
   
    @torch.no_grad()
    def sort_problem_matching(self, question, normalize=True):
        if normalize:
            question = self.normalize_text(question)
        idx = self.vocab.encode_as_ids(question)
        idx = torch.tensor(idx, dtype=torch.int64).unsqueeze(0)
        logit = self.qc(idx.to(self.device)).squeeze(0).cpu().numpy()
        order = (-logit).argsort().tolist()
        return order
    
    def try_solve(self, question, porder, normalize=True):
        if normalize:
            question = self.normalize_text(question)
        for pid in porder:
            problem = self.problems[pid]
            objects, numbers, variables, formulas, equations, lists = self.parser(question)
            try:
                result = problem.try_solve(numbers, variables, formulas, equations, lists, question)
                if result is None:
                    continue
                else:
                    answer, equation = result
                    return answer, equation
            except:
                continue
        return '', ''
    
    def solve(self, question, normalize=True):
        porder = self.sort_problem_matching(question, normalize)
        answer, equation = self.try_solve(question, porder, normalize)
        return answer, equation