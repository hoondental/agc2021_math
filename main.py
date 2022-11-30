import os
import re
import math
import time
import pickle
import json
import torch, torch.nn as nn
import numpy as np

from pkg.util import read_problemsheet, write_problemsheet, write_answersheet
from pkg.parse import *
from pkg.words import *
from problems import *
from pkg.solver import Solver

import sentencepiece as spm

from pkg.vocab import Vocab, CharVocab, SPVocab

Problems = [P1_1_1, P1_1_2, P1_1_3, P1_1_4, P1_1_5, P1_1_6, P1_1_7, P1_1_8, P1_1_9, P1_1_10, P1_1_11, P1_1_12, 
            P1_2_1, P1_2_2, P1_3_1, P1_4_1, 
            P2_1_1, P2_2_2, P2_3_1, 
            P3_1_1, P3_2_1, P3_2_2, P3_3_1, 
            P4_1_1, P4_2_1, P4_2_2, P4_3_1, 
            P5_1_1, P5_2_1, P5_3_1,
            P6_1_1, P6_3_1, P6_4_1,
            P7_1_1, P7_1_2, P7_3_1,
            P8_1_1, P8_2_1, P8_3_1, 
            P9_1_1, P9_2_1, P9_2_2, P9_3_1, P9_3_2]

path_problemsheet = '/home/agc2021/dataset/problemsheet.json'
path_samplesheet = 'sample.json'
path_answersheet = 'answersheet.json'

dir_trained = 'trained'
dir_question_classifier = os.path.join(dir_trained, 'question_classifier')
path_cfg = os.path.join(dir_question_classifier, 'cfg.pkl')
path_model = os.path.join(dir_question_classifier, 'trained.pth')

dir_tokenization = os.path.join(dir_trained, 'tokenization')
path_vocab_model = os.path.join(dir_tokenization, 'prob_512.model')

vocab = SPVocab(path_vocab_model)



with open(path_cfg, 'rb') as f:
    cfg = pickle.load(f)[0]
QC = cfg.create_object()    
state_dict = torch.load(path_model, map_location='cpu')['models'][0]
QC.load_state_dict(state_dict)


solver = Solver(QC, vocab, Problems, parse)

if os.path.exists(path_problemsheet):
    questions = read_problemsheet(path_problemsheet)
elif os.path.exists(path_samplesheet):
    questions = read_problemsheet(path_samplesheet)
else:
    raise Exception('Cannot find problemsheet!')
    

solutions = {}
for k, q in questions.items():
    answer, equation = solver.solve(q)
    solutions[k] = {"answer":answer, "equation":equation}
    
write_answersheet(solutions, path_answersheet)
