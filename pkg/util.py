import os
import re
import json


def read_problemsheet(path, normalize=False):
    with open(path, 'r') as f:
        data = json.load(f)
    problems = {}
    for k, v in data.items():
        q = v['question']
        if normalize:
            q = q.replace('?', '')
            q = q.replace('.', ' ')
            q = q.replace('  ', ' ')
            q = q.strip()
        problems[k] = q
    return problems


def write_problemsheet(problems, path):
    data = {}
    for k, v in problems.items():
        data[k] = {"question": v}
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
        
def write_answersheet(solutions, path):
    data = {}
    for k, solution in solutions.items():
        data[k] = {"answer":solution["answer"], "equation":solution["equation"]}
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
        
def blank(a):
    if isinstance(a, list) and not ' ' in a:
        return [''] + a
    else:
        return a
    
def list2str(a):
    b = [str(i) for i in a]
    return ', '.join(b)  




