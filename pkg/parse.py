import os
import time
import re
import math
import json
from .words import *
from .util import read_problemsheet, write_problemsheet

from .problem import *


  

        
    
# 1점, 2점 -> 1, 2        
def remove_redundant_units(text, ws_unit):
    for unit in ws_unit:
        re_unit = unit + ','
        text = re.sub(re_unit, ',', text)
    return text

# 세 자리 -> 3 자리
def num_digits_to_number(text):
    for i, r in enumerate(re_num_digits):
        text = re.sub(r, str(i) + ' 자리', text)
    return text
    
    
def find_objects(text, re_object, object_type):
    iters = re.finditer(re_object, text)
    parts = []
    pos = 0
    for it in iters:
        start = it.start()
        end = it.end()
        obj = it.group()
        if start > pos:
            parts.append(text[pos:start].strip())
            pos = start
        parts.append(object_type(text[start:end]))
        pos = end
    if pos < len(text) - 1:
        parts.append(text[pos:].strip())
    return parts

def objectize_text(text):
    text = remove_redundant_units(text, ws_unit)
    text = num_digits_to_number(text)
    parts = find_objects(text, re_equation, Equation)
    _parts = []
    for p in parts:
        if isinstance(p, str):
            _parts.extend(find_objects(p, re_formula_1, Formula))
        else: 
            _parts.append(p)
    for k, _re in re_list.items():
        parts = _parts
        _parts = []
        for p in parts:
            if isinstance(p, str):
                _parts.extend(find_objects(p, _re, List))
            else:
                _parts.append(p)
    parts = _parts
    _parts = []
    for p in parts:
        if isinstance(p, str):
            _parts.extend(find_objects(p, re_variable, Variable))
        else: 
            _parts.append(p)
    parts = _parts
    _parts = []
    for p in parts:
        if isinstance(p, str):
            _parts.extend(find_objects(p, re_number, Number))
        else: 
            _parts.append(p)
    return _parts

            
def parse(text):
    parts = objectize_text(text)
    objects = [p for p in parts if not isinstance(p, str)]
    numbers = []
    variables = []
    formulas = []
    equations = []
    lists = []
    for obj in objects:
        if isinstance(obj, Number):
            numbers.append(obj)
        elif isinstance(obj, Variable):
            variables.append(obj)
        elif isinstance(obj, Equation):
            equations.append(obj)
        elif isinstance(obj, Formula):
            formulas.append(obj)
        elif isinstance(obj, List):
            lists.append(obj)
        else:
            raise Exception('Unsupported object type. ', type(obj))
    return objects, numbers, variables, formulas, equations, lists
    
    
    
    
            
        
        
