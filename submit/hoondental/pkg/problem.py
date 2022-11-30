import os
import time
import re
import math
import numpy as np

from io import StringIO
from contextlib import redirect_stdout

from .words import *

class Object:
    def __init__(self, text):
        self.text = text.strip()
        self._value = None
        
    def value(self):
        return self._value or self.text
    
    def __str__(self):
        return str(self.value())
    
    def __repr__(self):
        return str(self.value())
    
class Number(Object):
    def __init__(self, text):
        super().__init__(text)
        self._value = to_number(text)  
        assert self._value is not None
        
    
class Variable(Object):
    def __init__(self, text):
        text = text.strip()
        assert text in list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        super().__init__(text)
        self._value = None

    
class Formula(Object):
    def __init__(self, text):
        super().__init__(text)
                
    def value(self, **kwargs):
        formula = self.text
        for k, v in kwargs.items():
            formula = formula.replace(k, str(v))
        return eval(formula)
    
    
class Equation(Formula):
    def __init__(self, text):
        text = re.sub('=+', '==', text)
        assert '==' in text
        super().__init__(text)
        

class List(Object):
    def __init__(self, text):
        super().__init__(text)
        parts = [x.strip() for x in self.text.split(',')]
        self.list = []
        for x in parts:
            if x in list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
                self.list.append(Variable(x))
            elif to_number(x) is not None:
                self.list.append(Number(x))
            else:
                self.list.append(Object(x))

    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, i):
        return self.list[i]
    
    def tolist(self):
        return [v.value() for v in self.list]
        


class Problem:
    @classmethod
    def description(cls):
        return '수학문제'
    
    @classmethod
    def gen_question(cls, *values):
        raise NotImplementedError
    
    @classmethod
    def equation(cls, *values):
        raise NotImplementedError

    @classmethod
    def random_values(cls):
        raise NotImplementedError
        
    @classmethod
    def random_question(cls):
        values = cls.random_values()
        question = cls.gen_question(*values)
        return values, question
        
    @classmethod
    def solve(cls, *values):
        equation = cls.equation(*values).strip()
        f = StringIO()
        with redirect_stdout(f):
            exec(equation)
        answer = f.getvalue().strip()
        return answer, equation
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        return '', ''
        
    

    
