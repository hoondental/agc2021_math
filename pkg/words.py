import os
import time
import re
import math
import json
import numpy as np


from jamo import h2j, j2hcj

from .util import *


_uchars = [i for i in range(0xac00, 0xd7a4)]  # 11171 characters
_uonset = [i for i in range(0x1100, 0x1113)]  
_unucleus = [i for i in range(0x1161, 0x1176)]
_ucoda = [i for i in range(0x11a8, 0x11c3)]

# 초성 종성 구별 없는 호환코드
_uconsonant = [i for i in range(0x3131, 0x314f)]
_uvowel = [i for i in range(0x314f, 0x3164)]

_chars = [chr(u) for u in _uchars]
_onset = [chr(u) for u in _uonset]
_nucleus = [chr(u) for u in _unucleus]
_coda = [chr(u) for u in _ucoda]
_consonant = [chr(u) for u in _uconsonant]
_vowel = [chr(u) for u in _uconsonant]

_bchars = [c.encode() for c in _chars]
_bonset = [c.encode() for c in _onset]
_bnucleus = [c.encode() for c in _nucleus]
_bcoda = [c.encode() for c in _coda]
_bconsonant = [c.encode() for c in _consonant]
_bvowel = [c.encode() for c in _vowel]



ws_person = ['남준', '석진', '윤기', '호석', '지민', '태형', '정국', '민영', '유정', '은지', '유나']
ws_gname = ['(가)', '(나)', '(다)', '(라)', '(마)', '(바)', '(사)', '(아)', '(자)', '(차)', '(카)', '(타)', '(파)', '(하)']
ws_variable = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
ws_family = ['아버지', '어머니', '자식', '아들', '할아버지', '할머니', '손자', '손녀', '형', '누나', '오빠', '언니', 
                     '동생', '남동생', '여동생', '삼촌', '이모', '고모', '조카']
ws_human = ws_person + ws_family + ['친구', '선생님', '학생', '아저씨', '아줌마', '아주머니', '이웃', '아이', '어른']
ws_location = ['서점', '마트', '문구점', '집', '학교', '수영장', '교실', '도서관', '독서실', '박물관', '운동장', '주차장', '정류장', 
              '아파트', '농장', '강당', '사무실', '병원']
ws_vehicle = ['자동차', '버스', '택시', '비행기', '트럭', '배', '자전거', '오토바이', '기차', '엘리베이터', '여객선', '화물차', '승용차']
ws_stationary = ['연필', '지우개', '색연필', '공책', '도화지', '색종이', '풀', '테이프']
ws_fruit = ['사과', '복숭아', '딸기', '자몽', '귤', '오렌지', '망고', '파인애플', '블루베리', '배', '감', '귤', '포도', '수박', '참외']
ws_vegetable = ['토마토', '무', '당근', '오이', '배추', '상추', '가지', '버섯']
ws_food = ws_fruit + ws_vegetable + ['사탕', '김밥', '빵', '라면', '과자', '음료수', '주스', '우유', '달걀', '계란', '떡']
ws_flower = ['장미', '백합', '튤립', '카네이션', '국화', '목련', '민들레', '벗꽃', '무궁화', '진달래', '철쭉']
ws_animal = ['오리', '닭', '토끼', '물고기', '고래', '거위', '달팽이', '개구리', '강아지', '개', '고양이', '비둘기', '병아리', '원숭이', '사자', 
            '호랑이', '늑대', '곰', '표범', '타조', '코끼리', '침팬지', '오랑우탄', '고릴라', '까치', '참새', '독수리', '매', '펭귄']
ws_unit_en = ['km', 'm', 'cm', 'mm', 'kg', 'g', 'ton', 'ml', 'liter']
ws_unit_kor = ['마리', '송이', '대', '그루', '권', '자루', '쪽', '살', '켤레', '줄', '벌', '타', '조각', '리터', '점', '점수', '가지']
ws_unit = ws_unit_en + ws_unit_kor
ws_body = ['손', '발', '손가락', '발가락', '팔', '다리', '몸', '머리']
ws_subject = ['국어', '영어', '수학', '사회', '과학', '음악', '미술', '체육']
ws_ball = ['공', '축구공', '배구공', '농구공', '야구공', '골프공', '당구공', '테니스볼']
ws_container = ['상자', '주머니', '바구니', '박스', '가방', '지갑', '창고', '컨테이너', '용기', '병', '유리병', '단지', '우리']
ws_sport = ['축구', '야구', '농구', '배구', '볼링', '당구', '골프', '배드민턴', '테니스', '탁구', '달리기', '멀리뛰기', '높이뛰기', '수영', '마라톤']
ws_color = ['흰색', '빨간색', '파랜색', '노란색', '녹색', '초록색', '보라색', '연두색', '주황색', '검정색', '회색', '자주색', '오렌지색', '분홍색']

ws_thing = ws_stationary + ws_ball + ws_vehicle
ws_object = ws_stationary + ws_ball
ws_object_wide = ws_object + ws_food + ws_flower + ws_animal

ws_a = ['한', '하나의', '한개의', '어떤']
ws_some = ['어떤']
ws_in = ['에', '에는', '안에', '안에는', '속에', '속에는']
ws_at = ['에', '에는']
ws_on = ['위', '위에', '위에는']
ws_to = ['에게', '에', '한테', '으로']
ws_and = ['와', '과']
ws_or = ['또는', '혹은', '이거나', '거나']
ws_be = ['이다', '입니다', '이에요', '이고', '인데', '이라고 할 때', '일 때', '이며']
ws_become = ['되다', '됩니다', '되었다', '되요', '되고', '되는데', '된다고 할 때', '될 때', '되며']
ws_be_same = ['같다', '같습니다', '똑같다', '똑 같다', '일치한다', '일치합니다', '같다고 할 때', '같을 때']
ws_bethere = ['있다', '있습니다', '있어요', '있었다', '있었습니다', '있었어요', '있으며', '있으니', '있고', '있을 때', '있는 데', '있다고 할 때']
ws_intotal = ['모두', '모두 해서', '더해서', '모두 더해서', '더하면', '다 더하면', '전부', '총', '다 합치면', '다 더하면']
ws_put_in = ['넣다', '넣었다', '넣어요', '넣었으요', '넣었습니다', '넣으니', '넣었더니', '넣어서']
ws_pull_out = ['빼다', '꺼내다', '뺐다', '꺼냈다', '뺐습니다', '꺼냈습니다', '빼냈다', '빼냈습니다', '빼고', '빼니', '빼서', '꺼내고', 
              '꺼내니', '꺼내서']
ws_except = ['빼고', '제외하고', '뺀', '제외한']
ws_more = ['더', '추가로']
ws_less = ['덜']
ws_inline = ['순서대로', '한 줄로', '일렬로', '차례로']
ws_dobetter = ['잘하다', '잘했습니다', '잘했고', '잘하였고', '잘했으며', '잘했지만', '낫다', '낫습니다', '나았고', '나았지만']
ws_doworse = ['못하다', '못했습니다', '못했고', '못하였고', '못했으며', '못했지만', '낮다', '낮습니다', '낮았고', '낮았지만']
ws_give = ['주다', '주었다', '줄 때', '주었을 때', '준다면', '주니']
ws_distribute = ['나누어' + g for g in ws_give] + ['나눠' + g for g in ws_give]
ws_ask = ['인가', '이냐', '입니까', '인가요', '일까요', '인지 구하시오', '인지 답하시오', '있습니까', '인지 쓰시오']
ws_ask_what = ['무엇' + ask for ask in ws_ask]
ws_how_many = ['몇', '몇 ', '얼마', '얼마 ']

def ask_how_many(unit=None):
    if unit is None:
        return np.random.choice(ws_how_many)  + np.random.choice(ws_ask)
    elif isinstance(unit, list) or isinstance(unit, tuple):
        return np.random.choice(ws_how_many) + np.random.choice(unit) + np.random.choice(ws_ask)
    elif isinstance(unit, str):
        return np.random.choice(ws_how_many) + unit + np.random.choice(ws_ask)
    else:
        raise Exception('Invalid unit. ', unit)
    

pos = {}
pos_jks_c = ['은', '이', '이가']
pos_jks_nc = ['는', '가']
pos['주격조사'] = pos_jks = pos_jks_c + pos_jks_nc
pos['목적격조사'] = pos_jko = ['을', '를', '이를']
pos['관형격조사'] = pos_jkg = []
pos['보격조사'] = pos_jkc = []
pos['부사격조사'] = pos_jkb = []
pos['호격조사'] = pos_jkv = []
pos['평서형종결어미'] = pos_efn = []
pos['의문형종결어미'] = pos_efq = ['얼마인가', '얼마인가요', '얼마일까', '얼마입니까', '무엇인가', '무엇인가요', '무엇일까', '무엇일까요', 
                           '몇 일까', '몇 인가요', '몇 일까요', '몇 개 인가', '몇 개 일까', '몇 개 일까요', '몇 개 입니까']
pos['명령형종결어미'] = pos_efo = ['구하시오', '구하세요']
pos['이다'] = pos_be = ['이다', '입니다', '이라고 할 때', '일 때', '이며']
pos['있다'] = pos_there = ['있다', '있습니다', '있어요', '있을 때', '있는 데', '있다고 할 때']
pos['수관형사'] = pos_mdn = ['영', '한', '두', '세', '네', '다섯', '여섯', '일곱', '여덟', '아홉', '열']


def ws2re(ws, add_space=True):
    _re = '|'.join(['(?:' + w + ')' for w in ws])
    if add_space:
        _re = '(?: *(?:' + _re + ') *)'
    else:
        _re = '(?:' + _re + ')'
    return _re

def ws_list2re(ws, add_space=True):
    _wsre = ws2re(ws, add_space)
    _re = '(?:' + '(?:' + _wsre  + ',)+' + _wsre  + ')'
    return _re



re_natural = '(?: *[0-9]+ *)'
re_float = '(?: *[0-9]+\.[0-9]+ *)'
re_fraction = '(?: *[0-9]+/[0-9]+ *)'
re_number = '(?:' + re_fraction + '|' + re_float + '|' + re_natural + ')'
re_variable = '(?: *[A-Z] *)'
re_alpha_numeric = '(?: *[0-9A-Z]+ *)'
re_formula_1 = '(?:' + '(?:' + re_alpha_numeric + '\+)+' + re_alpha_numeric + ')'
re_formula_0 = '(?:' + '(?:' + re_alpha_numeric + '\+)*' + re_alpha_numeric + ')'
re_equation = '(?:' + re_formula_0 + '=' + re_formula_0 + ')'

re_list = {}
re_list['number'] = '(?:' + '(?:' + re_number  + ',)+' + re_number  + ')'
re_list['alpha_numeric'] = '(?:' + '(?:' + re_alpha_numeric  + ',)+' + re_alpha_numeric  + ')'
re_list['person'] = ws_list2re(ws_person)
_ws_gname = ['\(가\)', '\(나\)', '\(다\)', '\(라\)', '\(마\)', '\(바\)', '\(사\)', '\(아\)', '\(자\)', '\(차\)', '\(카\)', '\(타\)', '\(파\)', '\(하\)']
re_list['gname'] = ws_list2re(_ws_gname)
re_list['variable'] = ws_list2re(ws_variable)
re_list['location'] = ws_list2re(ws_location)
re_list['vehicle'] = ws_list2re(ws_vehicle)
re_list['food'] = ws_list2re(ws_food)
re_list['flower'] = ws_list2re(ws_flower)
re_list['animal'] = ws_list2re(ws_animal)
re_list['subject'] = ws_list2re(ws_animal)
re_list['color'] = ws_list2re(ws_color)

re_any = {}
#re_any['number'] = '(?:' + '(?:' + re_number  + ',)+' + re_number  + ')'
#re_any['alpha_numeric'] = '(?:' + '(?:' + re_alpha_numeric  + ',)+' + re_alpha_numeric  + ')'
re_any['person'] = ws2re(ws_person)
re_any['gname'] = ws2re(_ws_gname)
re_any['variable'] = ws2re(ws_variable)
re_any['location'] = ws2re(ws_location)
re_any['vehicle'] = ws2re(ws_vehicle)
re_any['food'] = ws2re(ws_food)
re_any['flower'] = ws2re(ws_flower)
re_any['animal'] = ws2re(ws_animal)
re_any['subject'] = ws2re(ws_animal)
re_any['color'] = ws2re(ws_color)

re_num_digits = ['(?:' + r + ' ?자리)' for r in pos_mdn]



def to_number(text, strip=True):
    if strip:
        text = text.strip()
    m = re.match(re_natural, text)
    if m is not None and m.start() == 0 and m.end() == len(text):
        return int(text)
    m = re.match(re_float, text)
    if m is not None and m.start() == 0 and m.end() == len(text):
        return float(text)
    m = re.match(re_fraction, text)
    if m is not None and m.start() == 0 and m.end() == len(text):
        n, d = text.split('/')
        return float(n) / float(d)
    return None  


        
def make_sentence(*words, add_space=True):
    sentence = ''
    last_word = ''
    for i, _words in enumerate(words):
        if add_space:
            sentence += ' '
        if isinstance(_words, list) or isinstance(_words, tuple):
            if _words == pos_jks and len(last_word) > 0:
                _subwords = pos_jks_c if h2j(last_word)[-1] in _coda else pos_jks_nc
            else:
                _subwords = _words
            k = np.random.randint(len(_subwords))
            word = str(_subwords[k])
        else:
            word = str(_words)
        sentence += word
        last_word = word
    return ' '.join(sentence.split())



