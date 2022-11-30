import time
import re
import math

from pkg.problem import *
from pkg.words import *



# ============================================ 유형 1 =================================================
#        *********************************** type 1 - 1 *******************************
class P1_1(Problem):
    @classmethod
    def description(cls):
        return '두 수 사이의 수의 홀수, 짝수, 모든수의 합'

    @classmethod
    def random_values(cls):
        x, y = np.random.choice(list(range(1000)), size=2, replace=False)
        return min(x, y), max(x, y)    
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn != 2 or nv + nf + ne + nl > 0:
            return None
        else:
            n1, n2 = numbers[0].value(), numbers[1].value()
            vmin, vmax = min(n1, n2), max(n1, n2)
            return cls.solve(vmin, vmax)

# ----------------------------------------------------- 
class P1_1_1(P1_1):
    @classmethod
    def description(cls):
        return '범위가 주어진 홀수의 합, 경계 포함'
    
    @classmethod
    def gen_question(cls, vmin, vmax):
        return make_sentence(vmin, ['부터', '에서', '에서 부터', '보다 크거나 같고', '보다 많거나 같고'], vmax, 
                             ['까지', '까지의', '보다 작거나 같은', '보다 적거나 같은'], 
                             ['', '수 중', '수 중에서'], ['홀수', '홀수의', '홀수를'], ['합', '총합', '총 합', '더하면'], pos_jks, pos_efq+pos_efo)
    
    @classmethod
    def equation(cls, vmin, vmax):
        code = f'''
sum = 0
for i in range({vmin}, {vmax} + 1):
    if i % 2 == 1:
        sum += i
print(sum)
        '''
        return code       
    
        
# -----------------------------------------------------         
class P1_1_2(P1_1):
    @classmethod
    def description(cls):
        return '범위가 주어진 홀수의 합, 오른쪽 경계 포함'
    
    @classmethod
    def gen_question(cls, vmin, vmax):
        return make_sentence(vmin, ['보다 크고', '보다 많고'], vmax, ['까지', '까지의', '보다 작거나 같은', '보다 적거나 같은'], 
                             ['', '수 중', '수 중에서'], ['홀수', '홀수의', '홀수를'], ['합', '총합', '총 합', '더하면'], pos_jks, pos_efq+pos_efo)
    
    @classmethod
    def equation(cls, vmin, vmax):
        code = f'''
sum = 0
for i in range({vmin + 1}, {vmax} + 1):
    if i % 2 == 1:
        sum += i
print(sum)
        '''
        return code

# ----------------------------------------------------- 
class P1_1_3(P1_1):
    @classmethod
    def description(cls):
        return '범위가 주어진 홀수의 합, 왼쪽 경계 포함'
    
    @classmethod
    def gen_question(cls, vmin, vmax):
        return make_sentence(vmin, ['부터', '에서', '에서 부터', '보다 크거나 같고', '보다 많거나 같고'], vmax, ['보다 작은', '보다 적은'], 
                             ['', '수 중', '수 중에서'], ['홀수', '홀수의', '홀수를'], ['합', '총합', '총 합', '더하면'], pos_jks, pos_efq+pos_efo)
    
    @classmethod
    def equation(cls, vmin, vmax):
        code = f'''
sum = 0
for i in range({vmin}, {vmax}):
    if i % 2 == 1:
        sum += i
print(sum)
        '''
        return code
        
# -----------------------------------------------------         
class P1_1_4(P1_1):
    @classmethod
    def description(cls):
        return '범위가 주어진 홀수의 합, 경계 불포함'
    
    @classmethod
    def gen_question(cls, vmin, vmax):
        return make_sentence(vmin, ['보다 크고', '보다 많고'], vmax, ['보다 작은', '보다 적은'], 
                             ['', '수 중', '수 중에서'], ['홀수', '홀수의', '홀수를'], ['합', '총합', '총 합', '더하면'], pos_jks, pos_efq+pos_efo)
    
    @classmethod
    def equation(cls, vmin, vmax):
        code = f'''
sum = 0
for i in range({vmin + 1}, {vmax}):
    if i % 2 == 1:
        sum += i
print(sum)
        '''
        return code

# -----------------------------------------------------     
class P1_1_5(P1_1):
    @classmethod
    def description(cls):
        return '범위가 주어진 짝수의 합, 경계 포함'
    
    @classmethod
    def gen_question(cls, vmin, vmax):
        return make_sentence(vmin, ['부터', '에서', '에서 부터', '보다 크거나 같고', '보다 많거나 같고'], vmax, 
                             ['까지', '까지의', '보다 작거나 같은', '보다 적거나 같은'], 
                             ['', '수 중', '수 중에서'], ['짝수', '짝수의', '짝수를'], ['합', '총합', '총 합', '더하면'], pos_jks, pos_efq+pos_efo)
    
    @classmethod
    def equation(cls, vmin, vmax):
        code = f'''
sum = 0
for i in range({vmin}, {vmax} + 1):
    if i % 2 == 0:
        sum += i
print(sum)
        '''
        return code
        
# -----------------------------------------------------         
class P1_1_6(P1_1):
    @classmethod
    def description(cls):
        return '범위가 주어진 짝수의 합, 오른쪽 경계 포함'
    
    @classmethod
    def gen_question(cls, vmin, vmax):
        return make_sentence(vmin, ['보다 크고', '보다 많고'], vmax, ['까지', '까지의', '보다 작거나 같은', '보다 적거나 같은'], 
                             ['', '수 중', '수 중에서'], ['짝수', '짝수의', '짝수를'], ['합', '총합', '총 합', '더하면'], pos_jks, pos_efq+pos_efo)
    
    @classmethod
    def equation(cls, vmin, vmax):
        code = f'''
sum = 0
for i in range({vmin} + 1, {vmax} + 1):
    if i % 2 == 0:
        sum += i
print(sum)
        '''
        return code

# ----------------------------------------------------- 
class P1_1_7(P1_1):
    @classmethod
    def description(cls):
        return '범위가 주어진 짝수의 합, 왼쪽 경계 포함'
    
    @classmethod
    def gen_question(cls, vmin, vmax):
        return make_sentence(vmin, ['부터', '에서', '에서 부터', '보다 크거나 같고', '보다 많거나 같고'], vmax, ['보다 작은', '보다 적은'], 
                             ['', '수 중', '수 중에서'], ['짝수', '짝수의', '짝수를'], ['합', '총합', '총 합', '더하면'], pos_jks, pos_efq+pos_efo)
    
    @classmethod
    def equation(cls, vmin, vmax):
        code = f'''
sum = 0
for i in range({vmin}, {vmax}):
    if i % 2 == 0:
        sum += i
print(sum)
        '''
        return code
        
# -----------------------------------------------------         
class P1_1_8(P1_1):
    @classmethod
    def description(cls):
        return '범위가 주어진 짝수의 합, 경계 불포함'
    
    @classmethod
    def gen_question(cls, vmin, vmax):
        return make_sentence(vmin, ['보다 크고', '보다 많고'], vmax, ['보다 작은', '보다 적은'], 
                             ['', '수 중', '수 중에서'], ['짝수', '짝수의', '짝수를'], ['합', '총합', '총 합', '더하면'], pos_jks, pos_efq+pos_efo)
    
    @classmethod
    def equation(cls, vmin, vmax):
        code = f'''
sum = 0
for i in range({vmin} + 1, {vmax}):
    if i % 2 == 0:
        sum += i
print(sum)
        '''
        return code

# ----------------------------------------------------- 
class P1_1_9(P1_1):
    @classmethod
    def description(cls):
        return '범위가 주어진 수의 합, 경계 포함'
    
    @classmethod
    def gen_question(cls, vmin, vmax):
        return make_sentence(vmin, ['부터', '에서', '에서 부터', '보다 크거나 같고', '보다 많거나 같고'], vmax, 
                             ['까지', '까지의', '보다 작거나 같은', '보다 적거나 같은'], 
                             ['수', '수의', '수를'], ['합', '총합', '총 합', '더하면'], pos_jks, pos_efq+pos_efo)
    
    @classmethod
    def equation(cls, vmin, vmax):
        code = f'''
sum = 0
for i in range({vmin}, {vmax} + 1):
    sum += i
print(sum)
        '''
        return code
        
# -----------------------------------------------------         
class P1_1_10(P1_1):
    @classmethod
    def description(cls):
        return '범위가 주어진 수의 합, 오른쪽 경계 포함'
    
    @classmethod
    def gen_question(cls, vmin, vmax):
        return make_sentence(vmin, ['보다 크고', '보다 많고'], vmax, ['까지', '까지의', '보다 작거나 같은', '보다 적거나 같은'], 
                            ['수', '수의', '수를'], ['합', '총합', '총 합', '더하면'], pos_jks, pos_efq+pos_efo)
    
    @classmethod
    def equation(cls, vmin, vmax):
        code = f'''
sum = 0
for i in range({vmin} + 1, {vmax} + 1):
    sum += i
print(sum)
        '''
        return code

# ----------------------------------------------------- 
class P1_1_11(P1_1):
    @classmethod
    def description(cls):
        return '범위가 주어진 수의 합, 왼쪽 경계 포함'
    
    @classmethod
    def gen_question(cls, vmin, vmax):
        return make_sentence(vmin, ['부터', '에서', '에서 부터', '보다 크거나 같고', '보다 많거나 같고'], vmax, ['보다 작은', '보다 적은'], 
                             ['수', '수의', '수를'], ['합', '총합', '총 합', '더하면'], pos_jks, pos_efq+pos_efo)
    
    @classmethod
    def equation(cls, vmin, vmax):
        code = f'''
sum = 0
for i in range({vmin}, {vmax}):
    sum += i
print(sum)
        '''
        return code
        
# -----------------------------------------------------         
class P1_1_12(P1_1):
    @classmethod
    def description(cls):
        return '범위가 주어진 수의 합, 경계 불포함'
    
    @classmethod
    def gen_question(cls, vmin, vmax):
        return make_sentence(vmin, ['보다 크고', '보다 많고'], vmax, ['보다 작은', '보다 적은'], 
                             ['수', '수의', '수를'], ['합', '총합', '총 합', '더하면'], pos_jks, pos_efq+pos_efo)
    
    @classmethod
    def equation(cls, vmin, vmax):
        code = f'''
sum = 0
for i in range({vmin} + 1, {vmax}):
    sum += i
print(sum)
        '''
        return code

#        *********************************** type 1 - 2 *******************************
class P1_2(Problem):
    @classmethod
    def description(cls):
        return '컨테이너 안의 물건 개수 가감'

    @classmethod
    def random_values(cls):
        x, y = np.random.choice(list(range(50)), size=2, replace=False)
        return x, y

# ----------------------------------------------------- 
class P1_2_1(P1_2):
    @classmethod
    def description(cls):
        return '컨테이너 안의 물건 개수 추가'
    
    @classmethod
    def gen_question(cls, x, y):
        obj = np.random.choice(ws_object)
        container = np.random.choice(ws_container)
        person = np.random.choice(ws_person)
        return make_sentence(container, ['안에', '속에', '에'], x, ['개', '개의'], obj, pos_jks, pos_there, person, pos_jks, y, ['개', '개의'], 
                             obj, pos_jko, ['', '더'], ['넣었다', '집어 넣었다', '넣었습니다.' '넣을 때', '넣었을 때', '넣으면'], 
                             ['', container],  ['안에', '속에', '에'], ['', '있는'], obj, pos_jks,
                             ['', '모두', '전체', '다해서', '다 해서', '모두 해서', '모두 더해서'], ask_how_many('개'))
                                                                                             
    @classmethod
    def equation(cls, x, y):
        code = f'''
print({x} + {y})
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn != 2 or nv + nf + ne + nl > 0:
            return None
        else:
            n1, n2 = numbers[0].value(), numbers[1].value()
            return cls.solve(n1, n2)    
    
# -----------------------------------------------------     
class P1_2_2(P1_2):
    @classmethod
    def description(cls):
        return '컨테이너 안의 물건 개수 감소'
    
    @classmethod
    def gen_question(cls, x, y):
        obj = np.random.choice(ws_object)
        container = np.random.choice(ws_container)
        person = np.random.choice(ws_person)
        return make_sentence(container, ['안에', '속에', '에'], x, ['개', '개의'], obj, pos_jks, pos_there, person, pos_jks, y, ['개', '개의'], 
                             obj, pos_jko, ['뺐다', '꺼냈다', '꺼내니', '빼니', '뺄 때', '꺼낼 때', '빼면', '꺼내면', '꺼냈습니다'], 
                             ['', container],  ['안에', '속에', '에'], ['', '있는', '남아 있는', '남은'], obj, pos_jks,
                             ['', '모두', '전체', '다해서', '다 해서', '모두 해서'], ask_how_many('개'))
                                                                                             
    @classmethod
    def equation(cls, x, y):
        code = f'''
print({x} - {y})
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn != 2 or nv + nf + ne + nl > 0:
            return None
        else:
            n1, n2 = numbers[0].value(), numbers[1].value()
            if n1 < n2:
                return None
            return cls.solve(n1, n2)
        
        
#        *********************************** type 1 - 3 *******************************
class P1_3(Problem):
    @classmethod
    def description(cls):
        return 'n개 컨테이너에 들어있는 물체 수'

    @classmethod
    def random_values(cls):
        x, y = np.random.choice(list(range(1000)), size=2, replace=True)
        return x, y   

# ----------------------------------------------------- 
class P1_3_1(P1_3):    
    @classmethod
    def gen_question(cls, x, y):
        obj = np.random.choice(ws_object)
        container = np.random.choice(ws_container)
        unit = np.random.choice(ws_unit_kor)
        return make_sentence(blank(ws_a), container, ws_in, obj, pos_jks, x, unit, ['들어', '담겨'], ws_bethere, 
                             y, ['개', '개의'], container, ws_in, ['', '있는', '들어 있는', '담겨 있는'], obj, pos_jks, 
                             blank(ws_intotal), ask_how_many(unit))

    
    @classmethod
    def equation(cls, x, y):
        code = f'''
x = {x}
y = {y}
print(x * y)
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn != 2 or nv + nf + ne + nl > 0:
            return None
        else:
            n1, n2 = numbers[0].value(), numbers[1].value()
            return cls.solve(n1, n2)      
        

#        *********************************** type 1 - 4 *******************************
# ----------------------------------------------------- 
class P1_4_1(Problem):
    @classmethod
    def description(cls):
        return 'n 명을 제외한 평균'

    @classmethod
    def random_values(cls):
        n = np.random.choice([1, 2, 3, 4, 5])
        scores = np.random.choice(list(range(1, 200)), size=n, replace=True).tolist()
        mean = np.random.choice(list(range(1, 200)))
        N = np.random.randint(n+1, 100)
        return n, scores, mean, N
    
    @classmethod
    def gen_question(cls, n, scores, mean, N):
        people = np.random.choice(ws_person, size=n, replace=False)
        unit = '점'
        subject = np.random.choice(ws_subject)
        return make_sentence(', '.join(people), ['의', '이의'], subject, '점수', pos_jks, ['', '각각'], ', '.join([str(s) + unit for s in scores]), 
                             ws_be, ['', '이', '이들', '이 사람들'], ['', ['하나', '둘', '셋', '넷', '다섯'][n-1]], pos_jko, ws_except, 
                             ['', '학급의', '반의', '그룹의'], ['', subject], unit, pos_jks, mean, ws_be, ['', '학급의', '반의', '그룹의'], 
                             '평균', unit, pos_jks, ask_how_many(unit))
    
    @classmethod
    def equation(cls, n, scores, mean, N):
        code = f'''
N = {N}
n = {n}
scores = {scores}
mean = {mean}
total = (N - n) * mean + sum(scores)
new_mean = total / {N}
if new_mean / 1 == new_mean // 1:
    print(int(new_mean))
else:
    print('%.2f'%new_mean)
        '''
        return code
             
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn < 2 or nl != 2 or nv + nf + ne > 0:
            return None
        else:
            scores = [n.value() for n in lists[-1]]
            n = len(scores)
            mean = numbers[-2].value()
            N = numbers[-1].value()
            return cls.solve(n, scores, mean, N) 


# ============================================ 유형 2 =================================================
#        *********************************** type 2 - 1 *******************************
# ----------------------------------------------------- 
class P2_1_1(Problem):
    @classmethod
    def description(cls):
        return '줄에서 앞 뒤 순서'

    @classmethod
    def random_values(cls):
        n = np.random.randint(5, 100)
        m = np.random.randint(n)
        return n, m
    
    @classmethod
    def gen_question(cls, n, m):
        person = np.random.choice(ws_person)
        oneside, otherside = np.random.choice(['앞', '뒤'], 2, replace=False)
        human = np.random.choice(['사람', '사람들', '친구', '친구들' '학생', '학생들', '아이', '아이들'])
        unit = np.random.choice(['명', '사람'])
        return make_sentence(n, unit, ['', '의'], human, pos_jks, ['', '한', '한 줄로', '일렬로', '순서대로'], '줄을', 
                             ['섰습니다', '섰다', '섰더니', '섰고', '섰는데'], person, ['', '의'], oneside, ws_at, m, unit, ['', '의'], 
                             ['', '사람들이'], ['', '줄을'], '서', ws_be, person, ['', '의'], otherside, ws_at, ['', '서'], '있는', 
                             human, pos_jks, ask_how_many(unit))
    
    @classmethod
    def equation(cls, n, m):
        code = f'''
print({n} - {m} - 1)
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn != 2 or nv + nf + ne + nl > 0:
            return None
        else:
            n1, n2 = numbers[0].value(), numbers[1].value()
            if n1 < n2:
                return None
            return cls.solve(n1, n2)  
             

#        *********************************** type 2 - 2 *******************************
# ----------------------------------------------------- 
class P2_2_2(Problem):
    @classmethod
    def description(cls):
        return '두사람 사이 등수'

    @classmethod
    def random_values(cls):
        n = np.random.randint(1, 100)
        return n, n + 2
    
    @classmethod
    def gen_question(cls, n, m):
        p1, p2, p = np.random.choice(ws_person, 3, replace=False)
        competition = ws_subject + [s + ' 시험' for s in ws_subject] + ws_sport + [s + np.random.choice([' 시합', ' 경기']) for s in ws_sport] 
        oneside, otherside = np.random.choice(['앞', '뒤'], 2, replace=False)
        human = np.random.choice(['사람', '사람들', '친구', '친구들' '학생', '학생들', '아이', '아이들'])
        unit = '등'
        return make_sentence(competition, ws_at, p1, pos_jks, n, ['등', '등을'], ['이고', '이었으며', '입니다', '했고', '했으며', '했습니다'], 
                             p2, pos_jks, m, ['등', '등을'], ['이고', '이었으며', '입니다', '했고', '했으며', '했습니다'], 
                             p, pos_jks, p2 + '보다', ws_dobetter, p1 + '보다', ws_doworse, p, ['', '의 등수는', '이의 등수는'], 
                            ask_how_many(unit)) 
            
    @classmethod
    def equation(cls, n, m):
        code = f'''
print(int(({n} + {m}) / 2)) 
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn != 2 or nv + nf + ne + nl > 0:
            return None
        else:
            n1, n2 = numbers[0].value(), numbers[1].value()
            return cls.solve(n1, n2)      

#        *********************************** type 2 - 3 *******************************
# ----------------------------------------------------- 
class P2_3_1(Problem):
    @classmethod
    def description(cls):
        return '줄 순서 거꾸로 바꾸기'

    @classmethod
    def random_values(cls):
        n = np.random.randint(5, 100)
        m = np.random.randint(n)
        return n, m
    
    @classmethod
    def gen_question(cls, n, m):
        person = np.random.choice(ws_person)
        comp = np.random.choice(['키', '몸무게', '나이'])
        big, small = np.random.choice([['큰, 많은'], ['작은', '적은']], 2, replace=False) 
        oneside, otherside = np.random.choice(['앞', '뒤'], 2, replace=False)
        human = np.random.choice(['사람', '사람들', '친구', '친구들' '학생', '학생들', '아이', '아이들'])
        unit = np.random.choice(['명', '사람'])
        ws_inline_ = blank(ws_inline)
        return make_sentence(comp, pos_jks, big, human, ['부터', '먼저'], blank(ws_inline), n, unit, pos_jks, '서', ws_bethere, 
                             person, oneside, ws_at, ['', '부터'], m, ['번째에', '번째'], '서', ws_bethere, comp, small, human, '부터', 
                             blank(ws_inline), ['', '다시', '거꾸로'], ['', '줄을'], ['서면', '선다면', '섰습니다'], 
                             person, oneside, ws_at, ['', '부터'],
                             [ask_how_many('번째')] + ['몇 번째 서게 됩니까'])
    
    @classmethod
    def equation(cls, n, m):
        code = f'''
print({n} - {m} + 1)
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn != 2 or nv + nf + ne + nl > 0:
            return None
        else:
            n1, n2 = numbers[0].value(), numbers[1].value()
            if n1 < n2:
                return None
            return cls.solve(n1, n2) 
# ============================================ 유형 3 =================================================
#        *********************************** type 3 - 1 *******************************
# ----------------------------------------------------- 
class P3_1_1(Problem):
    @classmethod
    def description(cls):
        return 'n가지중 m가지 고르기'

    @classmethod
    def random_values(cls):
        n = np.random.randint(2, 20)
        m = np.random.randint(1, n)
        return n, m
    
    @classmethod
    def gen_question(cls, n, m):
        objects = np.random.choice(ws_object_wide, n, replace=False)
        ws_kind = ['가지', '가지의', '가지를']
        ws_sort = ['', '과일을', '야채를', '과목을', '음식을', '학용품을', '장난감을', '물건을', '꽃을', '공을']
        ws_do = ['', '사는', '고르는', '골라서 사는', '뽑는', '선택하는']
        ws_way = ['경우는', '경우의 수는', '방법은', '방법의 수는']
        return make_sentence(', '.join(objects), ['에서', '중에서', '중'], m, ws_kind, ws_sort, ws_do, ws_way, ws_intotal, 
                             ask_how_many('가지'))
        
    
    @classmethod
    def equation(cls, n, m):
        code = f'''
n = {n}
m = {m}
count = 1
for i in range(m):
    count *= (n - i) / (i + 1)
print(int(count))
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn != 1 or nl != 1 or nv + nf + ne > 0:
            return None
        else:
            n = len(lists[0])
            m = numbers[0].value()
            if n < m:
                return None
            return cls.solve(n, m) 
#        *********************************** type 3 - 2 *******************************
# ----------------------------------------------------- 
class P3_2_1(Problem):
    @classmethod
    def description(cls):
        return 'n개를 m명에게 나누어 주기'

    @classmethod
    def random_values(cls):
        n2 = np.random.randint(1, 5)
        n1 = np.random.randint(20)
        return n1, n2
    
    @classmethod
    def gen_question(cls, n1, n2):
        _ws_object = ws_food + ws_flower + ws_ball + ws_stationary
        obj = np.random.choice(_ws_object)
        _ws_receiver = ws_animal + ['사람', '친구']
        receiver = np.random.choice(_ws_receiver)
        unit = '가지'
        return make_sentence(obj, n1, '개를', ['', '다른', '서로 다른'], n2, ['', '마리의'], receiver, ws_to, ['', '나누어', '나눠', '나누려고'],
                             ['', '주려', '주려고'], ['한다', '할 때', '합니다'], obj, pos_jko, ['나눠', '나누어', '나누는'],
                             ['', '주는'], ['방법은', '경우는', '경우의 수는'], blank(ws_intotal), ask_how_many(unit)) 
       
    
    @classmethod
    def equation(cls, n1, n2):
        code = f'''
n1 = {n1}
n2 = {n2}
n = n1
count_n = 1
for i in range(1, n + n2):
    count_n *= i
count_d1 = 1
for i in range(1, n2):
    count_d1 *= i
count_d2 = 1
for i in range(1, n + 1):
    count_d2 *= i
count = count_n / (count_d1 * count_d2)
print(int(count))
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn != 2 or nv + nf + ne + nl > 0:
            return None
        else:
            n1, n2 = numbers[0].value(), numbers[1].value()
            return cls.solve(n1, n2) 
    
# -----------------------------------------------------     
class P3_2_2(Problem):
    @classmethod
    def description(cls):
        return 'n개를 m명에게 1개 이상 나누어 주기'

    @classmethod
    def random_values(cls):
        n2 = np.random.randint(1, 5)
        n3 = np.random.choice([1, 2, 3])
        n1 = np.random.randint(n2 * n3, 30)
        return n1, n2, n3
    
    @classmethod
    def gen_question(cls, n1, n2, n3):
        _ws_object = ws_food + ws_flower + ws_ball + ws_stationary
        obj = np.random.choice(_ws_object)
        _ws_receiver = ws_animal + ['사람', '친구']
        receiver = np.random.choice(_ws_receiver)
        unit = '가지'
        return make_sentence(obj, n1, '개를', ['', '다른', '서로 다른'], n2, ['', '마리의'], receiver, ws_to, ['', '나누어', '나눠', '나누려고'],
                             ['', '주려', '주려고'], ['한다', '할 때', '합니다'], receiver, pos_jks, '적어도', ['', obj], n3, ['개는', '개를'], 
                             ['받습니다', '받게 됩니다', '받을 수 있습니다', '받는다고 할 때'], obj, pos_jko, ['나눠', '나누어', '나누는'],
                             ['', '주는'], ['방법은', '경우는', '경우의 수는'], blank(ws_intotal), ask_how_many(unit)) 
       
    
    @classmethod
    def equation(cls, n1, n2, n3):
        code = f'''
n1 = {n1}
n2 = {n2}
n3 = {n3}
n = n1 - (n2 * n3)
count_n = 1
for i in range(1, n + n2):
    count_n *= i
count_d1 = 1
for i in range(1, n2):
    count_d1 *= i
count_d2 = 1
for i in range(1, n + 1):
    count_d2 *= i
count = count_n / (count_d1 * count_d2)
print(int(count))
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn != 3 or nv + nf + ne + nl > 0:
            return None
        else:
            n1, n2, n3 = numbers[0].value(), numbers[1].value(), numbers[2].value()
            return cls.solve(n1, n2, n3) 
#        *********************************** type 3 - 3 *******************************
# ----------------------------------------------------- 
class P3_3_1(Problem):
    @classmethod
    def description(cls):
        return 'n개의 숫자 한번씩만 사용하여 만들 수 있는 n자리 (m으로 확장 가능) 숫자'

    @classmethod
    def random_values(cls):
        n = np.random.choice([2, 3, 4, 5])
        numbers = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], n, replace=False).tolist()                           
        return [numbers]
    
    @classmethod
    def gen_question(cls, numbers):
        n = len(numbers)
        mdn = pos_mdn[n]
        return make_sentence(n, ['개', '개의'], ['수', '숫자'], list2str(numbers), '를', ['한 번씩', '한 번씩만', '단 한 번만', '단 한번만'], 
                             ['사용하여', '써서', '이용하여'], mdn, '자리', ['수', '숫자'], pos_jko, 
                             ['만들려고 합니다', '만든다', '만들 때', '만듭니다', '만들 면'], ['', '만들 수 있는', '가능한'],['', mdn + '자리'], 
                             ['수는', '숫자는'], blank(ws_intotal), ask_how_many(['개', '가지']))
   
    @classmethod
    def equation(cls, numbers):
        code = f'''
numbers = {numbers}
n = len(numbers)
count = 1
if 0 in numbers:
    for i in range(n):
        if i == 0:
            count *= n - 1
        else:
            count *= n - i
else:
    for i in range(n):
        count *= n - i
print(count)
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nl != 1 or nv + nf + ne > 0:
            return None
        else:
            return cls.solve(lists[0].tolist()) 

# ============================================ 유형 4 =================================================
#        *********************************** type 4 - 1 *******************************
# ----------------------------------------------------- 
class P4_1_1(Problem):
    @classmethod
    def description(cls):
        return 'n개의 숫자 중 가장 큰 수와 가장 작은 수의 차이'

    @classmethod
    def random_values(cls):
        n = np.random.randint(3, 15)
        numbers = np.random.randint(1, 1000, size=n).tolist()   
        return [numbers]
    
    @classmethod
    def gen_question(cls, numbers):
        n = len(numbers)
        return make_sentence(n, ['개', '개의'], ['수', '숫자'], list2str(numbers), pos_jks, ws_bethere, ['', '그 중에서', '그 중에', '이 중에'],
                             ['가장 큰 수와 가장 작은 수', '가장 작은 수와 가장 큰 수', '제일 큰 수와 제일 작은 수', '제일 작은 수와 제일 큰 수'], 
                             ['', '의'], ['차', '차이'], pos_jks, ask_how_many())
   
    @classmethod
    def equation(cls, numbers):
        code = f'''
numbers = {numbers}
print(max(numbers) - min(numbers))
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn > 2 or nl != 1 or nv + nf + ne > 0:
            return None
        else:
            return cls.solve(lists[0].tolist())     
#        *********************************** type 4 - 2 *******************************
# ----------------------------------------------------- 
class P4_2_1(Problem):
    @classmethod
    def description(cls):
        return '소수점 오른쪽 이동'

    @classmethod
    def random_values(cls):
        n = np.random.randint(1, 4)
        delta = (np.random.rand() * 5000 // 1) / 100
        return n, delta
    
    @classmethod
    def gen_question(cls, n, delta):
        mdn = pos_mdn[n-1]
        return make_sentence('어떤', ['수의', '숫자의', '소수의'], '소수점을', '오른쪽으로', mdn, '자리', ['', '만큼'], ['옮기면', '움직이면', '이동하면'], 
                             ['원래보다', '원래 수보다', '원래 숫자보다', '원래 소수보다'], delta, '만큼', 
                             ['크다', '커진다', '커집니다', '클 때', '커질 때', '커진다고 할 때'], '원래의', ['수', '숫자', '소수'], pos_jko, 
                             [ask_how_many()] + ws_ask_what)

   
    @classmethod
    def equation(cls, n, delta):
        code = f'''
n = {n}
delta = {delta}
original = delta / (10**n - 1)
print('%.2f'%original)
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn != 2 or nv + nf + ne + nl > 0:
            return None
        else:
            n = numbers[0].value()
            delta = numbers[1].value()
            return cls.solve(n, delta) 
    
# -----------------------------------------------------                                 

class P4_2_2(Problem):
    @classmethod
    def description(cls):
        return '소수점 왼쪽 이동'

    @classmethod
    def random_values(cls):
        n = np.random.randint(1, 4)
        delta = (np.random.rand() * 5000 // 1) / 1000
        return n, delta
    
    @classmethod
    def gen_question(cls, n, delta):
        mdn = pos_mdn[n-1]
        return make_sentence('어떤', ['수의', '숫자의', '소수의'], '소수점을', '왼쪽으로', mdn, '자리', ['', '만큼'], ['옮기면', '움직이면', '이동하면'], 
                             ['원래보다', '원래 수보다', '원래 숫자보다', '원래 소수보다'], delta, '만큼', 
                             ['작다', '작아진다', '작아집니다', '작아 질 때', '작을 때', '작아진다고 할 때'],
                             '원래의', ['수', '숫자', '소수'], pos_jko, 
                             [ask_how_many()] + ws_ask_what)

   
    @classmethod
    def equation(cls, n, delta):
        code = f'''
n = {n}
delta = {delta}
original = delta / (1 - 10**(-n))
original = (original * 100 // 1) / 100
print('%.2f'%original)
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn != 2 or nv + nf + ne + nl > 0:
            return None
        else:
            n = numbers[0].value()
            delta = numbers[1].value()
            return cls.solve(n, delta) 
    
    
#        *********************************** type 4 - 3 *******************************
class P4_3_1(Problem):
    @classmethod
    def description(cls):
        return 'n개의 수로 나누어지는 k 자리 수'

    @classmethod
    def random_values(cls):
        n = np.random.randint(2, 5)
        numbers = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9], n, replace=False).tolist()   
        k = np.random.choice([1, 2, 3, 4])
        return numbers, k
    
    @classmethod
    def gen_question(cls, numbers, k):
        n = len(numbers)
        mdn = pos_mdn[k-1]
        return make_sentence(n, ['개', '개의'], ['수', '숫자'], list2str(numbers), '로', 
                             ['나누어 지는', '나눌 수 있는', '나누어 떨어지는', '나눌때 나머지가 0이 되는'], 
                             mdn, '자리', ['수', '숫자'], pos_jks, blank(ws_intotal), ask_how_many('개'))
   
    @classmethod
    def equation(cls, numbers, k):
        code = f'''
numbers = {numbers}
k = {k}
ceil = 10**k
floor = 10**(k-1)
count = 0
for i in range(floor, ceil):
    divided = True
    for d in numbers:
        if i % d != 0:
            divided = False
    if divided:
        count += 1
print(count)
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn > 3 or nn == 1 or nl != 1 or nv + nf + ne > 0:
            return None
        else:
            k = numbers[-1].value()
            return cls.solve(lists[0].tolist(), k)  
# ============================================ 유형 5 =================================================
#        *********************************** type 5 - 1 *******************************
# ----------------------------------------------------- 
class P5_1_1(Problem):
    @classmethod
    def description(cls):
        return '두자리 수의 덧셈식'

    @classmethod
    def random_values(cls):
        n1 = np.random.randint(0, 10)
        n2 = np.random.randint(1, 10)
        n3 = np.random.randint(10, 100)
        return n1, n2, n3
    
    @classmethod
    def gen_question(cls, n1, n2, n3):
        x, y = np.random.choice(ws_variable, 2, replace=False).tolist()
        return make_sentence('두 자리 수의 덧셈식', ['', '이 다음과 같이 주어질 때'], x + str(n1) + '+' + str(n2) + y + '=' + str(n3), 
                             ['에서', '와 같을 때', ' 와 같이 주어질 때'], x, '에', ['해당하는', '들어갈','알맞은'], 
                              ['수를', '숫자를', '수는', '숫자는'], ws_ask_what + [ask_how_many()])

   
    @classmethod
    def equation(cls, n1, n2, n3):
        code = f'''
n1 = {n1}
n2 = {n2}
n3 = {n3}
if n3 % 10 >= n1:
    A  = int(n3 // 10) - n2
else:
    A  = int(n3 // 10) - n2 - 1
print(A)
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        return None
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if ne != 1:
            return '', ''
        else:
            eq = equations[0].text
            xy, z = eq.split('==')
            x, y = xy.strip().split('+')
            x = x.strip()
            y = y.strip()
            z = z.strip()
            try:
                n1 = int(x[0])
                n2 = int(y[1])
                n3 = int(z)    
            except: 
                try:
                    n1 = int(x[1])
                    n2 = int(y[0])
                    n3 = int(z) 
                except:
                    return None            
            return cls.solve(n1, n2, n3)  
#        *********************************** type 5 - 2 *******************************
# ----------------------------------------------------- 
class P5_2_1(Problem):
    @classmethod
    def description(cls):
        return '몫과 나머지가같을때 나누는 수중 큰 수'

    @classmethod
    def random_values(cls):
        n = np.random.randint(1, 10)
        return [n]
    
    @classmethod
    def gen_question(cls, n):
        x, y, z = np.random.choice(ws_variable, 3, replace=False).tolist()
        return make_sentence(x, pos_jko, n, ['로', '으로'], ['나누면', '나누니', '나누었더니'], ['', '몫은'], y, ws_be + ws_become, 
                             '나머지는', z, pos_jks, ws_be + ws_become, list2str([x, y, z]), pos_jks, '자연수', ws_be, 
                             ['', '이 식에서', '여기서', '여기에서'], '몫과 나머지가', ws_be_same, 
                             ['나누어 지는', '나뉘는'], ['수', '숫자'], x, ['', '중', '중에서'], '가장', '큰', 
                             ['수', '숫자'], ws_ask_what)
   
    @classmethod
    def equation(cls, n):
        code = f'''
n = {n}
print((n + 1) * (n - 1))
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn != 1:
            return None
        else:
            n = numbers[0].value()
            return cls.solve(n)  
    
#        *********************************** type 5 - 3 *******************************
# ----------------------------------------------------- 
class P5_3_1(Problem):
    @classmethod
    def description(cls):
        return '두 자연수의 연립 방정식'

    @classmethod
    def random_values(cls):
        n = np.random.randint(1, 20)
        return [n * 5]
    
    @classmethod
    def gen_question(cls, n):
        x, y = np.random.choice(ws_variable, 2, replace=False).tolist()
        return make_sentence('서로 다른 두 자연수', list2str([x, y]), pos_jks, ws_bethere, 
                             x + '+' + y + '=' + str(n), x + '=' + y + '+' + y + '+' + y + '+' + y, ws_be, 
                             x, pos_jko, ['구하시오', '얼마인가', '얼마인가요', '얼마입니까', '무엇인가', '무엇일까요'])

   
    @classmethod
    def equation(cls, n):
        code = f'''
n = {n}
print(4 * int(n / 5))
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        return '', ''
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if ne != 2:
            return None
        else:
            eq = equations[0].text
            n = int(eq.split('==')[1])
            return cls.solve(n) 
# ============================================ 유형 6 =================================================
#        *********************************** type 6 - 1 *******************************
# ----------------------------------------------------- 
class P6_1_1(Problem):
    @classmethod
    def description(cls):
        return '잘 못 뺀 계산'

    @classmethod
    def random_values(cls):
        n1, n2, n3 = np.random.randint(1, 100, size=3).tolist()
        return n1, n2, n3
    
    @classmethod
    def gen_question(cls, n1, n2, n3):
        return make_sentence('어떤', ['수', '숫자'], '에서', n1, pos_jko, '빼야', ['하는데', '하지만', '할 것을'], ['잘못하여', '실수로'], 
                             n2, pos_jko, ['뺐더니', '빼니', '빼서', '뺀 결과', '뺀 결과가'], n3, pos_jks, ws_be  + ws_become, 
                             ['바르게', '제대로', '올바로', '똑바로', '올바르게'], ['계산한', '구한'], 
                             ['결과는', '결과를', '답을', '답은'], ['구하시오', '얼마인가', '얼마인가요', '얼마입니까', '무엇인가', '무엇일까요'])

   
    @classmethod
    def equation(cls, n1, n2, n3):
        code = f'''
n1 = {n1}
n2 = {n2}
n3 = {n3}
n = n3 + n2
print(n - n1)
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn != 3:
            return None
        else:
            n1, n2, n3 = numbers[0].value(), numbers[1].value(), numbers[2].value()
            return cls.solve(n1, n2, n3) 
#        *********************************** type 6 - 2 *******************************
# ----------------------------------------------------- 


#        *********************************** type 6 - 3 *******************************
# ----------------------------------------------------- 
class P6_3_1(Problem):
    @classmethod
    def description(cls):
        return '뺄셈 계산 수정'

    @classmethod
    def random_values(cls):
        n1, n2, n3 = np.random.randint(1, 100, size=3).tolist()
        return n1, n2, n3
    
    @classmethod
    def gen_question(cls, n1, n2, n3):
        return make_sentence('어떤', ['수', '숫자'], '에서', n1, pos_jko, ['빼니', '뺐더니'], n2, pos_jks, ws_be + ws_become, 
                             '어떤', ['수', '숫자'], '에서', n3, pos_jko, ['빼면', '뺀다면'], 
                             ['', '결과는', '결과를', '답을', '답은'], ['구하시오', '얼마인가', '얼마인가요', '얼마입니까', '무엇인가', '무엇일까요'])

    @classmethod
    def equation(cls, n1, n2, n3):
        code = f'''
n1 = {n1}
n2 = {n2}
n3 = {n3}
n = n1 + n2
print(n - n3)
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn != 3:
            return None
        else:
            n1, n2, n3 = numbers[0].value(), numbers[1].value(), numbers[2].value()
            return cls.solve(n1, n2, n3) 
#        *********************************** type 6 - 4 *******************************
# ----------------------------------------------------- 
class P6_4_1(Problem):
    @classmethod
    def description(cls):
        return '잘못 계산한 곱셈'

    @classmethod
    def random_values(cls):
        n1, n2, n3 = np.random.randint(1, 100, size=3).tolist()
        return n1, n2, n3
    
    @classmethod
    def gen_question(cls, n1, n2, n3):
        if np.random.rand() < 0.5:
            return make_sentence('어떤', ['수', '숫자'], '에', n1, pos_jko, '곱해야', ['하는데', '하지만', '할 것을'], ['잘못하여', '실수로'], 
                             ['', '어떤 수에'], n2, pos_jko, ['곱하니', '곱했더니', '곱하고 나니', '곱한 결과', '곱한 결과가'], 
                              n3, pos_jks, ws_be  + ws_become, 
                             ['바르게', '제대로', '올바로', '똑바로', '올바르게'], ['계산한', '구한'], 
                             ['결과는', '결과를', '답을', '답은'], ['구하시오', '얼마인가', '얼마인가요', '얼마입니까', '무엇인가', '무엇일까요'])
        else:
            return make_sentence(n1, '에', '어떤', ['수', '숫자'], pos_jko, '곱해야', ['하는데', '하지만', '할 것을'], ['잘못하여', '실수로'], 
                             n2, '에', ['', '어떤 수를'], ['곱하니', '곱했더니', '곱하고 나니', '곱한 결과', '곱한 결과가'], 
                              n3, pos_jks, ws_be  + ws_become, 
                             ['바르게', '제대로', '올바로', '똑바로', '올바르게'], ['계산한', '구한'], 
                             ['결과는', '결과를', '답을', '답은'], ['구하시오', '얼마인가', '얼마인가요', '얼마입니까', '무엇인가', '무엇일까요'])

    @classmethod
    def equation(cls, n1, n2, n3):
        code = f'''
n1 = {n1}
n2 = {n2}
n3 = {n3}
n = n3 / n2
print(int(n * n1))
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn != 3:
            return None
        else:
            n1, n2, n3 = numbers[0].value(), numbers[1].value(), numbers[2].value()
            return cls.solve(n1, n2, n3) 
# ============================================ 유형 7 =================================================
class P7(Problem):
    @classmethod
    def description(cls):
        return 'n차의 등차 수열 규칙'
    
    @classmethod
    def get_equidiff_series(cls, n, order=0):
        diff = np.random.randint(1, 30)
        a0 = np.random.randint(20)
        series0 = [a0]
        for i in range(n):
            a0 += diff
            series0.append(a0)
        if order == 0:
            return series0[:n]
        a1 = np.random.randint(20)
        series1 = [a1]
        for a0 in series0:
            a1 += a0
            series1.append(a1)
        if order == 1:
            return series1[:n]
        a2 = np.random.randint(20)
        series2 = [a2]
        for a1 in series1:
            a2 += a1
            series2.append(a2)
        if order == 2:
            return series2[:n]
        raise Exception('order > 2 not supported.')
#        *********************************** type 7 - 1 *******************************
# ----------------------------------------------------- 
class P7_1_1(P7):
    @classmethod
    def description(cls):
        return '수열에서 n번째와 m번째 숫자의 차'

    @classmethod
    def random_values(cls):
        n = np.random.randint(5, 11)
        order = np.random.randint(3)
        series = cls.get_equidiff_series(n, order)
        n1, n2 = np.random.choice(list(range(3, 100)), 2, replace=False).tolist()
        return series, n1, n2
    
    @classmethod
    def gen_question(cls, series, n1, n2):
        x, y = np.random.choice(ws_variable, 2, replace=False).tolist()
        return make_sentence(['', '어떤 수열'], list2str(series), ['', '와', '과'], '같은', ['규칙에서', '규칙이 있을 때', '규칙을 따를 때'], 
                             n1, '번째', ['', '놓일', '놓인', '오는'], ['수', '숫자'], n2, '번째', ['', '놓일', '놓인', '오는'], ['수', '숫자'],
                             pos_jko, ['', '각각'], x, ws_and, y, ['라 할 때', '라 한다'] + ws_be, x + '-' + y, pos_jks, ask_how_many())

    @classmethod
    def equation(cls, series, n1, n2):
        code = f'''
series = {series}
n1 = {n1}
n2 = {n2}
N = len(series)
series0 = []
for i in range(N - 1):
    series0.append(series[i+1] - series[i])
N = len(series0)
series1 = []
for i in range(N - 1):
    series1.append(series0[i+1] - series0[i])
N = len(series1)
series2 = []
for i in range(N - 1):
    series2.append(series1[i+1] - series1[i])
if max(series0) == min(series0):
    order = 0
elif max(series1) == min(series1):
    order = 1
elif max(series2) == min(series2):
    order = 2
else:
    order = None    
n = max(n1, n2)    
if order == 0:
    _series0 = [series0[0]] * n
    _series = [series[0]]
    for i in range(n):
        _series.append(_series[-1] + _series0[i])
if order == 1:
    _series1 = [series1[0]] * n
    _series0 = [series0[0]]
    for i in range(n):
        _series0.append(_series0[-1] + _series1[i])
    _series = [series[0]]
    for i in range(n+1):
        _series.append(_series[-1] + _series0[i])
if order == 2:
    _series2 = [series2[0]] * n
    _series1 = [series1[0]]
    for i in range(n):
        _series1.append(_series1[-1] + _series2[i])
    _series0 = [series0[0]]
    for i in range(n+1):
        _series0.append(_series0[-1] + _series1[i])
    _series = [series[0]]
    for i in range(n+2):
        _series.append(_series[-1] + _series0[i])
print(_series[n2 - 1] - _series[n1 - 1])        
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn != 2 or nl != 1:
            return None
        else:
            n1, n2 = numbers[0].value(), numbers[1].value()
            series = lists[0].tolist()
            return cls.solve(series, n1, n2) 
# ----------------------------------------------------- 
class P7_1_2(P7):
    @classmethod
    def description(cls):
        return '수열에서 n번째와 m번째 숫자의 합'

    @classmethod
    def random_values(cls):
        n = np.random.randint(5, 11)
        order = np.random.randint(3)
        series = cls.get_equidiff_series(n, order)
        n1, n2 = np.random.choice(list(range(3, 100)), 2, replace=False).tolist()
        return series, n1, n2
    
    @classmethod
    def gen_question(cls, series, n1, n2):
        x, y = np.random.choice(ws_variable, 2, replace=False).tolist()
        return make_sentence(['', '어떤 수열'], list2str(series), ['', '와', '과'], '같은', ['규칙에서', '규칙이 있을 때', '규칙을 따를 때'], 
                             n1, '번째', ['', '놓일', '놓인', '오는'], ['수', '숫자'], n2, '번째', ['', '놓일', '놓인', '오는'], ['수', '숫자'],
                             pos_jko, ['', '각각'], x, ws_and, y, ['라 할 때', '라 한다'] + ws_be, x + '+' + y, ask_how_many())

    @classmethod
    def equation(cls, series, n1, n2):
        code = f'''
series = {series}
n1 = {n1}
n2 = {n2}
N = len(series)
series0 = []
for i in range(N - 1):
    series0.append(series[i+1] - series[i])
N = len(series0)
series1 = []
for i in range(N - 1):
    series1.append(series0[i+1] - series0[i])
N = len(series1)
series2 = []
for i in range(N - 1):
    series2.append(series1[i+1] - series1[i])
if max(series0) == min(series0):
    order = 0
elif max(series1) == min(series1):
    order = 1
elif max(series2) == min(series2):
    order = 2
else:
    order = None
    
n = max(n1, n2)    
if order == 0:
    _series0 = [series0[0]] * n
    _series = [series[0]]
    for i in range(n):
        _series.append(_series[-1] + _series0[i])
if order == 1:
    _series1 = [series1[0]] * n
    _series0 = [series0[0]]
    for i in range(n):
        _series0.append(_series0[-1] + _series1[i])
    _series = [series[0]]
    for i in range(n+1):
        _series.append(_series[-1] + _series0[i])
if order == 2:
    _series2 = [series2[0]] * n
    _series1 = [series1[0]]
    for i in range(n):
        _series1.append(_series1[-1] + _series2[i])
    _series0 = [series0[0]]
    for i in range(n+1):
        _series0.append(_series0[-1] + _series1[i])
    _series = [series[0]]
    for i in range(n+2):
        _series.append(_series[-1] + _series0[i])
print(_series[n2 - 1] + _series[n1 - 1])        
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn != 2 or nl != 1:
            return None
        else:
            n1, n2 = numbers[0].value(), numbers[1].value()
            series = lists[0].tolist()
            return cls.solve(series, n1, n2) 
#        *********************************** type 7 - 2 *******************************
# ----------------------------------------------------- 

#        *********************************** type 7 - 3 *******************************
# ----------------------------------------------------- 
class P7_3_1(P7):
    @classmethod
    def description(cls):
        return '수열의 n번째 항'

    @classmethod
    def random_values(cls):
        n = np.random.randint(5, 11)
        order = np.random.randint(3)
        series = cls.get_equidiff_series(n, order)
        n1, n2 = np.random.choice(list(range(3, 100)), 2, replace=False).tolist()
        return series, n
    
    @classmethod
    def gen_question(cls, series, n):
        x, y = np.random.choice(ws_variable, 2, replace=False).tolist()
        return make_sentence(['', '어떤 수열'], list2str(series), ['', '와', '과'], '같은', 
                             ['규칙에서', '규칙이 있을 때', '규칙을 따를 때', '규칙에 따라'], ['수', '숫자'], pos_jko, 
                             '배열하고', ['있다', '있습니다', '있을 때', '있다면'], n, '번째', ['', '수는', '숫자는'], 
                             ws_ask_what)

    @classmethod
    def equation(cls, series, n):
        code = f'''
series = {series}
n = {n}
N = len(series)
series0 = []
for i in range(N - 1):
    series0.append(series[i+1] - series[i])
N = len(series0)
series1 = []
for i in range(N - 1):
    series1.append(series0[i+1] - series0[i])
N = len(series1)
series2 = []
for i in range(N - 1):
    series2.append(series1[i+1] - series1[i])
if max(series0) == min(series0):
    order = 0
elif max(series1) == min(series1):
    order = 1
elif max(series2) == min(series2):
    order = 2
else:
    order = None
    
if order == 0:
    _series0 = [series0[0]] * n
    _series = [series[0]]
    for i in range(n):
        _series.append(_series[-1] + _series0[i])
if order == 1:
    _series1 = [series1[0]] * n
    _series0 = [series0[0]]
    for i in range(n):
        _series0.append(_series0[-1] + _series1[i])
    _series = [series[0]]
    for i in range(n+1):
        _series.append(_series[-1] + _series0[i])
if order == 2:
    _series2 = [series2[0]] * n
    _series1 = [series1[0]]
    for i in range(n):
        _series1.append(_series1[-1] + _series2[i])
    _series0 = [series0[0]]
    for i in range(n+1):
        _series0.append(_series0[-1] + _series1[i])
    _series = [series[0]]
    for i in range(n+2):
        _series.append(_series[-1] + _series0[i])
print(_series[n - 1])        
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn != 1 or nl != 1:
            return None
        else:
            n = numbers[0].value()
            series = lists[0].tolist()
            return cls.solve(series, n) 
# ============================================ 유형 8 =================================================
#        *********************************** type 8 - 1 *******************************
# ----------------------------------------------------- 
class P8_1_1(Problem):
    @classmethod
    def description(cls):
        return '반복되는 수열의 n번째 항'

    @classmethod
    def random_values(cls):
        n = np.random.randint(2, 10)
        N = np.random.randint(n + 1, 3 * n)
        numbers = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], n, replace=False).tolist()
        repeat = int(N // n)
        rest = N % n
        series = numbers * repeat + numbers[:rest]
        k = np.random.randint(1, 100)
        return series, k
    
    @classmethod
    def gen_question(cls, series, k):
        return make_sentence(list2str(series), ws_and, '같이', ['반복되는', '반복하는'], '수열이', ws_bethere, 
                             ['', '왼쪽에서', '처음에서', '왼쪽부터'],  
                             k, '번째', ['수', '숫자'], pos_jks, ws_ask_what)
        
                            
    @classmethod
    def equation(cls, series, k):
        code = f'''
series = {series}
k = {k}
N = len(series)
if k <= N:
    print(series[k-1])
else:
    for n in range(1, N + 1):
        repeat = int(N // n)
        rest = N % n
        reseries = series[:n] * repeat + series[:rest]
        if series == reseries:
            j = k % n
            break
    print(series[j-1])
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn != 1 or nl != 1:
            return None
        else:
            n = numbers[0].value()
            series = lists[0].tolist()
            return cls.solve(series, n) 
#        *********************************** type 8 - 2 *******************************
# ----------------------------------------------------- 
class P8_2_1(Problem):
    @classmethod
    def description(cls):
        return '반복되는 3가지 물체의 n번째 색깔'

    @classmethod
    def random_values(cls):
        n1, n2, n3 = np.random.choice([1, 2, 3, 4, 5], 3).tolist()
        k = np.random.randint(1, 100)
        colors = np.random.choice(ws_color, 3, replace=False).tolist()
        return colors, n1, n2, n3, k

    @classmethod
    def gen_question(cls, colors, n1, n2, n3, k):
        c1, c2, c3 = colors
        obj = np.random.choice(ws_object_wide)
        return make_sentence(['', '왼쪽부터', '왼쪽에서부터', '처음부터'], c1, obj, n1, ['', '개'], c2, obj, n2, ['', '개'], 
                             c3, obj, n3, ['', '개'], pos_jks, ['반복하여', '반복되어', '되풀이되어'], ['', '놓여'], 
                             ws_bethere, k, '번째', obj, '의', '색깔', ['을 쓰시오', '은 무엇인가', '은 무엇일까요', '은 무엇일까'])
     
                            

    
    @classmethod
    def equation(cls, colors, ns, k):
        code = f'''
colors = {colors}
ns = {ns}
k = {k}
N = sum(ns)
series = []
for i, c in enumerate(colors):
    series += [c] * ns[i] 
if k <= N:
    print(series[k-1])
else:
    for n in range(1, N + 1):
        repeat = int(N // n)
        rest = N % n
        reseries = series[:n] * repeat + series[:rest]
        if series == reseries:
            j = k % n
            break
    print(series[j-1])
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn < 3 :
            return None
        else:
            colors = [] 
            for it in re.finditer(re_any['color'], question):
                colors.append(it.group())
            if len(colors) != nn - 1:
                return None
            ns = [n.value() for n in numbers[:-1]]
            k = numbers[-1].value()
            return cls.solve(colors, ns, k) 
#        *********************************** type 8 - 3 *******************************
# ----------------------------------------------------- 
class P8_3_1(Problem):
    @classmethod
    def description(cls):
        return 'k개의 물건을 m에게 n개씩 순서대로 나누어 주기'

    @classmethod
    def random_values(cls):
        k = np.random.randint(1, 1000)
        n = np.random.randint(1, 10)
        m = np.random.randint(1, 10)
        people = np.random.choice(ws_person, m, replace=False).tolist()
        return people, k, n

    @classmethod
    def gen_question(cls, people, k, n):
        N = len(people)
        obj = np.random.choice(ws_object_wide)
        return make_sentence(['', str(k) + '개의'], obj, pos_jko, list2str(people), ['', str(N) + '명에게'], ws_to, ws_inline, n, 
                             ['개씩', '개 씩'], ws_distribute, k, '번째', obj, pos_jko, ['받은', '받는', '갖게 되는', '갖는'], 
                             '사람은', ['누구인가요', '누구인가', '누구일까'])
        
    @classmethod
    def equation(cls, people, k, n):
        code = f'''
people = {people}
k = {k}
n = {n}
N = n * len(people)
series = []
for i, p in enumerate(people):
    series += n * [p]
if k <= N:
    print(series[k-1])
else:
    for n in range(1, N + 1):
        repeat = int(N // n)
        rest = N % n
        reseries = series[:n] * repeat + series[:rest]
        if series == reseries:
            j = k % n
            break
    print(series[j-1])
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn < 2 or nl != 1:
            return None
        else:
            series = lists[0].tolist()
            n, k = numbers[-2].value(), numbers[-1].value()
            return cls.solve(series, k, n) 

# ============================================ 유형 9 =================================================
#        *********************************** type 9 - 1 *******************************
# ----------------------------------------------------- 
class P9_1_1(Problem):
    @classmethod
    def description(cls):
        return '수의 합의 크기 비교'

    @classmethod
    def random_values(cls):
        numbers = np.random.randint(1, 100, size=4).tolist()
        p1, p2 = np.random.choice(ws_person, 2, replace=False).tolist()
        return p1, p2, numbers
    
    @classmethod
    def gen_question(cls, p1, p2, numbers):
        n1, n2, n3, n4 = numbers
        _ws_have = ['가지고 있다', '가지고 있습니다', '갖고 있다', '갖고 있을때', '가지고 있다고 하면', 
                    '모았다', '모았습니다', '모았어요', '모았다고 하고', '모았다고 하면']
        return make_sentence(p1, pos_jks, n1, ws_and, n2, pos_jko, _ws_have, p2, pos_jks, n3, ws_and, n4, pos_jko, _ws_have, 
                             ['누가 모은 수가 더 큽니까', '누가 가진 수가 더 큽니까', '누가 모은 수가 더 큰가', '누가 가진 수가 더 큰가', 
                              '더 큰 수를 가진 사람은 누구인가', '더 큰 수를 모은 사람은 누구인가', '더 큰 수를 가진 사람은 누구입니까', 
                              '더 큰 수를 모은 사람은 누구입니까'])


    @classmethod
    def equation(cls, p1, p2, numbers):
        code = f'''
numbers = {numbers}
p1 = '{p1}'
p2 = '{p2}'
n1, n2, n3, n4 = numbers
if n1 + n2 >= n3 + n4:
    print(p1)
else:
    print(p2)
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn != 4:
            return None
        people = []
        for it in re.finditer(re_any['person'], question):
            people.append(it.group())
        ns = [n.value() for n in numbers]
        return cls.solve(people[0], people[1], ns) 
#        *********************************** type 9 - 2 *******************************
# ----------------------------------------------------- 
class P9_2_1(Problem):
    @classmethod
    def description(cls):
        return '4개의 크기 순서 비교'

    @classmethod
    def random_values(cls):
        N= 4
        c = np.random.choice([1,2])
        if c == 1:
            names = np.random.choice(ws_gname, N).tolist()
        elif c == 2:
            names = np.random.choice(ws_person, N).tolist()
        return [names]
    
    @classmethod
    def gen_question(cls, names):
        N = len(names)
        n1, n2, n3, n4 = names
        _ws_obj = (['', '사람', '학생', '아이', '어린이', '친구', '상자', '박스', '자동차', '공', '과일', '열매'])
        obj = np.random.choice(_ws_obj)
        return make_sentence(list2str(names), ['', str(N) + '개의'], obj, pos_jks, ws_bethere, n2, obj, pos_jks, n4, obj, '보다', 
                             ['크다', '큽니다', '크고', '크며'], n1, obj, pos_jks, n4, obj, '보다', 
                             ['작다', '작습니다', '작고', '작으며'], n2, obj, pos_jks, n3, obj, '보다', 
                             ['작다', '작습니다', '작고', '작으며', '작다고 할 때'], ['', '크기가'], '가장 작은', obj, 
                             ws_ask_what)
                             
    @classmethod
    def equation(cls, names):
        code = f'''
names = {names}
n1, n2, n3, n4 = names
print(n1)
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nl != 1:
            return None
        else:
            names = lists[0].tolist()
            return cls.solve(names)
# ----------------------------------------------------- 
class P9_2_2(Problem):
    @classmethod
    def description(cls):
        return '4개의 크기 순서 비교'

    @classmethod
    def random_values(cls):
        N= 4
        c = np.random.choice([1,2])
        if c == 1:
            names = np.random.choice(ws_gname, N, replace=False).tolist()
        elif c == 2:
            names = np.random.choice(ws_person, N, replace=False).tolist()
        return [names]
    
    @classmethod
    def gen_question(cls, names):
        N = len(names)
        n1, n2, n3, n4 = names
        _ws_obj = (['', '사람', '학생', '아이', '어린이', '친구', '상자', '박스', '자동차', '공', '과일', '열매'])
        obj = np.random.choice(_ws_obj)
        return make_sentence(list2str(names), ['', str(N) + '개의'], obj, pos_jks, ws_bethere, n2, obj, pos_jks, n4, obj, '보다', 
                             ['크다', '큽니다', '크고', '크며'], n1, obj, pos_jks, n4, obj, '보다', 
                             ['작다', '작습니다', '작고', '작으며'], n2, obj, pos_jks, n3, obj, '보다', 
                             ['작다', '작습니다', '작고', '작으며', '작다고 할 때'], ['', '크기가'], '가장 큰', obj, 
                             ws_ask_what)
                             
    @classmethod
    def equation(cls, names):
        code = f'''
names = {names}
n1, n2, n3, n4 = names
print(n3)
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nl != 1:
            return None
        else:
            names = lists[0].tolist()
            return cls.solve(names)

#        *********************************** type 9 - 3 *******************************
class P9_3_1(Problem):
    @classmethod
    def description(cls):
        return '숫자들 중 특정 값보다 큰 수들의 개수'

    @classmethod
    def random_values(cls):
        n = np.random.randint(3, 10)
        numbers = np.random.rand(n).tolist()
        th = np.random.rand()
        return numbers, th
    
    @classmethod
    def gen_question(cls, numbers, th):
        str_numbers = []
        N = len(numbers)
        probs = np.random.rand(N)
        ks = np.random.choice([1, 2], N)
        denoms = np.random.randint(2, 101, size=N)
        for i, n in enumerate(numbers):
            if probs[i] < 0.5:
                if ks[i] == 1:
                    str_numbers.append('%.1f'%n)
                else:
                    str_numbers.append('%.2f'%n)
            else:
                d = denoms[i]
                str_numbers.append(str(int(n * d)) + '/' + str(d))
        th = (th * 100 // 1) / 100
        return make_sentence(['', '어떤 수', str(N) + '개의 수'], list2str(str_numbers), pos_jks, ws_bethere, 
                             ['', '이 중에서', '그 중에서', '이 중에', '그 중에'], th, '보다', '큰', ['수는', '숫자는', '것은'], 
                             ask_how_many('개'))


    @classmethod
    def equation(cls, numbers, th):
        code = f'''
numbers = {numbers}
th = {th}
count = 0
for n in numbers:
    if n > th:
        count += 1
print(count)
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn < 1 or nl != 1:
            return None
        else:
            series = lists[0].tolist()
            th = numbers[-1].value()
            return cls.solve(series, th)
    
# -----------------------------------------------------     
    
class P9_3_2(Problem):
    @classmethod
    def description(cls):
        return '숫자들 중 특정 값보다 작은 수들의 개수'

    @classmethod
    def random_values(cls):
        n = np.random.randint(3, 10)
        numbers = np.random.rand(n).tolist()
        th = np.random.rand()
        return numbers, th
    
    @classmethod
    def gen_question(cls, numbers, th):
        str_numbers = []
        N = len(numbers)
        probs = np.random.rand(N)
        ks = np.random.choice([1, 2], N)
        denoms = np.random.randint(2, 101, size=N)
        for i, n in enumerate(numbers):
            if probs[i] < 0.5:
                if ks[i] == 1:
                    str_numbers.append('%.1f'%n)
                else:
                    str_numbers.append('%.2f'%n)
            else:
                d = denoms[i]
                str_numbers.append(str(int(n * d)) + '/' + str(d))
        th = (th * 100 // 1) / 100
        return make_sentence(['', '어떤 수', str(N) + '개의 수'], list2str(str_numbers), pos_jks, ws_bethere, 
                             ['', '이 중에서', '그 중에서', '이 중에', '그 중에'], th, '보다', '작은', ['수는', '숫자는', '것은'], 
                             ask_how_many('개'))


    @classmethod
    def equation(cls, numbers, th):
        code = f'''
numbers = {numbers}
th = {th}
count = 0
for n in numbers:
    if n < th:
        count += 1
print(count)
        '''
        return code
    
    @classmethod
    def try_solve(cls, numbers, variables, formulas, equations, lists, question=''):
        nn, nv, nf, ne, nl = len(numbers), len(variables), len(formulas), len(equations), len(lists)
        if nn < 1 or nl != 1:
            return None
        else:
            series = lists[0].tolist()
            th = numbers[-1].value()
            return cls.solve(series, th)
