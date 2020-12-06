from dreamcoder.domains.scan.make_tasks import make_tasks, import_data, tinput, tsubexp, tinput_const, tout
from dreamcoder.program import Primitive
from dreamcoder.type import arrow, tint, t0, t1, tbool, baseType


def hard_coded_solve(s):
    def left(s, word):
        if s.index(word) == 0:
            return ''
        # get rid of space left of splitting word
        return s[0:s.index(word) - 1]
    def right(s, word):
        # get rid of splitting word, and space right of it
        return s[s.index(word) + len(word) + 1:]

    def concat(a, b): 
        if a == '': return b
        elif b == '': return a
        else: return a + ' ' + b

    def get_word(s):
        if s == 'left':
            return 'LTURN'
        elif s == 'right':
            return 'RTURN'
        else:
            return s.upper()

    def translate(s):
        if 'after' in s:
            return concat(translate(right(s, 'after')), translate(left(s, 'after')))
        elif 'and' in s:
            return concat(translate(left(s, 'and')), translate(right(s, 'and')))
        elif 'thrice' in s:
            sub = translate(left(s, 'thrice'))
            return concat(sub, concat(sub, sub))
        elif 'twice' in s:
            sub = translate(left(s, 'twice'))
            return concat(sub, sub)
        elif 'opposite' in s:
            turn = translate(right(s, 'opposite'))
            action = translate(left(s, 'opposite'))
            return concat(turn, concat(turn, action))
        elif 'around' in s:
            sub = concat(translate(right(s, 'around')), translate(left(s, 'around')))
            return concat(sub, concat(sub, concat(sub, sub)))
        elif 'turn' in s:
            return translate(right(s, 'turn'))
        elif 'left' in s:
            return concat('LTURN', translate(left(s, 'left')))
        elif 'right' in s:
            return concat('RTURN', translate(left(s, 'right')))
        else:
            return get_word(s)

    return translate(s)

def _translate(input_str): return hard_coded_solve(input_str)

def _left(s):
    def left(s, word):
        if s.index(word) == 0:
            return ''
        # get rid of space left of splitting word
        return s[0:s.index(word) - 1]

    return lambda word: left(s, word)


def _right(s):
    def right(s, word):
        # get rid of splitting word, and space right of it
        return s[s.index(word) + len(word) + 1:]

    return lambda word: right(s, word)

def concat(a, b): 
    if a == '': return b
    elif b == '': return a
    else: return a + ' ' + b

def _concat(a):
    return lambda b: concat(a, b)

def _concat3(a):
    return lambda b: lambda c: concat(concat(a, b), c)

def _concat4(a):
    return lambda b: lambda c: lambda d: concat(concat(concat(a, b), c), d)

def _apply_fn(f):
    def apply(f, x):
        return f(x)

    return lambda x: f(x)

def _around(x):
    def around(x, y):
        z = _concat(x)(y)
        z = _concat4(z)(z)(z)(z)
        return z

    return lambda y: around(x, y)

input_words = ['jump', 'run', 'walk', 'look', 'turn', 'twice', 'thrice',
'around', 'and', 'after', 'opposite']
input_words2 = ['turn', 'twice', 'thrice', 'around', 'and', 'after', 'opposite']

p_input_consts = [Primitive(w, tinput_const, w) for w in input_words2]
p_translate = Primitive('translate', arrow(tsubexp, tout), _translate)
p_left = Primitive('left', arrow(tinput, tinput_const, tsubexp), _left)
p_right = Primitive('right', arrow(tinput, tinput_const, tsubexp), _right)
p_concat = Primitive('concat', arrow(tout, tout, tout), _concat)
p_concat3 = Primitive('concat3', arrow(tout, tout, tout, tout), _concat3)
p_concat4 = Primitive('concat4', arrow(tout, tout, tout, tout, tout), _concat4)
p_repeat4 = Primitive('repeat4', arrow(tout, tout), lambda x: _concat4(x)(x)(x)(x))
p_apply_fn = Primitive("apply_fn", arrow(arrow(tout, tout), tout, tout), _apply_fn)

# primitives = [p_translate, p_left, p_right, p_concat, p_concat3, p_concat4, p_apply_fn] + p_input_consts
primitives = [p_translate, p_left, p_right, p_concat, p_apply_fn] + p_input_consts
