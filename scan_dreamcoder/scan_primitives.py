from dreamcoder.domains.scan.make_tasks import make_tasks, import_data, tstr, tscan_input
from dreamcoder.program import Primitive
from dreamcoder.type import arrow, tint, t0, t1, tbool

def root_primitive(word):
    return Primitive('SCAN_' + word, tstr, word)

def root_primitive2(word):
    # pickle doesn't work on lambda functions, so have to make separate
    def _root_fn(s):
        # we eventually operate on the input string, which starts with space.
        return word if s[0] == ' ' else word + ' ' + s

    return Primitive('SCAN_' + word, arrow(tstr, tstr), _root_fn)

def _solve_smaller(word):
    # if is of the form "jump" or "jump left", etc. spits out the answer. 
    # otherwise raises error
    if ' ' not in word:
        assert word in ['jump', 'run', 'look', 'walk']
        return word.upper()
    
    a, b = word.split(' ')
    assert a in ['jump', 'run', 'look', 'walk', 'turn']
    assert b in ['left', 'right']
    if b == 'left':
        c = 'LTURN'
    else:
        c = 'RTURN'
    if a == 'turn':
        return c
    else:
        return c + ' ' + a.upper()



primitive_solve_smaller = Primitive('solve_smaller', arrow(tstr, tstr),
        _solve_smaller)

def _concat(a):
    return lambda b: a + ' ' + b

primitive_concat = Primitive("concat", arrow(tstr, tstr, tstr), _concat)


def _left(s):
    def left(s, w):
        if s.index(w) == 0:
            return ''
        # get rid of space left of splitting word
        return s[0:s.index(w) - 1]

    return lambda w: left(s, w)

def _right(s):
    def right(s, w):
        # get rid of splitting word, and space right of it
        return s[s.index(w) + len(w) + 1:]

    return lambda w: right(s, w)


def _apply_fn(f):
    def apply(f, x):
        return f(x)

    return lambda x: f(x)


def _pop_left(s):
    if ' ' not in s:
        return ''
    return s[0:s.index(' ')]

def _clip_left(s):
    if ' ' not in s:
        return ''
    return s[s.index(' ')+1:]

def _clip_right(s):
    if ' ' not in s:
        return ''
    return s[0:s.rindex(' ')]

def _pop_right(s):
    if ' ' not in s:
        return ''
    return s[s.rindex(' ')+1:]

primitive_pop_left = Primitive("pop_left", arrow(tstr, tstr), _pop_left)
primitive_pop_right = Primitive("pop_right", arrow(tstr, tstr), _pop_right)
primitive_clip_left = Primitive("clip_left", arrow(tstr, tstr), _clip_left)
primitive_clip_right = Primitive("clip_right", arrow(tstr, tstr), _clip_right)

# primitive_left = Primitive("left", arrow(tstr, tstr, tstr), _left)
# primitive_right = Primitive("right", arrow(tstr, tstr, tstr), _right)

primitive_apply_fn = Primitive("apply_fn2", arrow(arrow(t0, t1), t0, t1),
        _apply_fn)


words = ['LTURN', 'RTURN', 'WALK', 'JUMP', 'LOOK', 'RUN']
p_dict = {word: root_primitive2(word) for word in words}
# p_dict['endl'] = endl

primitives = p_dict.values()

primitives = [primitive_clip_left, primitive_clip_right,
        primitive_pop_left, primitive_pop_right, 
        primitive_concat, primitive_solve_smaller] #, primitive_apply_fn]

input_words = ['jump', 'run', 'walk', 'look', 'turn', 'twice', 'thrice',
'around', 'and', 'after', 'opposite']

# primitives = primitives + [Primitive(w, tstr, w) for w in input_words]

# primitives = [endl] + [root_primitive2(w) for w in words]




class RecursionDepthExceeded(Exception):
    pass

def _fix(argument):
    def inner(body):
        recursion_limit = [20]

        def fix(x):
            def r(z):
                recursion_limit[0] -= 1
                if recursion_limit[0] <= 0:
                    raise RecursionDepthExceeded()
                else:
                    return fix(z)

        # Y combinator: lambda f: (lambda x: (x x)) lambda x: f (x x))

        # lambda argument: lambda body: body(

            # lambda body: body(
            return body(r)(x)

        return fix(argument)

    return inner

def _fix2(argument):
    def fix(x):
        return body(fix)(x)

    return lambda body: fix(body)

# so body has to be a special function like almost-factorial which lazily
# evaluates. body takes in arrow(t0, t1) and t0, and returns t1. arrow(t0, 1) is
# the Y combinator?

# Apparently body(fix)(x) has type arrow(t0, t1)? Apparently fix has this type?
# x is the t0. For factorial, t0 and t1 are integer. Yeah, I guess fix is
# arrow(t0, t1). body could be: lambda f: lambda in: if in == 0 then 0 else
# f(n-1)



# intended program: lambda fix1 $0 fact
primitiveRecursion1 = Primitive("fix1",
                                arrow(t0,
                                      arrow(arrow(t0, t1), t0, t1),
                                      t1),
                                _fix)

primitiveRecursion2 = Primitive("fix2",
                                arrow(tint,
                                      arrow(arrow(tint, tint), tint, tint),
                                      tint),
                                _fix)



def _ite(cond):
    def if_then_else(cond, t_result, f_result):
        if cond:
            return t_result
        else:
            # abstracted so that evaluation is lazy
            return f_result()
    return lambda t_result: lambda f_result: if_then_else(cond, t_result,
            f_result)

def _equals_zero(n): return n == 0

def _mult(a): return lambda b: a * b

def _decr(a): return a - 1




def _factorial(f):
    def fact(n):
        # print(n)
        if n == 0:
            return 1
        else:
            # print('recurse')
            return n * f(n-1)

    return fact

primitiveFactorial = Primitive('fact', arrow(arrow(tint, tint), tint, tint),
        _factorial)

def factorial_solution2(n):
    # factorial = lambda f: lambda n: _ite(_equals_zero(n), 1, _mult(n,
        # f(_decr(n))))

    def factorial(f):
        def fact(n):
            a = _equals_zero(n)

            # if a:
                # return 1
            # else:
                # return _mult(n)(f(_decr(n)))
            # return _ite(a)(1)(lambda: _mult(n)(f(_decr(n))))
            # return _ite(_equals_zero(n))(1)(lambda: _fact_sub(n)(f))
            return _fact_sub2(n)(lambda: _fact_sub(n)(f))

        return fact

    return _fix(n)(factorial)

def _fact_sub(n):
    return lambda f: _mult(n)(f(_decr(n)))

def _fact_sub2(n):
    return lambda x: _ite(_equals_zero(n))(1)(x)


primitive_fact_sub = Primitive('fact_sub', arrow(tint, arrow(tint, tint), tint), _fact_sub)

primitive_fact_sub2 = Primitive('fact_sub2', arrow(tint, arrow(tint), tint), _fact_sub2)



solution_primitive = Primitive('sol', arrow(tint, tint), factorial_solution2)

primitive_ite = Primitive('ite', arrow(tbool, t0, arrow(t0), t0), _ite)
primitive_equals_zero = Primitive('eq0?', arrow(tint, tbool), _equals_zero)
primitive_mult = Primitive('mult', arrow(tint, tint, tint), _mult)
primitive_decr = Primitive('decr', arrow(tint, tint), _decr)
primitive_one = Primitive('1', tint, 1)

factorial_primitives = [primitiveRecursion1, primitive_ite,
        primitive_equals_zero, primitive_one, primitive_fact_sub] #primitive_mult, primitive_decr, primitive_one]
        # solution_primitive]
# factorial_primitives = [primitiveRecursion1, primitiveFactorial]

factorial_primitives = [primitiveRecursion1, primitive_fact_sub2,
        primitive_fact_sub]
# factorial_primitives = [solution_primitive]


def factorial_solution(n):
    return _fix(n)(_factorial)


def test_ite(n):
    print(n)
    print(n == 0)
    if n < -10: assert False
    return _ite(n == 0)(1)(n*test_ite(n-1))


