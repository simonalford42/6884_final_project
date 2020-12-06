import numpy as np
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamcoder.task import Task
from dreamcoder.domains.scan.scan_primitives import p_dict as p
from dreamcoder.domains.scan.scan_primitives import _pop_left, _pop_right, _solve_smaller, _concat, _apply_fn, _clip_left, _clip_right
from dreamcoder.frontier import FrontierEntry
from dreamcoder.program import Application, Abstraction, Index
from dreamcoder.program import Primitive
from dreamcoder.type import arrow
from dreamcoder.domains.scan.make_tasks import make_tasks, import_data, tstr, tscan_input


def check_solves(task, program):
    for i, ex in enumerate(task.examples):
        inp, out = ex[0][0], ex[1]
        predicted = program(inp)
        if predicted is None:
            return

        if predicted != out:
            print('didnt solve: {}'.format(task.name))
            print('Failed example ' + str(i) + ': input=')
            print(p._input(inp))
            print('output=')
            print(out)
            print('predicted=')
            print(predicted)
            print('predicted mistakes:')
            print(predicted.grid * (predicted.grid != out.grid))
            # assert False, 'did NOT pass!'
            print('Did not pass')
            return
    print('Passed {}'.format(task.name))


def test_twice():
    def solve(i):
        return _apply_fn(lambda x: _concat(x)(x))(_solve_smaller(_clip_right(i)))

    for task in make_tasks(import_data('data/SCAN/twice.txt')):
        print('task: {}'.format(task.examples[0]))
        check_solves(task, solve)



def test_hard():
    while True:
        s = input()
        print(hard_coded_solve(s))


def hard_coded_primitive():
    def solve(inp):
        # print('inp: {}'.format(inp))
        out = hard_coded_solve(inp[1:])
        # print('out: {}'.format(out))
        return out

    return Primitive('hard_code', arrow(tstr, tstr), solve)

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




def add_scan_solutions(frontiers, grammar):
    num_solutions = sum([len(f.entries) > 0 for f in frontiers])
    print('Total solved before augmenting SCAN solutions: {}'.format(num_solutions))

    for frontier in frontiers:
        if len(frontier.entries) > 0:
            program = frontier.entries[0].program
        else:
            program = make_program(frontier.task)

            logPrior = grammar.logLikelihood(frontier.task.request, program)

            frontier_entry = FrontierEntry(program, logPrior=logPrior,
                    logLikelihood=0.0)
            frontier.entries.append(frontier_entry)


def make_program(task):
    output_str = task.examples[0][1]
    output_list = output_str.split(' ')

    base = Index(0)
    for command in output_list[::-1]:
        base = Application(p_dict[command], base)

    program = Abstraction(base)
    return program


class ScanFeatures(nn.Module):
    def __init__(self, tasks, testingTasks=[], cuda=False):
        super().__init__()
        # map word to index 
        input_strs = [t.examples[0][0][0] for t in tasks]
        self.words = list(set(w for input_str in input_strs for w in input_str.split(' ')))
        self.outputDimensionality = len(self.words)
        self


    def featuresOfTask(self, t):
        input_str = t.examples[0][0][0]
        input_words = set(input_str.split(' '))
        vec = torch.tensor([1 if w in input_words else 0 for w in self.words])
        vec = vec.to(torch.float32)
        return vec

class ScanFn():
    def __init__(self, name, *args):
        super().__init__()
        self.name = name
        self.args = args


def parse_intermediate_rep(s): 
    def left(s, word):
        if s.index(word) == 0:
            return ''
        # get rid of space left of splitting word
        return s[0:s.index(word) - 1]

    def right(s, word):
        # get rid of splitting word, and space right of it
        return s[s.index(word) + len(word) + 1:]

    def parse(s):
        # s is the output string
        if 'after' in s:
            return ScanFn('after', 
                    parse(left(s, 'after')),
                    parse(right(s, 'after')))
        elif 'and' in s:
            return ScanFn('and', 
                    parse(left(s, 'and')),
                    parse(right(s, 'and')))
        elif 'thrice' in s:
            return ScanFn('thrice',
                    parse(left(s, 'thrice')))
        elif 'twice' in s:
            return ScanFn('twice',
                    parse(left(s, 'twice')))
        elif 'opposite' in s:
            action = parse(left(s, 'opposite'))
            turn = parse(right(s, 'opposite'))
            return ScanFn('opposite', action, turn)
        elif 'around' in s:
            action = parse(left(s, 'around'))
            turn = parse(right(s, 'around'))
            return ScanFn('around', action, turn)
        elif ' ' not in s:
            return ScanFn(s)
        elif 'left' in s:
            action = parse(left(s, 'left'))
            return ScanFn('left', action)
        elif 'right' in s:
            action = parse(left(s, 'right'))
            return ScanFn('right', action)
        else:
            assert False

    return parse(s)


def eval_rep(rep):
    def concat(*l):
        l = [a for a in l if len(a) > 0]
        return ' '.join(l)

    def get_word(s):
        if s == 'left':
            return 'LTURN'
        elif s == 'right':
            return 'RTURN'
        elif s == 'turn':
            return ''
        else:
            return s.upper()

    if rep.name == 'after':
        return concat(eval_rep(rep.args[1]), eval_rep(rep.args[0]))
    elif rep.name == 'and':
        return concat(eval_rep(rep.args[0]), eval_rep(rep.args[1]))
    elif rep.name == 'thrice':
        a = eval_rep(rep.args[0])
        return concat(a, a, a)
    elif rep.name == 'twice':
        a = eval_rep(rep.args[0])
        return concat(a, a)
    elif rep.name == 'opposite':
        action = eval_rep(rep.args[0])
        turn = eval_rep(rep.args[1])
        return concat(turn, turn, action)
    elif rep.name == 'around':
        action = eval_rep(rep.args[0])
        turn = eval_rep(rep.args[1])
        sub = concat(turn, action)
        return concat(sub, sub, sub, sub)
    elif len(rep.args) == 0:
        return get_word(rep.name)
    elif rep.name == 'left':
        action = eval_rep(rep.args[0])
        return concat('LTURN', action)
    elif rep.name == 'right':
        action = eval_rep(rep.args[0])
        return concat('RTURN', action)
    else:
        assert False


def to_string(inter_rep):
    if len(inter_rep.args) == 0: 
        return inter_rep.name
    else:
        return inter_rep.name + '(' + ', '.join(to_string(arg) for arg in
            inter_rep.args) + ')'


def polish_form(inter_rep):
    if len(inter_rep.args) == 0:
        return inter_rep.name
    else:
        return inter_rep.name + ' ' + ' '.join(polish_form(a) for a in inter_rep.args)


def parse_polish(s):

    def next(s):
        if ' ' not in s:
            return ''
        else:
            return s[s.index(' ')+1:]

    def first_word(s):
        if ' ' not in s:
            return s
        else:
            return s[0:s.index(' ')]

    # if direction follow true, we treat left/right as functions, not constants
    def parse(s, direction_follow=True):
        if s.startswith('after'):
            a1, rest = parse(next(s))
            a2, rest = parse(rest)
            return ScanFn('after', a1, a2), rest
        elif s.startswith('and'):
            a1, rest = parse(next(s))
            a2, rest = parse(rest)
            return ScanFn('and', a1, a2), rest
        elif s.startswith('thrice'):
            a1, rest = parse(next(s))
            return ScanFn('thrice', a1), rest
        elif s.startswith('twice'):
            a1, rest = parse(next(s))
            return ScanFn('twice', a1), rest
        elif s.startswith('opposite'):
            a1, rest = parse(next(s), False)
            a2, rest = parse(rest, False)
            return ScanFn('opposite', a1, a2), rest
        elif s.startswith('around'):
            a1, rest = parse(next(s), False)
            a2, rest = parse(rest, False)
            return ScanFn('around', a1, a2), rest
        elif (s.startswith('run') or s.startswith('jump') 
                or s.startswith('walk') or s.startswith('look')
                or s.startswith('turn')):
            start = first_word(s)
            return ScanFn(start), next(s)
        elif s.startswith('left') or s.startswith('right'):
            if ' ' not in s:
                return ScanFn(s), ''
            s2 = next(s)
            start = first_word(s)
            s3 = first_word(s2)
            if direction_follow and s3 in {'run', 'walk', 'jump', 'look', 'turn'}:
                a1, rest = parse(s2)
                return ScanFn(start, a1), rest
            else:
                return ScanFn(start), next(s)
        else:
            assert False, 'Hm: {}'.format(s)

    result, rest = parse(s)
    assert rest == '', 'Extra {}'.format(rest)
    return result
        
        
def test_intermediate_reps():
    # i = ' '
    # while i != '':
        # i = input()
        # print(to_string(parse_intermediate_rep(i)))
        # print(eval_rep(parse_intermediate_rep(i)))

    tasks = import_data()
    polishes = list(set([polish_form(parse_intermediate_rep(i)) for (i, o) in tasks]))
    assert len(polishes) == len(tasks)

    for (i, o) in tasks:
        inter_rep = parse_intermediate_rep(i)
        string = to_string(inter_rep)
        polish = polish_form(inter_rep)
        parsed_polish = parse_polish(polish)
        output = eval_rep(inter_rep)
        if to_string(parsed_polish) != string:
            assert False

        if output != o:
            assert False


def generate_intermediate_tasks(tasks, path):
    def format_line(i, p, o):
        return 'IN: {} PARSE: {} OUT: {}'.format(i, p, o)

    with open(path, 'w+') as f:
        for (i, o) in tasks:
            inter_rep = parse_intermediate_rep(i)
            string = to_string(inter_rep)
            polish = polish_form(inter_rep)
            f.write(format_line(i, string, polish) + '\n')


def main():
    test_twice()
    generate_intermediate_tasks(import_data(), 'data/SCAN/polish_with_parse.txt')

