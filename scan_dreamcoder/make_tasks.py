from dreamcoder.task import Task
from dreamcoder.type import arrow, baseType, tint
import math

tstr = baseType('str')
tscan_input = baseType('scan_input')

tinput = baseType('input')
tsubexp = baseType('subexp')
tinput_const = baseType('input_const')
tout = baseType('tout')

def import_data(path='data/SCAN/tasks.txt'):
    with open(path, 'r') as f:
        lines = f.readlines()
        # remove \n
        lines = [l[:-1] for l in lines]
        # format: IN: jump  OUT: JUMP
        tasks = []
        for line in lines:
            # put into "normal" format
            line = line.replace('I_', '')
            line = line.replace('TURN_LEFT', 'LTURN')
            line = line.replace('TURN_RIGHT', 'RTURN')

            a = line.index('IN: ') + 4
            b = line.index('OUT: ') - 1
            b2 = b + 6
            i = line[a:b]
            o = line[b2:]
            tasks.append((i, o))

        tasks = sorted(tasks, key=lambda t: len(t[1]))


    return tasks


def factorial_task():
    examples = [((x,), math.factorial(x)) for x in range(0, 10)]
    return Task('factorial', arrow(tint, tint), examples)


def make_single_task(scan_data, name='scan'):

    examples = [((i,), o) for (i, o) in scan_data]
    return Task(name, arrow(tinput, tout), examples)


def make_tasks(scan_data):

    def make_task(input_str, output_str, name):
        # only one example per task
        examples = [((input_str,), output_str)]
        # examples = [((' ' + input_str,), output_str)]
        # return Task(name, arrow(tscan_input, tstr), examples)
        # return Task(name, arrow(tstr, tstr), examples)
        return Task(name, arrow(tinput, tout), examples)

    return [make_task(i, o, str(n) + ': ' + i) for n, (i, o) in enumerate(scan_data)]




