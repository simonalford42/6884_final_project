import binutil
from dreamcoder.domains.scan.make_tasks import make_tasks, import_data, tstr, tscan_input, make_single_task
from dreamcoder.domains.scan.main import ScanFeatures, test_hard, hard_coded_primitive, test_intermediate_reps, main
from dreamcoder.domains.scan.scan_primitives import primitives, factorial_primitives
from dreamcoder.domains.scan.scan_primitives2 import primitives as primitives2, hard_coded_solve
from dreamcoder.domains.scan.test import main as test_main
from dreamcoder.program import Primitive
from dreamcoder.type import arrow
from dreamcoder.grammar import Grammar
from dreamcoder.dreamcoder import commandlineArguments, ecIterator

word = 'tasks'
training_data = import_data(path='data/SCAN/{}.txt'.format(word))

# training_tasks = [make_single_task(training_data, name=word)]
training_tasks = make_tasks(training_data)

grammar = Grammar.uniform(primitives2)

args = commandlineArguments(
    enumerationTimeout=1,
    iterations=1,
    recognitionTimeout=120,
    # featureExtractor=ScanFeatures,
    # auxiliary=False,
    helmholtzRatio=0.0,
    a=4,
    structurePenalty=0.0,
    aic=0.0,
    maximumFrontier=10,
    topK=1,
    taskReranker='unsolved',
    pseudoCounts=30.0,
    solver='python',
    compressor='ocaml',
        # choices=["pypy","rust","vs","pypy_vs","ocaml"])
    CPUs=1)

generator = ecIterator(grammar,
                       training_tasks,
                       testingTasks=[],
                       # outputPrefix='/experimentOutputs/arc/',
                       **args)

# test_intermediate_reps()
# main()
# test_main()
# test_hard()

for i, result in enumerate(generator):
    print('ecIterator count {}'.format(i))

