from dreamcoder.domains.scan.make_tasks import make_tasks, import_data, tstr, tscan_input, factorial_task
from dreamcoder.domains.scan.main import ScanFeatures, test_hard, hard_coded_primitive
from dreamcoder.domains.scan.scan_primitives import primitives, factorial_primitives, factorial_solution, test_ite
from dreamcoder.program import Primitive
from dreamcoder.type import arrow
from dreamcoder.grammar import Grammar
from dreamcoder.dreamcoder import commandlineArguments, ecIterator

def main():

    # print(test_ite(0))
    # print(test_ite(1))
    # print(test_ite(2))
    # print(test_ite(3))
    # assert False

    training_tasks = [factorial_task()]

    primitives = factorial_primitives

    grammar = Grammar.uniform(primitives)

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

    for i, result in enumerate(generator):
        print('ecIterator count {}'.format(i))

