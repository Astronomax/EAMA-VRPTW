import os

from eama import *

if __name__ == '__main__':
    problem_file = "C108.txt"
    assert os.path.exists(problem_file), "Problem file doesn't exist"
    problem = SolomonFormatParser(problem_file).get_problem()

    m, s = EAMA(problem, debug=False).powerful_route_minimization_heuristic()
    with open(f"""solutions/{problem_file.split(os.sep)[-1].split(".")[0]}.sol""", 'w') as f:
        f.write(problem.print_canonical(s))

    '''
    customers = [
        Customer(0, 0, 0, 1, 0, 100, 0),#0
        Customer(1, 0, 1, 1, 0, 2, 1),  #1
        Customer(2, 0, 2, 1, 2, 4, 1),  #2
        Customer(3, 0, 3, 1, 4, 6, 1),  #3
        Customer(4, 0, 4, 1, 6, 8, 1),  #4
        Customer(5, 0, 1, 1, 0, 2, 1),  #5
        Customer(6, 0, 2, 1, 2, 4, 1),  #6
        Customer(7, 0, 3, 1, 4, 6, 1),  #7
        Customer(8, 0, 4, 1, 6, 8, 1),  #8
    ]
    problem = Problem('test', customers, 10, 10)
    print(EAMA(problem).powerful_route_minimization_heuristic()[0])
    '''