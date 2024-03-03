from eama import *

import os
import time


if __name__ == '__main__':
    problem_file = "C108.txt"
    assert os.path.exists(problem_file), "Problem file doesn't exist"
    problem = SolomonFormatParser(problem_file).get_problem()  
    start_time = time.time()
    m, s = EAMA(problem, debug=True).powerful_route_minimization_heuristic()
    print(f'elapsed time: {time.time() - start_time}')
    with open(f"""solutions/{problem_file.split(os.sep)[-1].split(".")[0]}.sol""", 'w') as f:
        f.write(problem.print_canonical(s))