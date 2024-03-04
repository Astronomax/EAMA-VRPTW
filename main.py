from eama import *

import argparse
import os
import time


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='EAMA VRPTW solver')
    parser.add_argument('problem', type=str, help='Problem file (in Solomon format)')
    return parser.parse_args()

if __name__ == '__main__':
    args = arguments()
    assert os.path.exists(args.problem), "Incorrect problem file"

    problem = SolomonFormatParser(args.problem).get_problem()  
    start_time = time.time()
    m, s = EAMA(problem, debug=False).powerful_route_minimization_heuristic()
    print(f'elapsed time: {time.time() - start_time}')
    with open(f"""solutions/{args.problem.split(os.sep)[-1].split(".")[0]}.sol""", 'w') as f:
        f.write(problem.print_canonical(s))