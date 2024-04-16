from eama import *

import argparse
import os
import time
import cProfile


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='EAMA VRPTW solver')
    parser.add_argument('problem', type=str, help='Problem file (in Solomon format)')
    return parser.parse_args()

if __name__ == '__main__':
    args = arguments()
    assert os.path.exists(args.problem), "Incorrect problem file"
    problem = SolomonFormatParser(args.problem).get_problem()  
    start_time = time.time()
    rmh_settings = RMHSettings()
    rmh_settings.i_rand = 1000
    rmh_settings.t_max = 10 * 120
    rmh_settings.lower_bound = 6
    gip_settings = GIPSettings()
    eama_settings = EAMASettings()
    eama = EAMA(problem, rmh_settings=rmh_settings, gip_settings=gip_settings, eama_settings=eama_settings, debug=False)
    result = eama.powerful_route_minimization_heuristic(rmh_settings)
    #cProfile.runctx('eama.powerful_route_minimization_heuristic(rmh_settings)', globals=globals(), locals={'rmh_settings': rmh_settings})
    s = result.get_solution()
    print(f'elapsed time: {time.time() - start_time}')
    print(f'routes: {len(s)}')
    with open(f"""solutions/{args.problem.split(os.sep)[-1].split(".")[0]}.sol""", 'w') as f:
        f.write(problem.print_canonical(s))