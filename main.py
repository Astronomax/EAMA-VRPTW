from eama import *

import argparse
import os
import time


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='EAMA VRPTW solver')
    parser.add_argument('problem', type=str, help='Problem file (in Solomon format)')
    return parser.parse_args()

if __name__ == '__main__':
    #'''
    args = arguments()
    assert os.path.exists(args.problem), "Incorrect problem file"

    problem = SolomonFormatParser(args.problem).get_problem()  
    start_time = time.time()
    rmh_settings = RMHSettings()
    gip_settings = GIPSettings()
    eama_settings = EAMASettings()

    gip_settings.n_pop = 2

    eama = EAMA(problem, rmh_settings=rmh_settings, gip_settings=gip_settings, eama_settings=eama_settings, debug=False)
    s = eama.generate_initial_population()
    
    positions = {c.number: (c.x, c.y) for c in problem.customers}

    p_a = s[0]
    Graph(p_a).plot("p_a.png", positions)
    p_b = s[1]
    Graph(p_b).plot("p_b.png", positions)

    routes = eama.eax(p_a, p_b, strategy='single')#.plot("eax.png", positions
    Graph(routes).plot("eax.png", positions)

    len(set(sum([route.route._customers for route in routes], []))) == len(problem.customers)
    print(f'elapsed time: {time.time() - start_time}')
    #'''
    '''
    customers = [
        Customer(0, 0, 0, 0, 0, 100, 0),
        Customer(1, 1, 1, 0, 0, 100, 0),
        Customer(2, 1, 2, 0, 0, 100, 0),
        Customer(3, 0, 3, 0, 0, 100, 0),
        Customer(4, 0, 2, 0, 0, 100, 0),
    ]
    problem = Problem('my', customers, 100, 100)

    p_b = [PenaltyCalculator()]
    p_b[0].recalc(Route(problem, [customers[4], customers[3], customers[2], customers[1]]))
    p_a = [PenaltyCalculator()]
    p_a[0].recalc(Route(problem, [customers[1], customers[2], customers[4], customers[3]]))

    #positions = {c.number: (c.x, c.y) for c in problem.customers}
    #Graph(p_a).plot("p_a.png")
    #Graph(p_b).plot("p_b.png")

    rmh_settings = RMHSettings()
    gip_settings = GIPSettings()

    eama = EAMA(problem, rmh_settings=RMHSettings(), gip_settings=GIPSettings(), eama_settings=EAMASettings(), debug=False)
    routes = eama.eax(p_a, p_b, strategy='single')

    Graph(routes).plot("eax.png")
    '''
    #with open(f"""solutions/{args.problem.split(os.sep)[-1].split(".")[0]}.sol""", 'w') as f:
    #    f.write(problem.print_canonical(s))
