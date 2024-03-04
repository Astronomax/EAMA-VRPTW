import os
import time
import requests
from pathlib import Path

from eama import *
from validator import *

problem_dir = "./instances/GehringHomberger1000"
solution_dir = "./solutions"

def validate(routes):
    distance = 0
    for route in routes: 
        time = route._customers[0].e
        for source, target in zip(route._customers, route._customers[1:]):
            start_time = max([target.e, time + source.c(target)])
            assert start_time <= target.l
            time = start_time + target.s
            distance += source.c(target)
    return distance


for file in sorted(os.listdir(problem_dir)):
    instance_name = Path(file).stem.lower()
    problem = SolomonFormatParser(f'{problem_dir}/{file}').get_problem()  
    start_time = time.time()
    vehicles, s = EAMA(problem, t_max=10, debug=True).powerful_route_minimization_heuristic()
    print(f'elapsed time: {time.time() - start_time}')
    with open(f"""{solution_dir}/{instance_name}.sol""", 'w') as f:
        f.write(problem.print_canonical(s))
    distance = validate(s)
    url = 'http://212.233.123.219:3000/submit'
    submission = {
        'instance': instance_name,
        'vehicles': vehicles,
        'distance': distance,
        'commit': "9d33a1c772ae9e051ea27572988780464efc8cf0"
    }
    x = requests.post(url, json = submission)
    
    

