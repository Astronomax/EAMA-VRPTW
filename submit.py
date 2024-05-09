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


instances = [
    "C1_10_1.TXT",
    "C1_10_2.TXT",
    "C1_10_3.TXT",
    "C1_10_4.TXT",
    "C1_10_5.TXT",
    "C1_10_6.TXT",
    "C1_10_7.TXT",
    "C1_10_8.TXT",
    "C1_10_9.TXT",
    "C1_10_10.TXT",
    "C2_10_1.TXT",
    "C2_10_2.TXT",
    "C2_10_3.TXT",
    "C2_10_4.TXT",
    "C2_10_5.TXT",
    "C2_10_6.TXT",
    "C2_10_7.TXT",
    "C2_10_8.TXT",
    "C2_10_9.TXT",
    "C2_10_10.TXT",
    "R1_10_1.TXT",
    "R1_10_2.TXT",
    "R1_10_3.TXT",
    "R1_10_4.TXT",
    "R1_10_5.TXT",
    "R1_10_6.TXT",
    "R1_10_7.TXT",
    "R1_10_8.TXT",
    "R1_10_9.TXT",
    "R1_10_10.TXT",
    "R2_10_1.TXT",
    "R2_10_2.TXT",
    "R2_10_3.TXT",
    "R2_10_4.TXT",
    "R2_10_5.TXT",
    "R2_10_6.TXT",
    "R2_10_7.TXT",
    "R2_10_8.TXT",
    "R2_10_9.TXT",
    "R2_10_10.TXT",
    "RC1_10_1.TXT",
    "RC1_10_2.TXT",
    "RC1_10_3.TXT",
    "RC1_10_4.TXT",
    "RC1_10_5.TXT",
    "RC1_10_6.TXT",
    "RC1_10_7.TXT",
    "RC1_10_8.TXT",
    "RC1_10_9.TXT",
    "RC1_10_10.TXT",
    "RC2_10_1.TXT",
    "RC2_10_2.TXT",
    "RC2_10_3.TXT",
    "RC2_10_4.TXT",
    "RC2_10_5.TXT",
    "RC2_10_6.TXT",
    "RC2_10_7.TXT",
    "RC2_10_8.TXT",
    "RC2_10_9.TXT",
    "RC2_10_10.TXT",
]

for file_instance in instances:
    problem = SolomonFormatParser(f'{problem_dir}/{file_instance}').get_problem()  
    rmh_settings = RMHSettings()
    rmh_settings.i_rand = 1000
    rmh_settings.t_max = 10 * 60
    gip_settings = GIPSettings()
    eama_settings = EAMASettings()
    eama = EAMA(problem, rmh_settings=rmh_settings, gip_settings=gip_settings, eama_settings=eama_settings, debug=False)
    start_time = time.time()
    result = eama.powerful_route_minimization_heuristic(rmh_settings).get_solution()
    print(problem.name, len(result))
    print(f'elapsed time: {time.time() - start_time}')
    with open(f"""{solution_dir}/{problem.name}.sol""", 'w') as f:
        f.write(problem.print_canonical(result))





'''
for file in sorted(os.listdir(problem_dir)):
    instance_name = Path(file).stem.lower()
    problem = SolomonFormatParser(f'{problem_dir}/{file}').get_problem()  
    start_time = time.time()
    vehicles, s = EAMA(problem, t_max=3600, debug=True).powerful_route_minimization_heuristic()
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
'''
