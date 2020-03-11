"""
Created on 2018-08-12

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""

import argparse
from architectural_design.metaheuristics import (
    local_genetic_algorithm_local_nets,
    local_genetic_algorithm_distributed_nets,
    distributed_genetic_algorithm_local_nets)


parser = argparse.ArgumentParser(description='Test')

parser.add_argument('--untrained', dest='untrained',
                    action='store_true', default=False,
                    help='If selected, the networks will not be trained.')

parser.add_argument('-e', dest='experiment', type=int,
                    help=('1-local execution,' +
                          '2-local execution with distributed training,' +
                          '3-distributed execution with local training'),
                    required=True)

args = parser.parse_args()
if args.experiment == 1:
    local_genetic_algorithm_local_nets.main(args.untrained)
elif args.experiment == 2:
    local_genetic_algorithm_distributed_nets.main(args.untrained)
elif args.experiment == 3:
    distributed_genetic_algorithm_local_nets.main(args.untrained)
