"""Solves an instance.

Modify this file to implement your own solvers.

For usage, run `python3 solve.py --help`.
"""

import argparse
from pathlib import Path
from typing import Callable, Dict

from instance import Instance
from point import Point
from distance import Distance
from solution import Solution
from file_wrappers import StdinFileWrapper, StdoutFileWrapper
import numpy as np


# def within_dist(point, cities, dist):
#     within = []
#     for i in range(len(cities)):
#         if (Point.distance_obj(point, cities[i]) <= dist):
#             within.append(cities[i])
#     return within
            


def solve_naive(instance: Instance) -> Solution:
    return Solution(
        instance=instance,
        towers=instance.cities,
    )

# Goal is to greedily add tower that covers the most cities
def solve_greedy(instance: Instance) -> Solution:
    # Need a way to QUICKLY access point of highest coverage
    # Need to be able to REMOVE a city from ALL point's coverage

    # Graph implementation, nodes == cities/points, edges == connections
    

        


        
    


    

    return Solution(
        instance=instance,
        towers=instance.cities,
    )



SOLVERS: Dict[str, Callable[[Instance], Solution]] = {
    "naive": solve_naive
}


# You shouldn't need to modify anything below this line.
def infile(args):
    if args.input == "-":
        return StdinFileWrapper()

    return Path(args.input).open("r")


def outfile(args):
    if args.output == "-":
        return StdoutFileWrapper()

    return Path(args.output).open("w")


def main(args):
    with infile(args) as f:
        instance = Instance.parse(f.readlines())
        solver = SOLVERS[args.solver]
        solution = solver(instance)
        assert solution.valid()
        with outfile(args) as g:
            print("# Penalty: ", solution.penalty(), file=g)
            solution.serialize(g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve a problem instance.")
    parser.add_argument("input", type=str, help="The input instance file to "
                        "read an instance from. Use - for stdin.")
    parser.add_argument("--solver", required=True, type=str,
                        help="The solver type.", choices=SOLVERS.keys())
    parser.add_argument("output", type=str,
                        help="The output file. Use - for stdout.",
                        default="-")
    main(parser.parse_args())
