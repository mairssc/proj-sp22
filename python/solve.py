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


# Returns whether the point is within the dist of a city
def within_dist(point, city, dist):
    return Point.distance_obj(point, city) <= dist




def solve_naive(instance: Instance) -> Solution:
    return Solution(
        instance=instance,
        towers=instance.cities,
    )

# Goal is to greedily add tower that covers the most cities
def solve_greedy(instance: Instance) -> Solution:
    # Need a way to QUICKLY access point of highest coverage
    # Need to be able to REMOVE a city from ALL point's coverage
    # Given cities, dimensions, radius of tower, etc.

    side_len = instance.D
    hold_cities = instance.cities.copy()

    # points indexed by row, col. number represents cities it covers
    grid = np.zeros((side_len, side_len))
    city_points = [[] for i in range(len(hold_cities))]
    towers = []
    for i in range(side_len):
        for j in range(side_len):
            cur_point = Point(i, j)
            for k in range(len(hold_cities)):
                if within_dist(cur_point, hold_cities[k], instance.coverage_radius):
                    # If point (i, j) can reach city k, add point to city_points
                    city_points[k].append((i, j))
                    grid[i, j] += 1
    

    # Remove cities covered by the "max tower" each iteration and update the grid
    h = hold_cities.copy()
    while len(hold_cities) != 0:
        max_i, max_j = np.unravel_index(np.argmax(grid, axis=None), grid.shape)
        max_point = Point(max_i, max_j)

        towers.append(max_point)
        # removes covered cities from hold_cities 
        for k in range(len(h)):
            if within_dist(max_point, h[k], instance.coverage_radius) and h[k] in hold_cities:
                hold_cities.remove(h[k])
                # If removed city, decrement value of points that cover removed city
                for point in city_points[k]:
                    grid[point[0], point[1]] -= 1

    return Solution(
        instance=instance,
        towers=towers,
    )



SOLVERS: Dict[str, Callable[[Instance], Solution]] = {
    "naive": solve_naive,
    "greedy": solve_greedy
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
