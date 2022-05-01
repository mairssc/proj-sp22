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
import random as rand


# Returns whether the point is within the dist of a city
def within_dist(point, city, dist):
    return Point.distance_obj(point, city) <= dist


def basic_penalty(point, penalty_grid):
    return penalty_grid[point.x, point.y]


def space_penalty(penalty_grid, tower, side_len, radius):
    x = tower.x
    y = tower.y
    top = y + radius
    bot = y - radius
    left = x - radius
    right = x + radius
    penalty = 0
    for j in range(bot, top+1):
        for i in range(left, right+1):
            if 0 <= i <= side_len-1 and 0 <= j <= side_len-1 and within_dist(Point(i, j), tower, radius):
                # can change to multiply by a constant
                penalty += penalty_grid[i, j] * 1.1853
    return penalty


def update_penalty(penalty_grid, tower, side_len, radius):
    x = tower.x
    y = tower.y
    top = y + radius
    bot = y - radius
    left = x - radius
    right = x + radius
    for j in range(bot, top+1):
        for i in range(left, right+1):
            if 0 <= i <= side_len-1 and 0 <= j <= side_len-1 and within_dist(Point(i, j), tower, radius):
                # can change to multiply by a constant
                penalty_grid[i, j] *= 1.1853


def update_radius(grid, city, side_len):
    x = city.x
    y = city.y
    top = y + 3
    bot = y - 3
    left = x - 3
    right = x + 3
    output = []
    for j in range(bot, top+1):
        for i in range(left, right+1):
            if 0 <= i <= side_len-1 and 0 <= j <= side_len-1 and within_dist(Point(i, j), city, 3):
                grid[i, j] += 1
                output.append((i, j))
    return output

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
    for k in range(len(hold_cities)):
        city_points[k] += update_radius(grid, hold_cities[k], side_len)
    

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

def solve_random(instance: Instance) -> Solution:
    side_len = instance.D
    hold_cities = instance.cities.copy()
    grid = np.zeros((side_len, side_len))
    # Array of points that reach city k, represented as tuples
    points_reaching_city = [[] for i in range(len(hold_cities))]
    towers = []

    # update grid with 
    for k in range(len(hold_cities)):
        points_reaching_city[k] += update_radius(grid, hold_cities[k], side_len)

    # for i in range(side_len):
    #     for j in range(side_len):
    #         cur_point = Point(i, j)
    #         for k in range(len(hold_cities)):
    #             if within_dist(cur_point, hold_cities[k], instance.coverage_radius):
    #                 # If point (i, j) can reach city k, add point to city_points
    #                 points_reaching_city[k].append((i, j))
    #                 grid[i, j] += 1



    h = hold_cities.copy()
    while len(hold_cities) != 0:
        # Creates np.array of indices of non-zero points
        non_zero_points = np.nonzero(grid)


        # creates upper bound for indices
        upper_index = non_zero_points[0].size-1

        
        # Finds a rand index
        if upper_index != 0:
            cur_index = rand.randint(0, upper_index)
        else:
            cur_index = 0
        

        # Gives indices within grid for chosen random point
        i, j = non_zero_points[0][cur_index], non_zero_points[1][cur_index]

        cur_point = Point(i, j)
        towers.append(cur_point)

        for k in range(len(h)):
            if within_dist(cur_point, h[k], instance.coverage_radius) and h[k] in hold_cities:
                hold_cities.remove(h[k])
                # If removed city, decrement value of points that cover removed city
                for point in points_reaching_city[k]:
                    grid[point[0], point[1]] -= 1

    return Solution(
        instance=instance,
        towers=towers,
    )

# Goal is to greedily add tower that covers the most cities
def solve_greedy2(instance: Instance) -> Solution:
    # Need a way to QUICKLY access point of highest coverage
    # Need to be able to REMOVE a city from ALL point's coverage
    # Given cities, dimensions, radius of tower, etc.

    side_len = instance.D
    hold_cities = instance.cities.copy()

    # points indexed by row, col. number represents cities it covers
    grid = np.zeros((side_len, side_len))
    penalty_grid = np.ones((side_len, side_len))
    city_points = [[] for i in range(len(hold_cities))]
    towers = []
    for k in range(len(hold_cities)):
        city_points[k] += update_radius(grid, hold_cities[k], side_len)

    

    # Remove cities covered by the "max tower" each iteration and update the grid
    h = hold_cities.copy()

    # Look into how to bound this number
    NUM_MAX_POINTS = 10
    # contains NUM_MAX_POINTS number of max_points in the future
    max_points = []

    # Contains grid vals
    grid_values = []
    for i in range(NUM_MAX_POINTS):
        max_points.append(Point(0,0))
        grid_values.append(0)

    # contains the calculated penalties of placing a max_point tower in the penalty grid
    penalties = np.zeros(NUM_MAX_POINTS)
    while len(hold_cities) != 0:
        # CHECK FOR grid == 0 exception 
        for i in range(NUM_MAX_POINTS):
            max_i, max_j = np.unravel_index(np.argmax(grid, axis=None), grid.shape)
            max_point = Point(max_i, max_j)
            if (grid[max_i, max_j] == 0):
                penalties[i] = 1000000000
                max_points[i] = max_point
                grid_values[i] = grid[max_i, max_j]
                continue

            # Choose penalty type here
            # penalties[i] = space_penalty(penalty_grid, max_point, side_len, instance.penalty_radius)
            penalties[i] = basic_penalty(max_point, penalty_grid)


            max_points[i] = max_point
            grid_values[i] = grid[max_i, max_j]

            # update grid to get NEW max_point
            grid[max_i, max_j] = 0

        # opt_point is the point with MINIMUM penalty from the seen points
        opt_point = max_points[np.argmin(penalties)]

        # Reset grid
        for i in range(NUM_MAX_POINTS):
            # get point
            cur_point = max_points[i]

            # update point on original grid
            grid[cur_point.x, cur_point.y] = grid_values[i]

        # update penalty grid
        update_penalty(penalty_grid, opt_point, side_len, instance.penalty_radius)

        towers.append(opt_point)
        # removes covered cities from hold_cities 
        for k in range(len(h)):
            if within_dist(opt_point, h[k], instance.coverage_radius) and h[k] in hold_cities:
                hold_cities.remove(h[k])
                # If removed city, decrement value of points that cover removed city
                for point in city_points[k]:
                    grid[point[0], point[1]] -= 1

    return Solution(
        instance=instance,
        towers=towers,
    )

def solve_greedy3(instance: Instance) -> Solution:
    # Need a way to QUICKLY access point of highest coverage
    # Need to be able to REMOVE a city from ALL point's coverage
    # Given cities, dimensions, radius of tower, etc.

    side_len = instance.D
    hold_cities = instance.cities.copy()

    # points indexed by row, col. number represents cities it covers
    grid = np.zeros((side_len, side_len))
    penalty_grid = np.ones((side_len, side_len))
    city_points = [[] for i in range(len(hold_cities))]
    towers = []
    for k in range(len(hold_cities)):
        city_points[k] += update_radius(grid, hold_cities[k], side_len)

    

    # Remove cities covered by the "max tower" each iteration and update the grid
    h = hold_cities.copy()

    # Look into how to bound this number
    NUM_MAX_POINTS = 20
    # contains NUM_MAX_POINTS number of max_points in the future
    max_points = []

    # Contains grid vals
    grid_values = []
    for i in range(NUM_MAX_POINTS):
        max_points.append(Point(0,0))
        grid_values.append(0)

    # contains the calculated penalties of placing a max_point tower in the penalty grid
    penalties = np.zeros(NUM_MAX_POINTS)
    while len(hold_cities) != 0:
        # CHECK FOR grid == 0 exception 
        for i in range(NUM_MAX_POINTS):
            max_i, max_j = np.unravel_index(np.argmax(grid, axis=None), grid.shape)
            max_point = Point(max_i, max_j)
            if (grid[max_i, max_j] == 0):
                penalties[i] = 1000000000
                max_points[i] = max_point
                grid_values[i] = grid[max_i, max_j]
                continue

            # Choose penalty type here
            # penalties[i] = space_penalty(penalty_grid, max_point, side_len, instance.penalty_radius)
            penalties[i] = basic_penalty(max_point, penalty_grid)


            max_points[i] = max_point
            grid_values[i] = grid[max_i, max_j]

            # update grid to get NEW max_point
            grid[max_i, max_j] = 0

        # opt_point is the point with MINIMUM penalty from the seen points
        opt_point = max_points[np.argmin(penalties)]

        # Reset grid
        for i in range(NUM_MAX_POINTS):
            # get point
            cur_point = max_points[i]

            # update point on original grid
            grid[cur_point.x, cur_point.y] = grid_values[i]

        # update penalty grid
        update_penalty(penalty_grid, opt_point, side_len, instance.penalty_radius)

        towers.append(opt_point)
        # removes covered cities from hold_cities 
        for k in range(len(h)):
            if within_dist(opt_point, h[k], instance.coverage_radius) and h[k] in hold_cities:
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
    "greedy": solve_greedy,
    "random": solve_random,
    "greedy2": solve_greedy2,
    "greedy3": solve_greedy3
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
