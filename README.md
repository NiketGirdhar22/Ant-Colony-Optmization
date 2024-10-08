# Ant-Colony-Optimization
This repository contains a Python implementation of the Ant Colony Optimization (ACO) algorithm to solve the Traveling Salesman Problem (TSP).

# Overview
The Traveling Salesman Problem (TSP) is a classic combinatorial optimization problem that aims to find the shortest possible route visiting each city exactly once and returning to the origin city. This implementation uses the Ant Colony Optimization (ACO) algorithm, a nature-inspired metaheuristic based on the foraging behavior of ants, to find an approximate solution to the TSP.

# Usage
The main script aco.py runs the ACO algorithm to solve the TSP. You can modify parameters such as the number of ants, the number of iterations, and the pheromone decay rate to see how they affect the solution.

# Functions
initialize_pheromone(n_cities): Initializes the pheromone matrix with small values.

calculate_distance(route, distance_matrix): Calculates the total distance of a given route based on the distance matrix.

select_next_city(pheromone, distances, visited, alpha, beta): Selects the next city to visit based on the pheromone levels and the distances, using the parameters alpha (pheromone influence) and beta (distance influence).

generate_route(pheromone, distance_matrix, alpha, beta): Generates a route for an ant by probabilistically selecting the next city to visit based on the pheromone matrix.

update_pheromone(pheromone, all_routes, decay, n_best): Updates the pheromone levels on the routes taken by the ants, with a decay factor to simulate the evaporation of pheromones over time.

ant_colony_optimization(distance_matrix, n_ants, n_best, n_iterations, decay, alpha=1, beta=2): Main function to solve the TSP using ACO.

# How It Works
Initialization: The algorithm starts by initializing the pheromone matrix and placing each ant on a randomly chosen city.

Route Construction: Each ant builds a route by choosing the next city based on a probabilistic rule that favors shorter paths and higher pheromone levels.

Pheromone Update: After all ants have completed their tours, the pheromone levels are updated based on the quality of the solutions found (shorter paths get higher pheromone deposits).

Iteration: The process is repeated for a set number of iterations, or until a satisfactory solution is found.

# Customization
You can tweak the following parameters to optimize the ACO algorithm for your specific problem:

n_ants: Number of ants (solutions) per iteration.
n_best: Number of best solutions used to update the pheromone matrix.
n_iterations: Number of iterations to run the algorithm.
decay: Pheromone decay factor to simulate evaporation.
alpha: Controls the influence of the pheromone levels on the decision-making process.
beta: Controls the influence of the distance to the next city on the decision-making process.