import numpy as np

def initialize_pheromone(n_cities):
    return np.ones((n_cities, n_cities)) / n_cities

def calculate_distance(route, distance_matrix):
    return sum([distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)]) + distance_matrix[route[-1], route[0]]

def select_next_city(pheromone, distances, visited, alpha, beta):
    pheromone = np.copy(pheromone)
    pheromone[visited] = 0
    
    distances = np.copy(distances)
    distances[distances == 0] = 1e-10

    probabilities = pheromone ** alpha * ((1.0 / distances) ** beta)
    
    probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1e10, neginf=1e10)
    
    total = probabilities.sum()
    if total > 0:
        probabilities /= total
    
    if np.all(probabilities == 0):
        unvisited = [i for i in range(len(distances)) if i not in visited]
        return np.random.choice(unvisited)
    
    return np.random.choice(len(distances), p=probabilities)

def generate_route(pheromone, distance_matrix, alpha, beta):
    route = [np.random.randint(0, len(distance_matrix))]
    for _ in range(len(distance_matrix) - 1):
        next_city = select_next_city(pheromone[route[-1]], distance_matrix[route[-1]], route, alpha, beta)
        route.append(next_city)
    return route

def update_pheromone(pheromone, all_routes, decay, n_best):
    sorted_routes = sorted(all_routes, key=lambda x: x[1])
    for route, dist in sorted_routes[:n_best]:
        for i in range(len(route) - 1):
            pheromone[route[i], route[i + 1]] += 1.0 / dist
            pheromone[route[i + 1], route[i]] += 1.0 / dist
    pheromone *= decay
    return pheromone

def ant_colony_optimization(distance_matrix, n_ants, n_best, n_iterations, decay, alpha=1, beta=2):
    pheromone = initialize_pheromone(len(distance_matrix))
    all_time_shortest_path = None
    all_time_shortest_distance = np.inf

    for _ in range(n_iterations):
        all_routes = []
        for _ in range(n_ants):
            route = generate_route(pheromone, distance_matrix, alpha, beta)
            distance = calculate_distance(route, distance_matrix)
            all_routes.append((route, distance))
            if distance < all_time_shortest_distance:
                all_time_shortest_distance = distance
                all_time_shortest_path = route

        pheromone = update_pheromone(pheromone, all_routes, decay, n_best)

    return all_time_shortest_path, all_time_shortest_distance

num_cities = 5
random_matrix = np.random.rand(num_cities, num_cities)
distance_matrix = 20 + random_matrix * (150 - 20)
print(distance_matrix)

best_route, best_distance = ant_colony_optimization(
    distance_matrix,
    n_ants=10,
    n_best=3,
    n_iterations=100,
    decay=0.95,
    alpha=1,
    beta=2
)
print(f"Best Route: {best_route}")
print(f"Best Distance: {best_distance}")
