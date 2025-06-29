import pandas as pd
import numpy as np
from collections import defaultdict
from geopy.distance import geodesic
from sklearn.preprocessing import MinMaxScaler
import logging
from .som_cluster import SOMCluster
from typing import List, Dict, Tuple, Optional
import random
from itertools import permutations, combinations
import time

logger = logging.getLogger(__name__)

class RouteOptimizer:
    """
    Enhanced route optimization engine using multiple algorithms:
    - SOM clustering
    - Genetic Algorithm
    - Ant Colony Optimization
    - Improved Greedy algorithms
    """
    
    def __init__(self, office_location, max_passengers=4, min_passengers=3, cost_per_km=2.5):
        """
        Initialize enhanced route optimizer
        
        Args:
            office_location (dict): Office coordinates {'lat': float, 'lon': float}
            max_passengers (int): Maximum passengers per route
            min_passengers (int): Minimum passengers per route
            cost_per_km (float): Cost per kilometer
        """
        self.office_location = office_location
        self.MAX_PASSENGERS = max_passengers
        self.MIN_PASSENGERS = min_passengers
        self.COST_PER_KM = cost_per_km
        self.scaler = MinMaxScaler()
        
    def validate_staff_data(self, df):
        """
        Validate and clean staff location data
        
        Args:
            df (pandas.DataFrame): Staff data with required columns
            
        Returns:
            pandas.DataFrame: Cleaned and validated data
        """
        logger.info(f"Validating staff data with {len(df)} records")
        try:
            # Check required columns
            required_columns = ['staff_id', 'name', 'latitude', 'longitude', 'address']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Create a copy to avoid modifying the original dataframe
            clean_df = df.copy()
            
            # Convert coordinates to numeric values
            clean_df['latitude'] = pd.to_numeric(clean_df['latitude'], errors='coerce')
            clean_df['longitude'] = pd.to_numeric(clean_df['longitude'], errors='coerce')
            
            # Remove rows with invalid coordinates
            invalid_coords = clean_df[clean_df['latitude'].isna() | clean_df['longitude'].isna()]
            if not invalid_coords.empty:
                logger.warning(f"Removed {len(invalid_coords)} entries with invalid coordinates")
                clean_df = clean_df.dropna(subset=['latitude', 'longitude'])
            
            # Validate coordinate ranges for Ghana
            valid_lat_range = (4.5, 11.5)  # Ghana's latitude range
            valid_lon_range = (-3.5, 1.5)  # Ghana's longitude range
            
            # Create mask for valid coordinates
            coord_mask = (
                (clean_df['latitude'].between(*valid_lat_range)) &
                (clean_df['longitude'].between(*valid_lon_range))
            )
            
            invalid_range = clean_df[~coord_mask]
            if not invalid_range.empty:
                logger.warning(f"Removed {len(invalid_range)} entries with coordinates outside Ghana")
                clean_df = clean_df[coord_mask]
            
            # Remove duplicates based on staff_id
            duplicates = clean_df[clean_df.duplicated(subset=['staff_id'], keep='first')]
            if not duplicates.empty:
                logger.warning(f"Removed {len(duplicates)} duplicate staff entries")
                clean_df = clean_df.drop_duplicates(subset=['staff_id'], keep='first')
            
            # Validate minimum required staff
            if len(clean_df) < self.MIN_PASSENGERS:
                logger.error(f"Need at least {self.MIN_PASSENGERS} valid staff locations, but only found {len(clean_df)}")
                raise ValueError(
                    f"Need at least {self.MIN_PASSENGERS} valid staff locations, "
                    f"but only found {len(clean_df)}"
                )
            
            # Validate data types
            clean_df['staff_id'] = clean_df['staff_id'].astype(str)
            clean_df['name'] = clean_df['name'].astype(str)
            clean_df['address'] = clean_df['address'].astype(str)
            
            # Add distance to office column
            clean_df['distance_to_office'] = clean_df.apply(
                lambda row: geodesic(
                    (row['latitude'], row['longitude']),
                    (self.office_location['lat'], self.office_location['lon'])
                ).km,
                axis=1
            )
            
            logger.info(f"Successfully validated {len(clean_df)} staff records")
            return clean_df
            
        except Exception as e:
            logger.error(f"Error validating staff data: {str(e)}")
            return None

    def create_clusters(self, staff_data, grid_size=3, sigma=1.0, learning_rate=0.5, 
                       initialization='pca', random_state=42):
        """
        Create clusters based on staff locations using enhanced SOM
        
        Args:
            staff_data (pandas.DataFrame): Staff location data
            grid_size (int): SOM grid size
            sigma (float): SOM neighborhood radius
            learning_rate (float): SOM learning rate
            initialization (str): SOM initialization method
            random_state (int): Random seed for reproducibility
            
        Returns:
            pandas.DataFrame: Staff data with cluster assignments
        """
        logger.info(f"Creating enhanced clusters with grid_size={grid_size}, sigma={sigma}, "
                   f"learning_rate={learning_rate}, init={initialization}")
        if staff_data is None or len(staff_data) == 0:
            logger.warning("No staff data provided for clustering")
            return None
        
        try:
            # Calculate distances to office
            staff_data['distance_to_office'] = staff_data.apply(
                lambda row: geodesic(
                    (row['latitude'], row['longitude']),
                    (self.office_location['lat'], self.office_location['lon'])
                ).km,
                axis=1
            )
            
            # Prepare data for clustering
            locations = staff_data[['latitude', 'longitude', 'distance_to_office']].values
            normalized_data = self.scaler.fit_transform(locations)
            
            # Initialize and train enhanced SOM
            som = SOMCluster(
                input_len=3,
                grid_size=grid_size,
                sigma=sigma,
                learning_rate=learning_rate,
                initialization=initialization,
                random_state=random_state
            )
            
            som.train(normalized_data, epochs=3000, verbose=True)
            
            # Assign clusters
            staff_data['cluster'] = [som.get_cluster(loc) for loc in normalized_data]
            
            # Handle small clusters
            self._handle_small_clusters(staff_data)
            
            # Calculate clustering quality metrics
            quality_metrics = som.get_cluster_quality_metrics(normalized_data)
            logger.info(f"Clustering quality metrics: {quality_metrics}")
            
            cluster_counts = staff_data['cluster'].value_counts()
            logger.info(f"Enhanced clustering completed. Cluster distribution: {cluster_counts.to_dict()}")
            
            return staff_data
            
        except Exception as e:
            logger.error(f"Error in enhanced clustering: {str(e)}")
            return None

    def _handle_small_clusters(self, staff_data):
        """
        Handle clusters with fewer than minimum required passengers
        
        Args:
            staff_data (pandas.DataFrame): Staff data with cluster assignments
        """
        logger.info("Handling small clusters")
        cluster_sizes = staff_data['cluster'].value_counts()
        small_clusters = cluster_sizes[cluster_sizes < self.MIN_PASSENGERS].index
        
        if len(small_clusters) > 0:
            logger.info(f"Found {len(small_clusters)} small clusters to handle")
            for small_cluster in small_clusters:
                small_cluster_points = staff_data[staff_data['cluster'] == small_cluster]
                
                for idx, row in small_cluster_points.iterrows():
                    distances = []
                    for cluster_id in staff_data['cluster'].unique():
                        if cluster_id not in small_clusters:
                            cluster_points = staff_data[staff_data['cluster'] == cluster_id]
                            if not cluster_points.empty:
                                avg_dist = cluster_points.apply(
                                    lambda x: geodesic(
                                        (row['latitude'], row['longitude']),
                                        (x['latitude'], x['longitude'])
                                    ).km,
                                    axis=1
                                ).mean()
                                distances.append((cluster_id, avg_dist))
                    
                    if distances:
                        # Assign to nearest cluster
                        best_cluster = min(distances, key=lambda x: x[1])[0]
                        staff_data.loc[idx, 'cluster'] = best_cluster
                        logger.debug(f"Reassigned staff {row['name']} from cluster {small_cluster} to {best_cluster}")

    def optimize_routes(self, staff_data, algorithm='genetic', **kwargs):
        """
        Optimize routes using multiple algorithms
        
        Args:
            staff_data (pandas.DataFrame): Staff data with cluster assignments
            algorithm (str): Algorithm to use ('genetic', 'ant_colony', 'greedy', 'hybrid')
            **kwargs: Additional parameters for specific algorithms
            
        Returns:
            dict: Optimized routes
        """
        logger.info(f"Starting route optimization using {algorithm} algorithm")
        
        if algorithm == 'genetic':
            return self._genetic_algorithm_optimization(staff_data, **kwargs)
        elif algorithm == 'ant_colony':
            return self._ant_colony_optimization(staff_data, **kwargs)
        elif algorithm == 'greedy':
            return self._improved_greedy_optimization(staff_data, **kwargs)
        elif algorithm == 'hybrid':
            return self._hybrid_optimization(staff_data, **kwargs)
        else:
            logger.warning(f"Unknown algorithm {algorithm}, falling back to greedy")
            return self._improved_greedy_optimization(staff_data, **kwargs)

    def _genetic_algorithm_optimization(self, staff_data, population_size=50, 
                                      generations=100, mutation_rate=0.1, 
                                      crossover_rate=0.8, elite_size=5):
        """
        Genetic Algorithm for route optimization
        
        Args:
            staff_data (pandas.DataFrame): Staff data
            population_size (int): Size of the population
            generations (int): Number of generations
            mutation_rate (float): Mutation probability
            crossover_rate (float): Crossover probability
            elite_size (int): Number of elite individuals to preserve
            
        Returns:
            dict: Optimized routes
        """
        logger.info(f"Starting Genetic Algorithm optimization with population_size={population_size}, "
                   f"generations={generations}")
        
        # Convert staff data to list of dictionaries
        staff_list = staff_data.to_dict('records')
        
        # Initialize population
        population = self._initialize_genetic_population(staff_list, population_size)
        
        best_fitness_history = []
        
        for generation in range(generations):
            # Calculate fitness for all individuals
            fitness_scores = [(individual, self._calculate_genetic_fitness(individual)) 
                             for individual in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)  # Higher fitness is better
            
            # Store best fitness
            best_fitness_history.append(fitness_scores[0][1])
            
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: Best fitness = {fitness_scores[0][1]:.4f}")
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            for i in range(elite_size):
                new_population.append(fitness_scores[i][0])
            
            # Generate rest of population through crossover and mutation
            while len(new_population) < population_size:
                # Selection
                parent1 = self._tournament_selection(fitness_scores)
                parent2 = self._tournament_selection(fitness_scores)
                
                # Crossover
                if random.random() < crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if random.random() < mutation_rate:
                    child1 = self._mutate(child1)
                if random.random() < mutation_rate:
                    child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Ensure population size
            population = new_population[:population_size]
        
        # Return best solution
        best_individual = fitness_scores[0][0]
        return self._convert_genetic_solution_to_routes(best_individual)

    def _initialize_genetic_population(self, staff_list, population_size):
        """Initialize genetic algorithm population"""
        population = []
        for _ in range(population_size):
            # Create random route assignments
            individual = []
            remaining_staff = staff_list.copy()
            
            while remaining_staff:
                # Handle case when remaining staff is less than minimum required
                if len(remaining_staff) < self.MIN_PASSENGERS:
                    # Add remaining staff to the last route if it exists and has space
                    if individual and len(individual[-1]) + len(remaining_staff) <= self.MAX_PASSENGERS:
                        individual[-1].extend(remaining_staff)
                    else:
                        # Create a new route with remaining staff (will be handled by validation)
                        individual.append(remaining_staff)
                    break
                
                # Randomly select route size between min and max passengers
                max_route_size = min(self.MAX_PASSENGERS, len(remaining_staff))
                if max_route_size >= self.MIN_PASSENGERS:
                    route_size = random.randint(self.MIN_PASSENGERS, max_route_size)
                else:
                    route_size = len(remaining_staff)
                
                # Randomly select staff for this route
                route_staff = random.sample(remaining_staff, route_size)
                individual.append(route_staff)
                
                # Remove selected staff from remaining
                for staff in route_staff:
                    remaining_staff.remove(staff)
            
            population.append(individual)
        
        return population

    def _calculate_genetic_fitness(self, individual):
        """Calculate fitness score for genetic algorithm individual"""
        total_distance = 0
        total_cost = 0
        route_count = len(individual)
        
        for route in individual:
            if len(route) < self.MIN_PASSENGERS:
                return 0  # Invalid solution
            
            route_distance = self.calculate_route_metrics(route)[0]
            total_distance += route_distance
            total_cost += route_distance * self.COST_PER_KM
        
        # Fitness function: minimize cost while balancing route count
        # Higher fitness is better, so we use negative cost
        fitness = -total_cost - (route_count * 10)  # Penalty for more routes
        
        return fitness

    def _tournament_selection(self, fitness_scores, tournament_size=3):
        """Tournament selection for genetic algorithm"""
        tournament = random.sample(fitness_scores, tournament_size)
        return max(tournament, key=lambda x: x[1])[0]

    def _crossover(self, parent1, parent2):
        """Crossover operation for genetic algorithm"""
        # Simple crossover: take routes from both parents
        child1 = []
        child2 = []
        
        # Randomly select routes from parents
        for route in parent1:
            if random.random() < 0.5:
                child1.append(route)
            else:
                child2.append(route)
        
        for route in parent2:
            if random.random() < 0.5:
                child2.append(route)
            else:
                child1.append(route)
        
        return child1, child2

    def _mutate(self, individual):
        """Mutation operation for genetic algorithm"""
        mutated = individual.copy()
        
        # Random mutation: swap staff between routes
        if len(mutated) > 1:
            route1_idx = random.randint(0, len(mutated) - 1)
            route2_idx = random.randint(0, len(mutated) - 1)
            
            if route1_idx != route2_idx and mutated[route1_idx] and mutated[route2_idx]:
                # Swap a random staff member between routes
                staff1_idx = random.randint(0, len(mutated[route1_idx]) - 1)
                staff2_idx = random.randint(0, len(mutated[route2_idx]) - 1)
                
                mutated[route1_idx][staff1_idx], mutated[route2_idx][staff2_idx] = \
                    mutated[route2_idx][staff2_idx], mutated[route1_idx][staff1_idx]
        
        return mutated

    def _convert_genetic_solution_to_routes(self, individual):
        """Convert genetic algorithm solution to route format"""
        routes = {}
        for i, route in enumerate(individual):
            if len(route) >= self.MIN_PASSENGERS:
                routes[f'Route {i+1}'] = route
        
        return routes

    def _ant_colony_optimization(self, staff_data, n_ants=30, n_iterations=50, 
                               evaporation_rate=0.1, alpha=1.0, beta=2.0):
        """
        Ant Colony Optimization for route optimization
        
        Args:
            staff_data (pandas.DataFrame): Staff data
            n_ants (int): Number of ants
            n_iterations (int): Number of iterations
            evaporation_rate (float): Pheromone evaporation rate
            alpha (float): Pheromone importance
            beta (float): Distance importance
            
        Returns:
            dict: Optimized routes
        """
        logger.info(f"Starting Ant Colony Optimization with n_ants={n_ants}, "
                   f"n_iterations={n_iterations}")
        
        staff_list = staff_data.to_dict('records')
        
        # Initialize pheromone matrix
        n_staff = len(staff_list)
        pheromone = np.ones((n_staff, n_staff)) * 0.1
        
        best_solution = None
        best_cost = float('inf')
        
        for iteration in range(n_iterations):
            # Generate solutions for all ants
            ant_solutions = []
            ant_costs = []
            
            for ant in range(n_ants):
                solution = self._construct_ant_solution(staff_list, pheromone, alpha, beta)
                cost = self._calculate_ant_cost(solution, staff_list)
                ant_solutions.append(solution)
                ant_costs.append(cost)
                
                if cost < best_cost:
                    best_cost = cost
                    best_solution = solution.copy()
            
            # Update pheromone
            pheromone *= (1 - evaporation_rate)  # Evaporation
            
            # Add pheromone for best solution
            self._update_pheromone(pheromone, best_solution, best_cost)
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Best cost = {best_cost:.4f}")
        
        return self._convert_ant_solution_to_routes(best_solution, staff_list)

    def _construct_ant_solution(self, staff_list, pheromone, alpha, beta):
        """Construct solution for a single ant"""
        solution = []
        remaining_staff = list(range(len(staff_list)))
        
        while remaining_staff:
            # Start new route
            route = []
            current_staff = random.choice(remaining_staff)
            route.append(current_staff)
            remaining_staff.remove(current_staff)
            
            # Add more staff to route
            while len(route) < self.MAX_PASSENGERS and remaining_staff:
                # Calculate probabilities for next staff
                probabilities = []
                for staff_idx in remaining_staff:
                    # Calculate pheromone and distance factors
                    pheromone_factor = np.mean([pheromone[current_staff, staff_idx], 
                                              pheromone[staff_idx, current_staff]])
                    
                    distance = geodesic(
                        (staff_list[current_staff]['latitude'], staff_list[current_staff]['longitude']),
                        (staff_list[staff_idx]['latitude'], staff_list[staff_idx]['longitude'])
                    ).km
                    distance_factor = 1.0 / (1.0 + distance)
                    
                    prob = (pheromone_factor ** alpha) * (distance_factor ** beta)
                    probabilities.append(prob)
                
                # Select next staff based on probabilities
                if sum(probabilities) > 0:
                    probabilities = np.array(probabilities) / sum(probabilities)
                    next_staff_idx = np.random.choice(remaining_staff, p=probabilities)
                else:
                    next_staff_idx = random.choice(remaining_staff)
                
                route.append(next_staff_idx)
                remaining_staff.remove(next_staff_idx)
                current_staff = next_staff_idx
            
            solution.append(route)
        
        return solution

    def _calculate_ant_cost(self, solution, staff_list):
        """Calculate cost for ant colony solution"""
        total_cost = 0
        for route in solution:
            if len(route) >= self.MIN_PASSENGERS:
                route_staff = [staff_list[i] for i in route]
                route_distance = self.calculate_route_metrics(route_staff)[0]
                total_cost += route_distance * self.COST_PER_KM
        
        return total_cost

    def _update_pheromone(self, pheromone, solution, cost):
        """Update pheromone matrix"""
        for route in solution:
            for i in range(len(route)):
                for j in range(i + 1, len(route)):
                    pheromone[route[i], route[j]] += 1.0 / cost
                    pheromone[route[j], route[i]] += 1.0 / cost

    def _convert_ant_solution_to_routes(self, solution, staff_list):
        """Convert ant colony solution to route format"""
        routes = {}
        for i, route_indices in enumerate(solution):
            if len(route_indices) >= self.MIN_PASSENGERS:
                route_staff = [staff_list[idx] for idx in route_indices]
                routes[f'Route {i+1}'] = route_staff
        
        return routes

    def _improved_greedy_optimization(self, staff_data):
        """
        Improved greedy algorithm with multiple strategies
        
        Args:
            staff_data (pandas.DataFrame): Staff data
            
        Returns:
            dict: Optimized routes
        """
        logger.info("Starting improved greedy optimization")
        
        # Group by clusters first
        clustered_routes = self._cluster_based_greedy(staff_data)
        
        # Apply local optimization
        optimized_routes = self._local_optimization(clustered_routes)
        
        return optimized_routes

    def _cluster_based_greedy(self, staff_data):
        """Greedy optimization within clusters"""
        routes = {}
        route_counter = 1
        
        # Process each cluster
        for cluster_id in staff_data['cluster'].unique():
            cluster_data = staff_data[staff_data['cluster'] == cluster_id].copy()
            
            # Sort by distance to office
            cluster_data = cluster_data.sort_values('distance_to_office')
            
            # Create routes within cluster
            while len(cluster_data) >= self.MIN_PASSENGERS:
                # Take up to MAX_PASSENGERS closest to office
                route_size = min(self.MAX_PASSENGERS, len(cluster_data))
                route_staff = cluster_data.head(route_size).to_dict('records')
                
                routes[f'Route {route_counter}'] = route_staff
                route_counter += 1
                
                # Remove assigned staff
                cluster_data = cluster_data.iloc[route_size:]
            
            # Handle remaining staff in cluster
            if len(cluster_data) > 0:
                self._assign_remaining_staff(cluster_data.to_dict('records'), routes)
        
        return routes

    def _local_optimization(self, routes):
        """Apply local optimization to improve routes"""
        improved_routes = routes.copy()
        
        # Try to swap staff between routes to reduce total distance
        for _ in range(10):  # Limited iterations to avoid infinite loop
            improved = False
            
            route_names = list(improved_routes.keys())
            for i, route1_name in enumerate(route_names):
                for j, route2_name in enumerate(route_names[i+1:], i+1):
                    route1 = improved_routes[route1_name]
                    route2 = improved_routes[route2_name]
                    
                    # Try swapping staff between routes
                    if self._try_route_swap(route1, route2):
                        improved_routes[route1_name] = route1
                        improved_routes[route2_name] = route2
                        improved = True
            
            if not improved:
                break
        
        return improved_routes

    def _try_route_swap(self, route1, route2):
        """Try to swap staff between two routes to improve total distance"""
        original_distance1 = self.calculate_route_metrics(route1)[0]
        original_distance2 = self.calculate_route_metrics(route2)[0]
        original_total = original_distance1 + original_distance2
        
        best_improvement = 0
        best_swap = None
        
        # Try swapping each staff member
        for i, staff1 in enumerate(route1):
            for j, staff2 in enumerate(route2):
                # Create test routes with swapped staff
                test_route1 = route1.copy()
                test_route2 = route2.copy()
                test_route1[i] = staff2
                test_route2[j] = staff1
                
                # Check if routes are still valid
                if len(test_route1) >= self.MIN_PASSENGERS and len(test_route2) >= self.MIN_PASSENGERS:
                    new_distance1 = self.calculate_route_metrics(test_route1)[0]
                    new_distance2 = self.calculate_route_metrics(test_route2)[0]
                    new_total = new_distance1 + new_distance2
                    
                    improvement = original_total - new_total
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_swap = (i, j)
        
        # Apply best swap if it improves the solution
        if best_swap and best_improvement > 0:
            i, j = best_swap
            route1[i], route2[j] = route2[j], route1[i]
            return True
        
        return False

    def _hybrid_optimization(self, staff_data):
        """
        Hybrid optimization combining multiple algorithms
        
        Args:
            staff_data (pandas.DataFrame): Staff data
            
        Returns:
            dict: Optimized routes
        """
        logger.info("Starting hybrid optimization")
        
        # Step 1: Use SOM clustering for initial grouping
        clustered_data = self.create_clusters(staff_data)
        
        # Step 2: Apply genetic algorithm for global optimization
        genetic_routes = self._genetic_algorithm_optimization(clustered_data, 
                                                            population_size=30, 
                                                            generations=50)
        
        # Step 3: Apply local optimization for fine-tuning
        final_routes = self._local_optimization(genetic_routes)
        
        return final_routes

    def _assign_remaining_staff(self, remaining_staff, routes):
        """Assign remaining staff to existing routes"""
        logger.info(f"Assigning {len(remaining_staff)} remaining staff to existing routes")
        if not isinstance(remaining_staff, list):
            logger.warning("remaining_staff is not a list")
            return
            
        for staff in remaining_staff:
            best_route = None
            min_detour = float('inf')
            
            for route_name, route_group in routes.items():
                if len(route_group) < self.MAX_PASSENGERS:
                    # Calculate current route distance
                    current_distance = self.calculate_route_metrics(route_group)[0]
                    
                    # Calculate new route distance with added staff member
                    test_route = route_group.copy()
                    test_route.append(staff)
                    new_distance = self.calculate_route_metrics(test_route)[0]
                    
                    # Calculate detour distance
                    detour = new_distance - current_distance
                    
                    if detour < min_detour:
                        min_detour = detour
                        best_route = route_name
            
            if best_route:
                routes[best_route].append(staff)
                logger.debug(f"Assigned staff {staff.get('name', 'Unknown')} to {best_route}")

    def calculate_route_metrics(self, route):
        """
        Calculate metrics for a single route
        
        Args:
            route (list): List of staff dictionaries in the route
            
        Returns:
            tuple: (total_distance, route_cost, passenger_count)
        """
        if not route:
            return 0, 0, 0
        
        total_distance = 0
        current_location = (self.office_location['lat'], self.office_location['lon'])
        
        # Calculate distance from office to first passenger
        if route:
            first_passenger = (route[0]['latitude'], route[0]['longitude'])
            total_distance += geodesic(current_location, first_passenger).km
            current_location = first_passenger
        
        # Calculate distances between passengers
        for i in range(len(route) - 1):
            passenger1 = (route[i]['latitude'], route[i]['longitude'])
            passenger2 = (route[i + 1]['latitude'], route[i + 1]['longitude'])
            total_distance += geodesic(passenger1, passenger2).km
        
        # Calculate distance from last passenger back to office
        if route:
            last_passenger = (route[-1]['latitude'], route[-1]['longitude'])
            total_distance += geodesic(last_passenger, 
                                     (self.office_location['lat'], self.office_location['lon'])).km
        
        route_cost = total_distance * self.COST_PER_KM
        passenger_count = len(route)
        
        return total_distance, route_cost, passenger_count

    def get_route_summary(self, route):
        """
        Get detailed summary for a route
        
        Args:
            route (list): List of staff dictionaries in the route
            
        Returns:
            dict: Route summary
        """
        distance, cost, passenger_count = self.calculate_route_metrics(route)
        
        # Calculate average distance to office
        avg_distance_to_office = np.mean([staff['distance_to_office'] for staff in route])
        
        # Calculate route efficiency (distance per passenger)
        efficiency = distance / passenger_count if passenger_count > 0 else 0
        
        return {
            'passenger_count': passenger_count,
            'total_distance': distance,
            'route_cost': cost,
            'avg_distance_to_office': avg_distance_to_office,
            'efficiency': efficiency,
            'passengers': [staff['name'] for staff in route]
        }

    def calculate_total_metrics(self, routes):
        """
        Calculate total metrics for all routes
        
        Args:
            routes (dict): Dictionary of routes
            
        Returns:
            dict: Total metrics
        """
        total_distance = 0
        total_cost = 0
        total_passengers = 0
        route_count = len(routes)
        
        route_details = {}
            
        for route_name, route in routes.items():
            distance, cost, passenger_count = self.calculate_route_metrics(route)
            total_distance += distance
            total_cost += cost
            total_passengers += passenger_count
            
            route_details[route_name] = self.get_route_summary(route)
        
        avg_distance_per_route = total_distance / route_count if route_count > 0 else 0
        avg_cost_per_route = total_cost / route_count if route_count > 0 else 0
        avg_passengers_per_route = total_passengers / route_count if route_count > 0 else 0
        
        return {
            'total_routes': route_count,
            'total_distance': total_distance,
            'total_cost': total_cost,
            'total_passengers': total_passengers,
            'avg_distance_per_route': avg_distance_per_route,
            'avg_cost_per_route': avg_cost_per_route,
            'avg_passengers_per_route': avg_passengers_per_route,
            'route_details': route_details
        } 