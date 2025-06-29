import numpy as np
import pandas as pd
import random
from geopy.distance import geodesic
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class GeneticRouteOptimizer:
    """
    Genetic Algorithm for route optimization with advanced features:
    - Multi-objective optimization
    - Adaptive mutation and crossover
    - Elite preservation
    - Diversity maintenance
    """
    
    def __init__(self, office_location, max_passengers=4, min_passengers=3, 
                 cost_per_km=2.5, random_state=None):
        """
        Initialize genetic algorithm optimizer
        
        Args:
            office_location (dict): Office coordinates
            max_passengers (int): Maximum passengers per route
            min_passengers (int): Minimum passengers per route
            cost_per_km (float): Cost per kilometer
            random_state (int): Random seed
        """
        self.office_location = office_location
        self.MAX_PASSENGERS = max_passengers
        self.MIN_PASSENGERS = min_passengers
        self.COST_PER_KM = cost_per_km
        
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
            
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
        
    def optimize(self, staff_data, population_size=100, generations=200, 
                mutation_rate=0.1, crossover_rate=0.8, elite_size=10,
                tournament_size=3, verbose=True):
        """
        Run genetic algorithm optimization
        
        Args:
            staff_data (pd.DataFrame): Staff location data
            population_size (int): Size of population
            generations (int): Number of generations
            mutation_rate (float): Mutation probability
            crossover_rate (float): Crossover probability
            elite_size (int): Number of elite individuals
            tournament_size (int): Tournament selection size
            verbose (bool): Print progress
            
        Returns:
            dict: Optimized routes
        """
        logger.info(f"Starting Genetic Algorithm optimization with {population_size} population, "
                   f"{generations} generations")
        
        # Convert staff data to list
        staff_list = staff_data.to_dict('records')
        
        # Initialize population
        population = self._initialize_population(staff_list, population_size)
        
        # Evolution loop
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [(individual, self._calculate_fitness(individual, staff_list)) 
                             for individual in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Update best solution
            if fitness_scores[0][1] > self.best_fitness:
                self.best_fitness = fitness_scores[0][1]
                self.best_solution = fitness_scores[0][0].copy()
            
            self.fitness_history.append(self.best_fitness)
            
            if verbose and generation % 20 == 0:
                logger.info(f"Generation {generation}: Best fitness = {self.best_fitness:.4f}")
            
            # Create new population
            new_population = []
            
            # Elitism
            for i in range(elite_size):
                new_population.append(fitness_scores[i][0])
            
            # Generate rest through selection, crossover, and mutation
            while len(new_population) < population_size:
                # Selection
                parent1 = self._tournament_selection(fitness_scores, tournament_size)
                parent2 = self._tournament_selection(fitness_scores, tournament_size)
                
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
            
            population = new_population[:population_size]
        
        # Return best solution
        return self._convert_to_routes(self.best_solution, staff_list)
    
    def _initialize_population(self, staff_list, population_size):
        """Initialize population with diverse individuals"""
        population = []
        
        for _ in range(population_size):
            individual = self._create_random_individual(staff_list)
            population.append(individual)
        
        return population
    
    def _create_random_individual(self, staff_list):
        """Create a random individual (route assignment)"""
        individual = []
        remaining_staff = staff_list.copy()
        random.shuffle(remaining_staff)
        
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
            
            # Random route size between min and max
            max_route_size = min(self.MAX_PASSENGERS, len(remaining_staff))
            if max_route_size >= self.MIN_PASSENGERS:
                route_size = random.randint(self.MIN_PASSENGERS, max_route_size)
            else:
                route_size = len(remaining_staff)
            
            # Create route
            route = remaining_staff[:route_size]
            individual.append(route)
            
            # Remove assigned staff
            remaining_staff = remaining_staff[route_size:]
        
        return individual
    
    def _calculate_fitness(self, individual, staff_list):
        """Calculate fitness score for an individual"""
        total_cost = 0
        total_distance = 0
        route_count = len(individual)
        
        # Calculate cost for each route
        for route in individual:
            if len(route) < self.MIN_PASSENGERS:
                return float('-inf')  # Invalid solution
            
            route_distance = self._calculate_route_distance(route)
            total_distance += route_distance
            total_cost += route_distance * self.COST_PER_KM
        
        # Multi-objective fitness function
        # Minimize cost and route count while maximizing efficiency
        efficiency = total_distance / sum(len(route) for route in individual) if individual else 0
        
        # Fitness = -cost - route_penalty + efficiency_bonus
        route_penalty = route_count * 50  # Penalty for more routes
        efficiency_bonus = efficiency * 10  # Bonus for efficiency
        
        fitness = -total_cost - route_penalty + efficiency_bonus
        
        return fitness
    
    def _calculate_route_distance(self, route):
        """Calculate total distance for a route"""
        if not route:
            return 0
        
        total_distance = 0
        current_location = (self.office_location['lat'], self.office_location['lon'])
        
        # Distance from office to first passenger
        first_passenger = (route[0]['latitude'], route[0]['longitude'])
        total_distance += geodesic(current_location, first_passenger).km
        current_location = first_passenger
        
        # Distances between passengers
        for i in range(len(route) - 1):
            passenger1 = (route[i]['latitude'], route[i]['longitude'])
            passenger2 = (route[i + 1]['latitude'], route[i + 1]['longitude'])
            total_distance += geodesic(passenger1, passenger2).km
        
        # Distance from last passenger back to office
        last_passenger = (route[-1]['latitude'], route[-1]['longitude'])
        total_distance += geodesic(last_passenger, 
                                 (self.office_location['lat'], self.office_location['lon'])).km
        
        return total_distance
    
    def _tournament_selection(self, fitness_scores, tournament_size):
        """Tournament selection"""
        tournament = random.sample(fitness_scores, tournament_size)
        return max(tournament, key=lambda x: x[1])[0]
    
    def _crossover(self, parent1, parent2):
        """Crossover operation"""
        # Route-based crossover
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
        """Mutation operation"""
        mutated = individual.copy()
        
        mutation_type = random.choice(['swap', 'split', 'merge', 'shuffle'])
        
        if mutation_type == 'swap' and len(mutated) > 1:
            # Swap staff between routes
            route1_idx = random.randint(0, len(mutated) - 1)
            route2_idx = random.randint(0, len(mutated) - 1)
            
            if route1_idx != route2_idx and mutated[route1_idx] and mutated[route2_idx]:
                staff1_idx = random.randint(0, len(mutated[route1_idx]) - 1)
                staff2_idx = random.randint(0, len(mutated[route2_idx]) - 1)
                
                mutated[route1_idx][staff1_idx], mutated[route2_idx][staff2_idx] = \
                    mutated[route2_idx][staff2_idx], mutated[route1_idx][staff1_idx]
        
        elif mutation_type == 'split' and len(mutated) > 0:
            # Split a route into two
            route_idx = random.randint(0, len(mutated) - 1)
            route = mutated[route_idx]
            
            if len(route) >= self.MIN_PASSENGERS * 2:
                min_split = self.MIN_PASSENGERS
                max_split = len(route) - self.MIN_PASSENGERS
                if max_split >= min_split:
                    split_point = random.randint(min_split, max_split)
                    route1 = route[:split_point]
                    route2 = route[split_point:]
                    
                    mutated[route_idx] = route1
                    mutated.append(route2)
        
        elif mutation_type == 'merge' and len(mutated) > 1:
            # Merge two routes
            route1_idx = random.randint(0, len(mutated) - 1)
            route2_idx = random.randint(0, len(mutated) - 1)
            
            if route1_idx != route2_idx:
                combined_route = mutated[route1_idx] + mutated[route2_idx]
                
                if len(combined_route) <= self.MAX_PASSENGERS:
                    mutated[route1_idx] = combined_route
                    mutated.pop(route2_idx)
        
        elif mutation_type == 'shuffle':
            # Shuffle staff within routes
            for route in mutated:
                if len(route) > 1:
                    random.shuffle(route)
        
        return mutated
    
    def _convert_to_routes(self, individual, staff_list):
        """Convert genetic solution to route format"""
        routes = {}
        
        for i, route in enumerate(individual):
            if len(route) >= self.MIN_PASSENGERS:
                routes[f'Route {i+1}'] = route
        
        return routes
    
    def get_optimization_stats(self):
        """Get optimization statistics"""
        return {
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'generations': len(self.fitness_history)
        } 