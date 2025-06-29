import numpy as np
import pandas as pd
import random
from geopy.distance import geodesic
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class AntColonyOptimizer:
    """
    Ant Colony Optimization for route optimization with:
    - Pheromone-based path finding
    - Multiple ant colonies
    - Adaptive pheromone update
    - Local optimization
    """
    
    def __init__(self, office_location, max_passengers=4, min_passengers=3, 
                 cost_per_km=2.5, random_state=None):
        """
        Initialize ant colony optimizer
        
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
        self.best_cost = float('inf')
        self.cost_history = []
        
    def optimize(self, staff_data, n_ants=50, n_iterations=100, 
                evaporation_rate=0.1, alpha=1.0, beta=2.0, 
                q=100, local_search=True, verbose=True):
        """
        Run ant colony optimization
        
        Args:
            staff_data (pd.DataFrame): Staff location data
            n_ants (int): Number of ants
            n_iterations (int): Number of iterations
            evaporation_rate (float): Pheromone evaporation rate
            alpha (float): Pheromone importance
            beta (float): Distance importance
            q (float): Pheromone deposit constant
            local_search (bool): Apply local search
            verbose (bool): Print progress
            
        Returns:
            dict: Optimized routes
        """
        logger.info(f"Starting Ant Colony Optimization with {n_ants} ants, "
                   f"{n_iterations} iterations")
        
        # Convert staff data to list
        staff_list = staff_data.to_dict('records')
        n_staff = len(staff_list)
        
        # Initialize pheromone matrix
        pheromone = np.ones((n_staff, n_staff)) * 0.1
        
        # Calculate distance matrix
        distance_matrix = self._calculate_distance_matrix(staff_list)
        
        # Main optimization loop
        for iteration in range(n_iterations):
            # Generate solutions for all ants
            ant_solutions = []
            ant_costs = []
            
            for ant in range(n_ants):
                solution = self._construct_solution(staff_list, pheromone, 
                                                  distance_matrix, alpha, beta)
                
                if local_search:
                    solution = self._local_search(solution, staff_list)
                
                cost = self._calculate_solution_cost(solution, staff_list)
                ant_solutions.append(solution)
                ant_costs.append(cost)
                
                # Update best solution
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = solution.copy()
            
            # Update pheromone
            pheromone = self._update_pheromone(pheromone, ant_solutions, ant_costs, 
                                             evaporation_rate, q)
            
            self.cost_history.append(self.best_cost)
            
            if verbose and iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Best cost = {self.best_cost:.4f}")
        
        # Return best solution
        return self._convert_to_routes(self.best_solution, staff_list)
    
    def _calculate_distance_matrix(self, staff_list):
        """Calculate distance matrix between all staff members"""
        n_staff = len(staff_list)
        distance_matrix = np.zeros((n_staff, n_staff))
        
        for i in range(n_staff):
            for j in range(i + 1, n_staff):
                distance = geodesic(
                    (staff_list[i]['latitude'], staff_list[i]['longitude']),
                    (staff_list[j]['latitude'], staff_list[j]['longitude'])
                ).km
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        return distance_matrix
    
    def _construct_solution(self, staff_list, pheromone, distance_matrix, alpha, beta):
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
                    pheromone_factor = np.mean([
                        pheromone[current_staff, staff_idx],
                        pheromone[staff_idx, current_staff]
                    ])
                    
                    distance_factor = 1.0 / (1.0 + distance_matrix[current_staff, staff_idx])
                    
                    # Calculate probability
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
    
    def _local_search(self, solution, staff_list):
        """Apply local search to improve solution"""
        improved_solution = solution.copy()
        
        # Try to improve each route
        for route_idx, route in enumerate(improved_solution):
            if len(route) > 1:
                # Try different orderings of staff in route
                best_route = route
                best_distance = self._calculate_route_distance(route, staff_list)
                
                # Try swapping adjacent staff
                for i in range(len(route) - 1):
                    test_route = route.copy()
                    test_route[i], test_route[i + 1] = test_route[i + 1], test_route[i]
                    
                    test_distance = self._calculate_route_distance(test_route, staff_list)
                    if test_distance < best_distance:
                        best_route = test_route
                        best_distance = test_distance
                
                improved_solution[route_idx] = best_route
        
        return improved_solution
    
    def _calculate_route_distance(self, route, staff_list):
        """Calculate total distance for a route"""
        if not route:
            return 0
        
        total_distance = 0
        current_location = (self.office_location['lat'], self.office_location['lon'])
        
        # Distance from office to first passenger
        first_staff = staff_list[route[0]]
        first_location = (first_staff['latitude'], first_staff['longitude'])
        total_distance += geodesic(current_location, first_location).km
        current_location = first_location
        
        # Distances between passengers
        for i in range(len(route) - 1):
            staff1 = staff_list[route[i]]
            staff2 = staff_list[route[i + 1]]
            
            location1 = (staff1['latitude'], staff1['longitude'])
            location2 = (staff2['latitude'], staff2['longitude'])
            
            total_distance += geodesic(location1, location2).km
        
        # Distance from last passenger back to office
        last_staff = staff_list[route[-1]]
        last_location = (last_staff['latitude'], last_staff['longitude'])
        total_distance += geodesic(last_location, 
                                 (self.office_location['lat'], self.office_location['lon'])).km
        
        return total_distance
    
    def _calculate_solution_cost(self, solution, staff_list):
        """Calculate total cost for a solution"""
        total_cost = 0
        
        for route in solution:
            if len(route) >= self.MIN_PASSENGERS:
                route_distance = self._calculate_route_distance(route, staff_list)
                total_cost += route_distance * self.COST_PER_KM
        
        return total_cost
    
    def _update_pheromone(self, pheromone, ant_solutions, ant_costs, evaporation_rate, q):
        """Update pheromone matrix"""
        # Evaporation
        pheromone *= (1 - evaporation_rate)
        
        # Add pheromone for each ant's solution
        for solution, cost in zip(ant_solutions, ant_costs):
            if cost > 0:  # Avoid division by zero
                delta_pheromone = q / cost
                
                for route in solution:
                    for i in range(len(route)):
                        for j in range(i + 1, len(route)):
                            pheromone[route[i], route[j]] += delta_pheromone
                            pheromone[route[j], route[i]] += delta_pheromone
        
        return pheromone
    
    def _convert_to_routes(self, solution, staff_list):
        """Convert ant colony solution to route format"""
        routes = {}
        
        for i, route_indices in enumerate(solution):
            if len(route_indices) >= self.MIN_PASSENGERS:
                route_staff = [staff_list[idx] for idx in route_indices]
                routes[f'Route {i+1}'] = route_staff
        
        return routes
    
    def get_optimization_stats(self):
        """Get optimization statistics"""
        return {
            'best_cost': self.best_cost,
            'cost_history': self.cost_history,
            'iterations': len(self.cost_history)
        } 