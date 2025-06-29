import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional
from .som_cluster import SOMCluster
from .genetic_optimizer import GeneticRouteOptimizer
from .ant_colony_optimizer import AntColonyOptimizer
from geopy.distance import geodesic

logger = logging.getLogger(__name__)

class HybridRouteOptimizer:
    """
    Hybrid route optimizer that combines multiple algorithms:
    - SOM clustering for initial grouping
    - Genetic Algorithm for global optimization
    - Ant Colony Optimization for local refinement
    - Local search for final improvement
    """
    
    def __init__(self, office_location, max_passengers=4, min_passengers=3, 
                 cost_per_km=2.5, random_state=None):
        """
        Initialize hybrid optimizer
        
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
        self.random_state = random_state
        
        # Initialize component optimizers
        self.genetic_optimizer = GeneticRouteOptimizer(
            office_location, max_passengers, min_passengers, cost_per_km, random_state
        )
        
        self.ant_colony_optimizer = AntColonyOptimizer(
            office_location, max_passengers, min_passengers, cost_per_km, random_state
        )
        
        self.best_solution = None
        self.best_cost = float('inf')
        self.optimization_history = []
        
    def optimize(self, staff_data, strategy='full', verbose=True):
        """
        Run hybrid optimization
        
        Args:
            staff_data (pd.DataFrame): Staff location data
            strategy (str): Optimization strategy ('full', 'fast', 'balanced')
            verbose (bool): Print progress
            
        Returns:
            dict: Optimized routes
        """
        logger.info(f"Starting Hybrid optimization with strategy: {strategy}")
        
        if strategy == 'full':
            return self._full_optimization(staff_data, verbose)
        elif strategy == 'fast':
            return self._fast_optimization(staff_data, verbose)
        elif strategy == 'balanced':
            return self._balanced_optimization(staff_data, verbose)
        else:
            logger.warning(f"Unknown strategy {strategy}, using balanced")
            return self._balanced_optimization(staff_data, verbose)
    
    def _full_optimization(self, staff_data, verbose):
        """Full optimization pipeline"""
        if verbose:
            logger.info("Step 1: SOM Clustering for initial grouping")
        
        # Step 1: SOM clustering
        clustered_data = self._som_clustering(staff_data)
        
        if verbose:
            logger.info("Step 2: Genetic Algorithm for global optimization")
        
        # Step 2: Genetic algorithm
        genetic_routes = self.genetic_optimizer.optimize(
            clustered_data, 
            population_size=150, 
            generations=300,
            verbose=verbose
        )
        
        if verbose:
            logger.info("Step 3: Ant Colony Optimization for refinement")
        
        # Step 3: Ant colony optimization
        ant_routes = self.ant_colony_optimizer.optimize(
            staff_data,
            n_ants=100,
            n_iterations=150,
            verbose=verbose
        )
        
        if verbose:
            logger.info("Step 4: Local search for final improvement")
        
        # Step 4: Local search
        final_routes = self._local_search_improvement(genetic_routes)
        
        # Compare and select best solution
        genetic_cost = self._calculate_total_cost(genetic_routes)
        ant_cost = self._calculate_total_cost(ant_routes)
        final_cost = self._calculate_total_cost(final_routes)
        
        if verbose:
            logger.info(f"Genetic Algorithm cost: {genetic_cost:.2f}")
            logger.info(f"Ant Colony cost: {ant_cost:.2f}")
            logger.info(f"Final optimized cost: {final_cost:.2f}")
        
        # Select best solution
        if final_cost <= genetic_cost and final_cost <= ant_cost:
            self.best_solution = final_routes
            self.best_cost = final_cost
        elif genetic_cost <= ant_cost:
            self.best_solution = genetic_routes
            self.best_cost = genetic_cost
        else:
            self.best_solution = ant_routes
            self.best_cost = ant_cost
        
        return self.best_solution
    
    def _fast_optimization(self, staff_data, verbose):
        """Fast optimization pipeline"""
        if verbose:
            logger.info("Fast optimization: SOM + Genetic Algorithm")
        
        # Step 1: SOM clustering
        clustered_data = self._som_clustering(staff_data)
        
        # Step 2: Genetic algorithm with reduced parameters
        genetic_routes = self.genetic_optimizer.optimize(
            clustered_data,
            population_size=50,
            generations=100,
            verbose=verbose
        )
        
        # Step 3: Quick local search
        final_routes = self._quick_local_search(genetic_routes)
        
        self.best_solution = final_routes
        self.best_cost = self._calculate_total_cost(final_routes)
        
        return self.best_solution
    
    def _balanced_optimization(self, staff_data, verbose):
        """Balanced optimization pipeline"""
        if verbose:
            logger.info("Balanced optimization: SOM + Genetic + Local Search")
        
        # Step 1: SOM clustering
        clustered_data = self._som_clustering(staff_data)
        
        # Step 2: Genetic algorithm with balanced parameters
        genetic_routes = self.genetic_optimizer.optimize(
            clustered_data,
            population_size=100,
            generations=200,
            verbose=verbose
        )
        
        # Step 3: Local search
        final_routes = self._local_search_improvement(genetic_routes)
        
        self.best_solution = final_routes
        self.best_cost = self._calculate_total_cost(final_routes)
        
        return self.best_solution
    
    def _som_clustering(self, staff_data):
        """Apply SOM clustering for initial grouping"""
        # Calculate distances to office
        staff_data = staff_data.copy()
        staff_data['distance_to_office'] = staff_data.apply(
            lambda row: geodesic(
                (row['latitude'], row['longitude']),
                (self.office_location['lat'], self.office_location['lon'])
            ).km,
            axis=1
        )
        
        # Prepare data for clustering
        locations = staff_data[['latitude', 'longitude', 'distance_to_office']].values
        
        # Initialize and train SOM
        som = SOMCluster(
            input_len=3,
            grid_size=3,
            sigma=1.0,
            learning_rate=0.5,
            initialization='pca',
            random_state=self.random_state
        )
        
        som.train(locations, epochs=2000, verbose=False)
        
        # Assign clusters
        staff_data['cluster'] = [som.get_cluster(loc) for loc in locations]
        
        # Handle small clusters
        self._handle_small_clusters(staff_data)
        
        return staff_data
    
    def _handle_small_clusters(self, staff_data):
        """Handle clusters with fewer than minimum required passengers"""
        cluster_sizes = staff_data['cluster'].value_counts()
        small_clusters = cluster_sizes[cluster_sizes < self.MIN_PASSENGERS].index
        
        if len(small_clusters) > 0:
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
                        best_cluster = min(distances, key=lambda x: x[1])[0]
                        staff_data.loc[idx, 'cluster'] = best_cluster
    
    def _local_search_improvement(self, routes):
        """Apply comprehensive local search to improve routes"""
        improved_routes = routes.copy()
        
        # Multiple improvement passes
        for iteration in range(5):
            improved = False
            
            # Try route merging
            improved_routes, merge_improved = self._try_route_merging(improved_routes)
            improved = improved or merge_improved
            
            # Try staff swapping
            improved_routes, swap_improved = self._try_staff_swapping(improved_routes)
            improved = improved or swap_improved
            
            # Try route reordering
            improved_routes, reorder_improved = self._try_route_reordering(improved_routes)
            improved = improved or reorder_improved
            
            if not improved:
                break
        
        return improved_routes
    
    def _quick_local_search(self, routes):
        """Apply quick local search"""
        improved_routes = routes.copy()
        
        # Single pass of improvements
        improved_routes, _ = self._try_staff_swapping(improved_routes)
        improved_routes, _ = self._try_route_reordering(improved_routes)
        
        return improved_routes
    
    def _try_route_merging(self, routes):
        """Try to merge routes to reduce total cost"""
        improved_routes = routes.copy()
        improved = False
        
        route_names = list(improved_routes.keys())
        
        for i, route1_name in enumerate(route_names):
            for j, route2_name in enumerate(route_names[i+1:], i+1):
                route1 = improved_routes[route1_name]
                route2 = improved_routes[route2_name]
                
                # Check if routes can be merged
                if len(route1) + len(route2) <= self.MAX_PASSENGERS:
                    # Calculate current cost
                    current_cost = (self._calculate_route_cost(route1) + 
                                  self._calculate_route_cost(route2))
                    
                    # Calculate merged route cost
                    merged_route = route1 + route2
                    merged_cost = self._calculate_route_cost(merged_route)
                    
                    # If merging improves cost
                    if merged_cost < current_cost:
                        improved_routes[route1_name] = merged_route
                        del improved_routes[route2_name]
                        improved = True
                        break
            
            if improved:
                break
        
        return improved_routes, improved
    
    def _try_staff_swapping(self, routes):
        """Try swapping staff between routes"""
        improved_routes = routes.copy()
        improved = False
        
        route_names = list(improved_routes.keys())
        
        for i, route1_name in enumerate(route_names):
            for j, route2_name in enumerate(route_names[i+1:], i+1):
                route1 = improved_routes[route1_name]
                route2 = improved_routes[route2_name]
                
                # Try swapping each staff member
                for staff1_idx, staff1 in enumerate(route1):
                    for staff2_idx, staff2 in enumerate(route2):
                        # Create test routes
                        test_route1 = route1.copy()
                        test_route2 = route2.copy()
                        test_route1[staff1_idx] = staff2
                        test_route2[staff2_idx] = staff1
                        
                        # Check if routes are still valid
                        if (len(test_route1) >= self.MIN_PASSENGERS and 
                            len(test_route2) >= self.MIN_PASSENGERS):
                            
                            # Calculate cost improvement
                            current_cost = (self._calculate_route_cost(route1) + 
                                          self._calculate_route_cost(route2))
                            new_cost = (self._calculate_route_cost(test_route1) + 
                                      self._calculate_route_cost(test_route2))
                            
                            if new_cost < current_cost:
                                improved_routes[route1_name] = test_route1
                                improved_routes[route2_name] = test_route2
                                improved = True
                                break
                    
                    if improved:
                        break
                
                if improved:
                    break
        
        return improved_routes, improved
    
    def _try_route_reordering(self, routes):
        """Try reordering staff within routes"""
        improved_routes = routes.copy()
        improved = False
        
        for route_name, route in improved_routes.items():
            if len(route) > 1:
                best_route = route
                best_cost = self._calculate_route_cost(route)
                
                # Try different orderings
                for i in range(len(route) - 1):
                    for j in range(i + 1, len(route)):
                        test_route = route.copy()
                        test_route[i], test_route[j] = test_route[j], test_route[i]
                        
                        test_cost = self._calculate_route_cost(test_route)
                        if test_cost < best_cost:
                            best_route = test_route
                            best_cost = test_cost
                            improved = True
                
                improved_routes[route_name] = best_route
        
        return improved_routes, improved
    
    def _calculate_route_cost(self, route):
        """Calculate cost for a single route"""
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
        
        return total_distance * self.COST_PER_KM
    
    def _calculate_total_cost(self, routes):
        """Calculate total cost for all routes"""
        total_cost = 0
        for route in routes.values():
            total_cost += self._calculate_route_cost(route)
        return total_cost
    
    def get_optimization_stats(self):
        """Get optimization statistics"""
        return {
            'best_cost': self.best_cost,
            'optimization_history': self.optimization_history
        } 