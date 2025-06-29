import time
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from .som_cluster import SOMCluster
from .route_optimizer import RouteOptimizer
from .genetic_optimizer import GeneticRouteOptimizer
from .ant_colony_optimizer import AntColonyOptimizer
from .hybrid_optimizer import HybridRouteOptimizer

logger = logging.getLogger(__name__)

class AlgorithmComparison:
    """
    Utility class to compare different route optimization algorithms
    """
    
    def __init__(self, office_location, max_passengers=4, min_passengers=3, cost_per_km=2.5):
        """
        Initialize algorithm comparison
        
        Args:
            office_location (dict): Office coordinates
            max_passengers (int): Maximum passengers per route
            min_passengers (int): Minimum passengers per route
            cost_per_km (float): Cost per kilometer
        """
        self.office_location = office_location
        self.MAX_PASSENGERS = max_passengers
        self.MIN_PASSENGERS = min_passengers
        self.COST_PER_KM = cost_per_km
        
        # Initialize all optimizers
        self.optimizers = {
            'SOM + Greedy': RouteOptimizer(office_location, max_passengers, min_passengers, cost_per_km),
            'Genetic Algorithm': GeneticRouteOptimizer(office_location, max_passengers, min_passengers, cost_per_km),
            'Ant Colony': AntColonyOptimizer(office_location, max_passengers, min_passengers, cost_per_km),
            'Hybrid (Balanced)': HybridRouteOptimizer(office_location, max_passengers, min_passengers, cost_per_km),
            'Hybrid (Fast)': HybridRouteOptimizer(office_location, max_passengers, min_passengers, cost_per_km),
            'Hybrid (Full)': HybridRouteOptimizer(office_location, max_passengers, min_passengers, cost_per_km)
        }
        
    def compare_algorithms(self, staff_data, verbose=True):
        """
        Compare all algorithms on the same dataset
        
        Args:
            staff_data (pd.DataFrame): Staff location data
            verbose (bool): Print detailed results
            
        Returns:
            pd.DataFrame: Comparison results
        """
        logger.info("Starting algorithm comparison")
        
        results = []
        
        for name, optimizer in self.optimizers.items():
            logger.info(f"Testing {name}")
            
            start_time = time.time()
            
            try:
                if name == 'SOM + Greedy':
                    # Use the original route optimizer
                    clustered_data = optimizer.create_clusters(staff_data)
                    routes = optimizer.optimize_routes(clustered_data, algorithm='greedy')
                elif name == 'Genetic Algorithm':
                    routes = optimizer.optimize(staff_data, population_size=100, generations=150, verbose=False)
                elif name == 'Ant Colony':
                    routes = optimizer.optimize(staff_data, n_ants=50, n_iterations=100, verbose=False)
                elif name == 'Hybrid (Balanced)':
                    routes = optimizer.optimize(staff_data, strategy='balanced', verbose=False)
                elif name == 'Hybrid (Fast)':
                    routes = optimizer.optimize(staff_data, strategy='fast', verbose=False)
                elif name == 'Hybrid (Full)':
                    routes = optimizer.optimize(staff_data, strategy='full', verbose=False)
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Calculate metrics
                metrics = self._calculate_metrics(routes, staff_data)
                
                result = {
                    'Algorithm': name,
                    'Execution Time (s)': execution_time,
                    'Total Cost': metrics['total_cost'],
                    'Total Distance (km)': metrics['total_distance'],
                    'Number of Routes': metrics['num_routes'],
                    'Average Route Cost': metrics['avg_route_cost'],
                    'Average Route Distance': metrics['avg_route_distance'],
                    'Average Passengers per Route': metrics['avg_passengers_per_route'],
                    'Route Efficiency': metrics['route_efficiency'],
                    'Success': True
                }
                
                if verbose:
                    logger.info(f"{name}: Cost={metrics['total_cost']:.2f}, "
                              f"Routes={metrics['num_routes']}, Time={execution_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error with {name}: {str(e)}")
                result = {
                    'Algorithm': name,
                    'Execution Time (s)': 0,
                    'Total Cost': float('inf'),
                    'Total Distance (km)': float('inf'),
                    'Number of Routes': 0,
                    'Average Route Cost': float('inf'),
                    'Average Route Distance': float('inf'),
                    'Average Passengers per Route': 0,
                    'Route Efficiency': 0,
                    'Success': False
                }
            
            results.append(result)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(results)
        
        if verbose:
            self._print_comparison_summary(comparison_df)
        
        return comparison_df
    
    def _calculate_metrics(self, routes, staff_data):
        """Calculate comprehensive metrics for routes"""
        total_cost = 0
        total_distance = 0
        total_passengers = 0
        num_routes = len(routes)
        
        for route_name, route in routes.items():
            if len(route) >= self.MIN_PASSENGERS:
                # Calculate route distance
                route_distance = self._calculate_route_distance(route)
                route_cost = route_distance * self.COST_PER_KM
                
                total_distance += route_distance
                total_cost += route_cost
                total_passengers += len(route)
        
        avg_route_cost = total_cost / num_routes if num_routes > 0 else 0
        avg_route_distance = total_distance / num_routes if num_routes > 0 else 0
        avg_passengers_per_route = total_passengers / num_routes if num_routes > 0 else 0
        
        # Route efficiency (lower is better)
        route_efficiency = total_distance / total_passengers if total_passengers > 0 else float('inf')
        
        return {
            'total_cost': total_cost,
            'total_distance': total_distance,
            'num_routes': num_routes,
            'avg_route_cost': avg_route_cost,
            'avg_route_distance': avg_route_distance,
            'avg_passengers_per_route': avg_passengers_per_route,
            'route_efficiency': route_efficiency
        }
    
    def _calculate_route_distance(self, route):
        """Calculate total distance for a route"""
        if not route:
            return 0
        
        total_distance = 0
        current_location = (self.office_location['lat'], self.office_location['lon'])
        
        # Distance from office to first passenger
        first_passenger = (route[0]['latitude'], route[0]['longitude'])
        total_distance += self._calculate_distance(current_location, first_passenger)
        current_location = first_passenger
        
        # Distances between passengers
        for i in range(len(route) - 1):
            passenger1 = (route[i]['latitude'], route[i]['longitude'])
            passenger2 = (route[i + 1]['latitude'], route[i + 1]['longitude'])
            total_distance += self._calculate_distance(passenger1, passenger2)
        
        # Distance from last passenger back to office
        last_passenger = (route[-1]['latitude'], route[-1]['longitude'])
        total_distance += self._calculate_distance(last_passenger, 
                                                 (self.office_location['lat'], self.office_location['lon']))
        
        return total_distance
    
    def _calculate_distance(self, point1, point2):
        """Calculate distance between two points"""
        from geopy.distance import geodesic
        return geodesic(point1, point2).km
    
    def _print_comparison_summary(self, comparison_df):
        """Print a summary of the comparison results"""
        print("\n" + "="*80)
        print("ALGORITHM COMPARISON RESULTS")
        print("="*80)
        
        # Sort by total cost
        sorted_df = comparison_df.sort_values('Total Cost')
        
        print(f"\nBest Algorithm: {sorted_df.iloc[0]['Algorithm']}")
        print(f"Best Cost: ${sorted_df.iloc[0]['Total Cost']:.2f}")
        print(f"Execution Time: {sorted_df.iloc[0]['Execution Time (s)']:.2f}s")
        
        print("\nDetailed Results:")
        print("-" * 80)
        print(f"{'Algorithm':<20} {'Cost':<10} {'Routes':<8} {'Time(s)':<8} {'Efficiency':<12}")
        print("-" * 80)
        
        for _, row in sorted_df.iterrows():
            if row['Success']:
                print(f"{row['Algorithm']:<20} ${row['Total Cost']:<9.2f} "
                      f"{row['Number of Routes']:<8} {row['Execution Time (s)']:<8.2f} "
                      f"{row['Route Efficiency']:<12.2f}")
            else:
                print(f"{row['Algorithm']:<20} {'FAILED':<10} {'-':<8} {'-':<8} {'-':<12}")
        
        print("="*80)
    
    def get_best_algorithm(self, comparison_df, metric='Total Cost'):
        """
        Get the best algorithm based on a specific metric
        
        Args:
            comparison_df (pd.DataFrame): Comparison results
            metric (str): Metric to optimize for
            
        Returns:
            str: Best algorithm name
        """
        successful_results = comparison_df[comparison_df['Success'] == True]
        
        if successful_results.empty:
            return None
        
        if metric == 'Total Cost':
            best_idx = successful_results['Total Cost'].idxmin()
        elif metric == 'Execution Time (s)':
            best_idx = successful_results['Execution Time (s)'].idxmin()
        elif metric == 'Route Efficiency':
            best_idx = successful_results['Route Efficiency'].idxmin()
        else:
            best_idx = successful_results['Total Cost'].idxmin()
        
        return successful_results.loc[best_idx, 'Algorithm']
    
    def generate_performance_report(self, comparison_df, output_file=None):
        """
        Generate a detailed performance report
        
        Args:
            comparison_df (pd.DataFrame): Comparison results
            output_file (str): Optional file to save report
            
        Returns:
            str: Report content
        """
        report = []
        report.append("ROUTE OPTIMIZATION ALGORITHM PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall summary
        successful_algorithms = comparison_df[comparison_df['Success'] == True]
        report.append(f"Total Algorithms Tested: {len(comparison_df)}")
        report.append(f"Successful Algorithms: {len(successful_algorithms)}")
        report.append("")
        
        if not successful_algorithms.empty:
            # Best performers
            best_cost = successful_algorithms.loc[successful_algorithms['Total Cost'].idxmin()]
            fastest = successful_algorithms.loc[successful_algorithms['Execution Time (s)'].idxmin()]
            most_efficient = successful_algorithms.loc[successful_algorithms['Route Efficiency'].idxmin()]
            
            report.append("BEST PERFORMERS:")
            report.append(f"  Lowest Cost: {best_cost['Algorithm']} (${best_cost['Total Cost']:.2f})")
            report.append(f"  Fastest: {fastest['Algorithm']} ({fastest['Execution Time (s)']:.2f}s)")
            report.append(f"  Most Efficient: {most_efficient['Algorithm']} ({most_efficient['Route Efficiency']:.2f} km/passenger)")
            report.append("")
            
            # Detailed statistics
            report.append("DETAILED STATISTICS:")
            report.append("-" * 40)
            
            for _, row in successful_algorithms.iterrows():
                report.append(f"\n{row['Algorithm']}:")
                report.append(f"  Cost: ${row['Total Cost']:.2f}")
                report.append(f"  Distance: {row['Total Distance (km)']:.2f} km")
                report.append(f"  Routes: {row['Number of Routes']}")
                report.append(f"  Avg Passengers/Route: {row['Average Passengers per Route']:.1f}")
                report.append(f"  Execution Time: {row['Execution Time (s)']:.2f}s")
                report.append(f"  Efficiency: {row['Route Efficiency']:.2f} km/passenger")
        
        report_content = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            logger.info(f"Performance report saved to {output_file}")
        
        return report_content 