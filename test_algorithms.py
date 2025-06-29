#!/usr/bin/env python3
"""
Test script for enhanced route optimization algorithms
"""

import pandas as pd
import numpy as np
import logging
from algorithms import (
    SOMCluster, 
    RouteOptimizer, 
    GeneticRouteOptimizer, 
    AntColonyOptimizer, 
    HybridRouteOptimizer,
    AlgorithmComparison
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data(n_staff=20):
    """Create sample staff data for testing"""
    np.random.seed(42)
    
    # Office location (Accra, Ghana)
    office_location = {
        'lat': 5.582636441579255,
        'lon': -0.143551646497661
    }
    
    # Generate random staff locations around Accra
    staff_data = []
    for i in range(n_staff):
        # Random coordinates within reasonable distance from Accra
        lat = office_location['lat'] + np.random.uniform(-0.1, 0.1)
        lon = office_location['lon'] + np.random.uniform(-0.1, 0.1)
        
        staff_data.append({
            'staff_id': f'EMP{i+1:03d}',
            'name': f'Employee {i+1}',
            'latitude': lat,
            'longitude': lon,
            'address': f'Address {i+1}, Accra, Ghana'
        })
    
    return pd.DataFrame(staff_data), office_location

def test_som_clustering():
    """Test enhanced SOM clustering"""
    print("\n" + "="*60)
    print("TESTING ENHANCED SOM CLUSTERING")
    print("="*60)
    
    staff_data, office_location = create_sample_data(15)
    
    # Calculate distances to office
    staff_data['distance_to_office'] = staff_data.apply(
        lambda row: np.sqrt(
            (row['latitude'] - office_location['lat'])**2 + 
            (row['longitude'] - office_location['lon'])**2
        ) * 111,  # Rough conversion to km
        axis=1
    )
    
    # Prepare data for clustering
    locations = staff_data[['latitude', 'longitude', 'distance_to_office']].values
    
    # Test different initialization methods
    initialization_methods = ['random', 'linear', 'pca']
    
    for init_method in initialization_methods:
        print(f"\nTesting SOM with {init_method} initialization:")
        
        som = SOMCluster(
            input_len=3,
            grid_size=3,
            sigma=1.0,
            learning_rate=0.5,
            initialization=init_method,
            random_state=42
        )
        
        # Train SOM
        som.train(locations, epochs=1000, verbose=False)
        
        # Get cluster assignments
        clusters = [som.get_cluster(loc) for loc in locations]
        staff_data[f'cluster_{init_method}'] = clusters
        
        # Calculate quality metrics
        quality_metrics = som.get_cluster_quality_metrics(locations)
        
        print(f"  Quantization Error: {quality_metrics['quantization_error']:.6f}")
        print(f"  Topographic Error: {quality_metrics['topographic_error']:.6f}")
        if quality_metrics['silhouette_score'] is not None:
            print(f"  Silhouette Score: {quality_metrics['silhouette_score']:.6f}")
        
        # Show cluster distribution
        cluster_counts = staff_data[f'cluster_{init_method}'].value_counts().sort_index()
        print(f"  Cluster Distribution: {cluster_counts.to_dict()}")

def test_genetic_algorithm():
    """Test genetic algorithm optimization"""
    print("\n" + "="*60)
    print("TESTING GENETIC ALGORITHM OPTIMIZATION")
    print("="*60)
    
    staff_data, office_location = create_sample_data(12)
    
    optimizer = GeneticRouteOptimizer(
        office_location=office_location,
        max_passengers=4,
        min_passengers=3,
        cost_per_km=2.5,
        random_state=42
    )
    
    # Run optimization
    routes = optimizer.optimize(
        staff_data,
        population_size=50,
        generations=100,
        verbose=True
    )
    
    # Get optimization stats
    stats = optimizer.get_optimization_stats()
    
    print(f"\nOptimization Results:")
    print(f"Best Fitness: {stats['best_fitness']:.4f}")
    print(f"Generations: {stats['generations']}")
    print(f"Number of Routes: {len(routes)}")
    
    # Show route details
    for route_name, route in routes.items():
        print(f"\n{route_name}:")
        for staff in route:
            print(f"  - {staff['name']} ({staff['latitude']:.4f}, {staff['longitude']:.4f})")

def test_ant_colony():
    """Test ant colony optimization"""
    print("\n" + "="*60)
    print("TESTING ANT COLONY OPTIMIZATION")
    print("="*60)
    
    staff_data, office_location = create_sample_data(12)
    
    optimizer = AntColonyOptimizer(
        office_location=office_location,
        max_passengers=4,
        min_passengers=3,
        cost_per_km=2.5,
        random_state=42
    )
    
    # Run optimization
    routes = optimizer.optimize(
        staff_data,
        n_ants=30,
        n_iterations=50,
        verbose=True
    )
    
    # Get optimization stats
    stats = optimizer.get_optimization_stats()
    
    print(f"\nOptimization Results:")
    print(f"Best Cost: {stats['best_cost']:.4f}")
    print(f"Iterations: {stats['iterations']}")
    print(f"Number of Routes: {len(routes)}")
    
    # Show route details
    for route_name, route in routes.items():
        print(f"\n{route_name}:")
        for staff in route:
            print(f"  - {staff['name']} ({staff['latitude']:.4f}, {staff['longitude']:.4f})")

def test_hybrid_optimizer():
    """Test hybrid optimizer"""
    print("\n" + "="*60)
    print("TESTING HYBRID OPTIMIZER")
    print("="*60)
    
    staff_data, office_location = create_sample_data(15)
    
    optimizer = HybridRouteOptimizer(
        office_location=office_location,
        max_passengers=4,
        min_passengers=3,
        cost_per_km=2.5,
        random_state=42
    )
    
    # Test different strategies
    strategies = ['fast', 'balanced', 'full']
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy:")
        
        routes = optimizer.optimize(
            staff_data,
            strategy=strategy,
            verbose=True
        )
        
        # Calculate total cost
        total_cost = 0
        for route in routes.values():
            route_distance = 0
            current_location = (office_location['lat'], office_location['lon'])
            
            for staff in route:
                staff_location = (staff['latitude'], staff['longitude'])
                route_distance += np.sqrt(
                    (staff_location[0] - current_location[0])**2 + 
                    (staff_location[1] - current_location[1])**2
                ) * 111
                current_location = staff_location
            
            # Return to office
            route_distance += np.sqrt(
                (current_location[0] - office_location['lat'])**2 + 
                (current_location[1] - office_location['lon'])**2
            ) * 111
            
            total_cost += route_distance * 2.5
        
        print(f"  Total Cost: ${total_cost:.2f}")
        print(f"  Number of Routes: {len(routes)}")

def test_algorithm_comparison():
    """Test algorithm comparison"""
    print("\n" + "="*60)
    print("TESTING ALGORITHM COMPARISON")
    print("="*60)
    
    staff_data, office_location = create_sample_data(20)
    
    # Create comparison object
    comparison = AlgorithmComparison(
        office_location=office_location,
        max_passengers=4,
        min_passengers=3,
        cost_per_km=2.5
    )
    
    # Run comparison
    results = comparison.compare_algorithms(staff_data, verbose=True)
    
    # Generate performance report
    report = comparison.generate_performance_report(results)
    print("\n" + report)
    
    # Get best algorithm
    best_algorithm = comparison.get_best_algorithm(results, 'Total Cost')
    print(f"\nBest algorithm for cost optimization: {best_algorithm}")
    
    fastest_algorithm = comparison.get_best_algorithm(results, 'Execution Time (s)')
    print(f"Fastest algorithm: {fastest_algorithm}")

def main():
    """Run all tests"""
    print("ENHANCED ROUTE OPTIMIZATION ALGORITHMS TEST")
    print("="*80)
    
    try:
        # Test individual algorithms
        test_som_clustering()
        test_genetic_algorithm()
        test_ant_colony()
        test_hybrid_optimizer()
        
        # Test algorithm comparison
        test_algorithm_comparison()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 