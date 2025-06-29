import pandas as pd
import numpy as np
from collections import defaultdict
from geopy.distance import geodesic
from sklearn.preprocessing import MinMaxScaler
import logging
from .som_cluster import SOMCluster

logger = logging.getLogger(__name__)

class RouteOptimizer:
    """
    Route optimization engine using SOM clustering and greedy algorithms
    """
    
    def __init__(self, office_location, max_passengers=4, min_passengers=3, cost_per_km=2.5):
        """
        Initialize route optimizer
        
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

    def create_clusters(self, staff_data, grid_size=3, sigma=1.0, learning_rate=0.5):
        """
        Create clusters based on staff locations using SOM
        
        Args:
            staff_data (pandas.DataFrame): Staff location data
            grid_size (int): SOM grid size
            sigma (float): SOM neighborhood radius
            learning_rate (float): SOM learning rate
            
        Returns:
            pandas.DataFrame: Staff data with cluster assignments
        """
        logger.info(f"Creating clusters with grid_size={grid_size}, sigma={sigma}, learning_rate={learning_rate}")
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
            
            # Initialize and train SOM
            som = SOMCluster(
                input_len=3,
                grid_size=grid_size,
                sigma=sigma,
                learning_rate=learning_rate
            )
            
            som.train(normalized_data)
            
            # Assign clusters
            staff_data['cluster'] = [som.get_cluster(loc) for loc in normalized_data]
            
            # Handle small clusters
            self._handle_small_clusters(staff_data)
            
            cluster_counts = staff_data['cluster'].value_counts()
            logger.info(f"Clustering completed. Cluster distribution: {cluster_counts.to_dict()}")
            
            return staff_data
            
        except Exception as e:
            logger.error(f"Error in clustering: {str(e)}")
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
                        nearest_cluster = min(distances, key=lambda x: x[1])[0]
                        staff_data.at[idx, 'cluster'] = nearest_cluster
                        logger.debug(f"Moved staff {row['name']} from cluster {small_cluster} to {nearest_cluster}")

    def optimize_routes(self, staff_data):
        """
        Optimize routes within each cluster
        
        Args:
            staff_data (pandas.DataFrame): Staff data with cluster assignments
            
        Returns:
            dict: Dictionary of optimized routes
        """
        logger.info("Starting route optimization")
        routes = defaultdict(list)
        route_counter = 0
        
        try:
            if 'distance_to_office' not in staff_data.columns:
                staff_data['distance_to_office'] = staff_data.apply(
                    lambda row: geodesic(
                        (row['latitude'], row['longitude']),
                        (self.office_location['lat'], self.office_location['lon'])
                    ).km,
                    axis=1
                )
            
            for cluster_id in staff_data['cluster'].unique():
                cluster_group = staff_data[staff_data['cluster'] == cluster_id].copy()
                logger.info(f"Processing cluster {cluster_id} with {len(cluster_group)} staff members")
                
                while len(cluster_group) >= self.MIN_PASSENGERS:
                    current_route = []
                    remaining = cluster_group.copy()
                    
                    # Start with furthest person from office
                    start_person = remaining.nlargest(1, 'distance_to_office').iloc[0]
                    current_route.append(start_person.to_dict())
                    remaining = remaining.drop(start_person.name)
                    
                    while len(current_route) < self.MAX_PASSENGERS and not remaining.empty:
                        last_point = current_route[-1]
                        
                        remaining['temp_distance'] = remaining.apply(
                            lambda row: geodesic(
                                (last_point['latitude'], last_point['longitude']),
                                (row['latitude'], row['longitude'])
                            ).km,
                            axis=1
                        )
                        
                        next_person = remaining.nsmallest(1, 'temp_distance').iloc[0]
                        current_route.append(next_person.to_dict())
                        remaining = remaining.drop(next_person.name)
                        
                        if len(current_route) >= self.MIN_PASSENGERS:
                            break
                    
                    if len(current_route) >= self.MIN_PASSENGERS:
                        route_name = f'Route {route_counter + 1}'
                        routes[route_name] = current_route
                        route_counter += 1
                        assigned_ids = [p['staff_id'] for p in current_route]
                        cluster_group = cluster_group[~cluster_group['staff_id'].isin(assigned_ids)]
                        logger.info(f"Created {route_name} with {len(current_route)} passengers")
                    else:
                        break
                
                if len(cluster_group) > 0:
                    logger.info(f"Assigning {len(cluster_group)} remaining staff from cluster {cluster_id}")
                    self._assign_remaining_staff(cluster_group, routes)
            
            logger.info(f"Route optimization completed. Created {len(routes)} routes")
            return routes
            
        except Exception as e:
            logger.error(f"Error in route optimization: {str(e)}")
            return {}

    def _assign_remaining_staff(self, remaining_staff, routes):
        """
        Assign remaining staff to existing routes
        
        Args:
            remaining_staff (pandas.DataFrame): Remaining staff to assign
            routes (dict): Existing routes
        """
        logger.info(f"Assigning {len(remaining_staff)} remaining staff to existing routes")
        if not isinstance(remaining_staff, pd.DataFrame):
            logger.warning("remaining_staff is not a DataFrame")
            return
            
        for idx, row in remaining_staff.iterrows():
            best_route = None
            min_detour = float('inf')
            staff_dict = row.to_dict()
            
            for route_name, route_group in routes.items():
                if len(route_group) < self.MAX_PASSENGERS:
                    # Calculate current route distance
                    current_distance = self.calculate_route_metrics(route_group)[0]
                    
                    # Calculate new route distance with added staff member
                    test_route = route_group.copy()
                    test_route.append(staff_dict)
                    new_distance = self.calculate_route_metrics(test_route)[0]
                    
                    # Calculate detour distance
                    detour = new_distance - current_distance
                    
                    if detour < min_detour:
                        min_detour = detour
                        best_route = route_name
            
            if best_route:
                routes[best_route].append(staff_dict)
                logger.debug(f"Assigned staff {staff_dict.get('name', 'Unknown')} to {best_route}")

    def calculate_route_metrics(self, route):
        """
        Calculate total distance and cost for a route
        
        Args:
            route (list): List of dictionaries containing staff information
            
        Returns:
            tuple: (total_distance in km, total_cost in currency units)
        """
        if not route:
            logger.debug("Empty route provided, returning zero metrics")
            return 0, 0
        
        try:
            total_distance = 0
            points = [(p['latitude'], p['longitude']) for p in route]
            points.append((self.office_location['lat'], self.office_location['lon']))
            
            for i in range(len(points) - 1):
                distance = geodesic(points[i], points[i + 1]).km
                total_distance += distance
            
            cost = total_distance * self.COST_PER_KM
            logger.debug(f"Route metrics calculated: distance={total_distance:.2f}km, cost={cost:.2f}")
            return total_distance, cost
            
        except Exception as e:
            logger.error(f"Error calculating route metrics: {str(e)}")
            return 0, 0

    def get_route_summary(self, route):
        """
        Get a detailed summary of route metrics
        
        Args:
            route (list): List of dictionaries containing staff information
            
        Returns:
            dict: Dictionary containing route metrics
        """
        try:
            distance, cost = self.calculate_route_metrics(route)
            
            summary = {
                'total_distance': distance,
                'total_cost': cost,
                'passenger_count': len(route),
                'cost_per_passenger': cost / len(route) if route else 0,
                'distance_per_passenger': distance / len(route) if route else 0,
                'start_point': route[0]['address'] if route else None,
                'end_point': 'Office',
                'stops': len(route)
            }
            
            logger.debug(f"Route summary: {summary}")
            return summary
        except Exception as e:
            logger.error(f"Error generating route summary: {str(e)}")
            return None

    def calculate_total_metrics(self, routes):
        """
        Calculate total metrics for all routes
        
        Args:
            routes (dict): Dictionary of routes
            
        Returns:
            dict: Dictionary containing total metrics
        """
        try:
            total_metrics = {
                'total_distance': 0,
                'total_cost': 0,
                'total_passengers': 0,
                'number_of_routes': len(routes),
                'average_route_distance': 0,
                'average_route_cost': 0,
                'average_passengers_per_route': 0
            }
            
            for route_name, route in routes.items():
                distance, cost = self.calculate_route_metrics(route)
                total_metrics['total_distance'] += distance
                total_metrics['total_cost'] += cost
                total_metrics['total_passengers'] += len(route)
            
            if routes:
                total_metrics['average_route_distance'] = total_metrics['total_distance'] / len(routes)
                total_metrics['average_route_cost'] = total_metrics['total_cost'] / len(routes)
                total_metrics['average_passengers_per_route'] = total_metrics['total_passengers'] / len(routes)
            
            total_metrics['cost_per_passenger'] = (
                total_metrics['total_cost'] / total_metrics['total_passengers']
                if total_metrics['total_passengers'] > 0 else 0
            )
            
            logger.info(f"Total metrics calculated: {total_metrics}")
            return total_metrics
            
        except Exception as e:
            logger.error(f"Error calculating total metrics: {str(e)}")
            return None 