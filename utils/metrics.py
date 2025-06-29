import logging

logger = logging.getLogger(__name__)

def format_metrics(metrics):
    """
    Format metrics for display
    
    Args:
        metrics (dict): Dictionary containing metrics
        
    Returns:
        dict: Dictionary containing formatted metrics
    """
    try:
        formatted = {}
        for key, value in metrics.items():
            if 'distance' in key.lower():
                formatted[key] = f"{value:.2f} km"
            elif 'cost' in key.lower():
                formatted[key] = f"GHC{value:.2f}"
            elif 'average' in key.lower():
                if 'distance' in key.lower():
                    formatted[key] = f"{value:.2f} km"
                elif 'cost' in key.lower():
                    formatted[key] = f"GHC{value:.2f}"
                else:
                    formatted[key] = f"{value:.2f}"
            else:
                formatted[key] = value
        return formatted
        
    except Exception as e:
        logger.error(f"Error formatting metrics: {str(e)}")
        return None

def create_metrics_summary(routes, optimizer):
    """
    Create a complete metrics summary for display
    
    Args:
        routes (dict): Dictionary of routes
        optimizer: Route optimizer instance
        
    Returns:
        dict: Dictionary containing formatted metrics summary
    """
    try:
        metrics = optimizer.calculate_total_metrics(routes)
        if metrics:
            formatted_metrics = format_metrics(metrics)
            
            summary = {
                'Overview': {
                    'Total Routes': metrics['number_of_routes'],
                    'Total Passengers': metrics['total_passengers'],
                    'Total Distance': formatted_metrics['total_distance'],
                    'Total Cost': formatted_metrics['total_cost']
                },
                'Averages': {
                    'Average Route Distance': formatted_metrics['average_route_distance'],
                    'Average Route Cost': formatted_metrics['average_route_cost'],
                    'Average Passengers per Route': f"{metrics['average_passengers_per_route']:.1f}",
                    'Cost per Passenger': formatted_metrics['cost_per_passenger']
                },
                'Routes': {}
            }
            
            for route_name, route in routes.items():
                route_summary = optimizer.get_route_summary(route)
                if route_summary:
                    formatted_route_summary = format_metrics(route_summary)
                    summary['Routes'][route_name] = formatted_route_summary
            
            logger.info("Metrics summary created successfully")
            return summary
        return None
        
    except Exception as e:
        logger.error(f"Error creating metrics summary: {str(e)}")
        return None 