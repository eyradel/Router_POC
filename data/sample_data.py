import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_sample_staff_data(num_staff=20):
    """
    Generate sample staff location data for testing
    
    Args:
        num_staff (int): Number of staff members to generate
        
    Returns:
        pandas.DataFrame: Sample staff data
    """
    logger.info(f"Generating sample data for {num_staff} staff members")
    
    try:
        # Ghana locations around Accra
        locations = [
            'Adabraka', 'Osu', 'Cantonments', 'Airport Residential',
            'East Legon', 'Spintex', 'Tema', 'Teshie', 'Labadi',
            'Labone', 'Ridge', 'Roman Ridge', 'Dzorwulu', 'Abelemkpe',
            'North Kaneshie', 'Dansoman', 'Mamprobi', 'Chorkor',
            'Abeka', 'Achimota'
        ]
        
        # Generate random coordinates within Accra area
        latitudes = np.random.uniform(5.5526, 5.6126, num_staff)
        longitudes = np.random.uniform(-0.1735, -0.1135, num_staff)
        
        sample_data = pd.DataFrame({
            'staff_id': range(1, num_staff + 1),
            'name': [f'Employee {i}' for i in range(1, num_staff + 1)],
            'latitude': latitudes,
            'longitude': longitudes,
            'address': locations[:num_staff] if num_staff <= len(locations) else locations * (num_staff // len(locations) + 1)
        })
        
        logger.info(f"Successfully generated sample data with {len(sample_data)} records")
        return sample_data
        
    except Exception as e:
        logger.error(f"Error generating sample data: {str(e)}")
        return None

def get_sample_csv_template():
    """
    Get a template for the required CSV format
    
    Returns:
        str: CSV template content
    """
    template = """staff_id,name,latitude,longitude,address
1,John Doe,5.5826,-0.1435,Adabraka
2,Jane Smith,5.5926,-0.1335,Osu
3,Bob Johnson,5.5726,-0.1535,Cantonments
4,Alice Brown,5.6026,-0.1235,Airport Residential
5,Charlie Wilson,5.5626,-0.1635,East Legon"""
    
    return template 