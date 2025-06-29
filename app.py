import streamlit as st
import pandas as pd
import googlemaps
import os
import logging
from dotenv import load_dotenv
from streamlit_folium import st_folium

# Import our modular components
from algorithms import RouteOptimizer
from maps import MapCreator
from ui import (
    load_css, create_navbar, create_sidebar_config, 
    create_file_uploader, create_optimize_button,
    display_staff_directory, display_route_details
)
from data import generate_sample_staff_data
from utils import create_metrics_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('router_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(layout="wide")

def initialize_optimizer():
    """Initialize the route optimizer with Google Maps API"""
    try:
        google_maps_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not google_maps_key:
            logger.warning("Google Maps API key not found in environment variables")
            google_maps_key = st.secrets.get("GOOGLE_MAPS_API_KEY")
        
        # Office location in Accra, Ghana
        office_location = {
            'lat': 5.582636441579255,
            'lon': -0.143551646497661
        }
        
        # Initialize Google Maps client
        gmaps_client = None
        if google_maps_key:
            try:
                gmaps_client = googlemaps.Client(key=google_maps_key)
                logger.info("Google Maps client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Google Maps client: {e}")
        
        # Initialize route optimizer
        optimizer = RouteOptimizer(
            office_location=office_location,
            max_passengers=4,
            min_passengers=3,
            cost_per_km=2.5
        )
        
        # Initialize map creator
        map_creator = MapCreator(
            office_location=office_location,
            google_maps_client=gmaps_client
        )
        
        logger.info("Route optimizer and map creator initialized successfully")
        return optimizer, map_creator
        
    except Exception as e:
        logger.error(f"Error initializing optimizer: {str(e)}")
        st.error(f"Error: {str(e)}")
        return None, None

def main():
    """Main application function"""
    logger.info("Starting Ride Router application")
    
    # Load UI components
    load_css()
    create_navbar()
    
    # Initialize session state
    if 'optimization_done' not in st.session_state:
        st.session_state.optimization_done = False
        logger.debug("Initialized optimization_done session state")
    if 'staff_data' not in st.session_state:
        st.session_state.staff_data = None
        logger.debug("Initialized staff_data session state")
    if 'routes' not in st.session_state:
        st.session_state.routes = None
        logger.debug("Initialized routes session state")

    # Initialize optimizer and map creator
    optimizer, map_creator = initialize_optimizer()
    if not optimizer or not map_creator:
        st.error("Failed to initialize application components")
        return

    # Sidebar configuration
    with st.sidebar:
        config = create_sidebar_config()
        
        # Handle data input
        if config['data_option'] == "Upload CSV":
            logger.info("User selected CSV upload option")
            uploaded_file = create_file_uploader()
            
            if uploaded_file:
                logger.info(f"CSV file uploaded: {uploaded_file.name}")
                try:
                    df = pd.read_csv(uploaded_file)
                    logger.info(f"CSV file loaded with {len(df)} rows")
                    st.session_state.staff_data = optimizer.validate_staff_data(df)
                    if st.session_state.staff_data is not None:
                        st.success("Data validated successfully!")
                    else:
                        st.error("Data validation failed")
                except Exception as e:
                    logger.error(f"Error processing CSV file: {str(e)}")
                    st.error(f"Error: {str(e)}")
                    st.session_state.staff_data = None
        else:
            logger.info("User selected sample data option")
            if st.button("Load Sample Data"):
                logger.info("Loading sample data")
                sample_data = generate_sample_staff_data(20)
                if sample_data is not None:
                    st.session_state.staff_data = optimizer.validate_staff_data(sample_data)
                    if st.session_state.staff_data is not None:
                        st.success("Sample data loaded successfully!")
                    else:
                        st.error("Failed to validate sample data")
                else:
                    st.error("Failed to generate sample data")
        
        # Route optimization
        if create_optimize_button():
            logger.info("User clicked optimize routes button")
            if st.session_state.staff_data is not None:
                with st.spinner("Optimizing routes..."):
                    try:
                        logger.info(f"Starting route optimization with parameters: grid_size={config['grid_size']}, sigma={config['sigma']}, learning_rate={config['learning_rate']}")
                        
                        # Create clusters
                        clustered_data = optimizer.create_clusters(
                            st.session_state.staff_data,
                            grid_size=config['grid_size'],
                            sigma=config['sigma'],
                            learning_rate=config['learning_rate']
                        )
                        
                        if clustered_data is not None:
                            # Optimize routes
                            st.session_state.routes = optimizer.optimize_routes(clustered_data)
                            if st.session_state.routes:
                                st.session_state.optimization_done = True
                                logger.info(f"Route optimization completed successfully. Created {len(st.session_state.routes)} routes")
                                st.success("Routes optimized successfully!")
                            else:
                                logger.warning("Route optimization failed - no routes created")
                                st.error("Route optimization failed. Try different parameters.")
                        else:
                            logger.error("Clustering failed")
                            st.error("Clustering failed. Try different parameters.")
                    except Exception as e:
                        logger.error(f"Optimization error: {str(e)}")
                        st.error(f"Optimization error: {str(e)}")
            else:
                logger.warning("No staff data available for optimization")
                st.warning("Please upload valid staff data first.")

    # Main content area
    if st.session_state.staff_data is not None:
        logger.debug("Displaying main content area")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.session_state.optimization_done:
                logger.info("Displaying route map")
                st.subheader(" Route Map")
                m = map_creator.create_map(st.session_state.routes)
                if m is not None:
                    st_folium(m, width=800, height=600)
                    
                    st.info("""
                    **Map Controls:**
                    - Use the layer control (top right) to toggle routes and map type
                    - Click on markers for detailed information
                    - Zoom in/out using the mouse wheel or +/- buttons
                    """)
                else:
                    logger.error("Failed to create map")
        
        with col2:
            # Staff directory
            display_staff_directory(st.session_state.staff_data)
            
            # Route details
            if st.session_state.optimization_done:
                logger.info("Displaying route details")
                metrics = display_route_details(st.session_state.routes, optimizer)
                logger.info("Metrics dashboard displayed successfully")

    logger.info("Application session completed successfully")

if __name__ == "__main__":
    logger.info("Starting Ride Router application from main entry point")
    main() 