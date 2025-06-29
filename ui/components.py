import streamlit as st
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_css():
    """Load CSS styles for the application"""
    logger.debug("Loading CSS styles")
    st.markdown(
        """
        <meta charset="UTF-8">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.19.1/css/mdb.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.2/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
        <link rel="icon" href="https://www.4th-ir.com/favicon.ico">
        
        <title>4thir-POC-repo</title>
        <meta name="title" content="4thir-POC-repo" />
        <meta name="description" content="view our proof of concepts" />

        <!-- Open Graph / Facebook -->
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://4thir-poc-repositoty.streamlit.app/" />
        <meta property="og:title" content="4thir-POC-repo" />
        <meta property="og:description" content="view our proof of concepts" />
        <meta property="og:image" content="https://www.4th-ir.com/favicon.ico" />

        <!-- Twitter -->
        <meta property="twitter:card" content="summary_large_image" />
        <meta property="twitter:url" content="https://4thir-poc-repositoty.streamlit.app/" />
        <meta property="twitter:title" content="4thir-POC-repo" />
        <meta property="twitter:description" content="view our proof of concepts" />
        <meta property="twitter:image" content="https://www.4th-ir.com/favicon.ico" />

        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        """,
        unsafe_allow_html=True,
    )

    
    st.markdown(
        """
        <style>
            header {visibility: hidden;}
            .main {
                margin-top: -20px;
                padding-top: 10px;
            }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .navbar {
                padding: 1rem;
                margin-bottom: 2rem;
                background-color: #4267B2;
                color: white;
            }
            .card {
                padding: 1rem;
                margin-bottom: 1rem;
                transition: transform 0.2s;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .card:hover {
                transform: scale(1.02);
            }
            .metric-card {
               
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem;
                
            }
            .search-box {
                margin-bottom: 1rem;
                padding: 0.5rem;
                border-radius: 4px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

def create_navbar():
    """Create the navigation bar with a fixed, well-aligned logo"""
    logger.debug("Creating navigation bar")
    st.markdown(
        """
        <nav class="navbar fixed-top navbar-expand-lg navbar-dark bg-bluext-bold shadow-sm">
            <a class="navbar-brand text-primary d-flex align-items-center" href="#" target="_blank" style="font-weight: bold; font-size: 1.3rem;">
                <img src="" alt="4th-ir logo" style="height:32px; margin-right:10px; vertical-align:middle;">Ride Router
            </a>
        </nav>
        """,
        unsafe_allow_html=True
    )

def show_metrics_dashboard(metrics):
    """Display metrics dashboard with formatted cards"""
    logger.debug("Displaying metrics dashboard")
    st.markdown("""
        <div class="card" style="padding: 1rem;  border-radius: 8px; margin-bottom: 1rem;">
            <h3 style="margin-bottom: 1rem;">Route Metrics Dashboard</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
    """, unsafe_allow_html=True)
    
    for key, value in metrics.items():
        st.markdown(f"""
            <div class="metric-card card">
                <h4>{key.replace('_', ' ').title()}</h4>
                <p style="font-size: 1.5rem; font-weight: bold;">{value}</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

def create_sidebar_config():
    """Create the sidebar configuration panel"""
    logger.debug("Creating sidebar configuration")
    
    st.header("Configuration")
    
    st.subheader("1. Data Input")
    data_option = st.radio(
        "Choose data input method",
        ["Upload CSV", "Use Sample Data"]
    )
    
    st.subheader("2. Route Parameters")
    grid_size = st.slider(
        "Cluster Grid Size",
        min_value=2,
        max_value=5,
        value=3,
        help="Controls the number of potential clusters"
    )
    
    sigma = st.slider(
        "Cluster Radius",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Controls how spread out the clusters are"
    )
    
    learning_rate = st.slider(
        "Learning Rate",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Controls how quickly the clustering algorithm adapts"
    )
    
    # Debug section
    st.subheader("3. Debug & Logs")
    debug_mode = st.checkbox("Show Debug Information")
    
    if debug_mode:
        logger.info("Debug mode enabled by user")
        st.info("Debug mode is active - check console for detailed logs")
        
        # Show recent log entries
        try:
            with open('router_app.log', 'r') as f:
                log_lines = f.readlines()
                recent_logs = log_lines[-20:]  # Show last 20 lines
                st.text_area("Recent Log Entries:", value=''.join(recent_logs), height=200)
        except FileNotFoundError:
            st.warning("Log file not found yet")
    
    return {
        'data_option': data_option,
        'grid_size': grid_size,
        'sigma': sigma,
        'learning_rate': learning_rate,
        'debug_mode': debug_mode
    }

def create_file_uploader():
    """Create file uploader component"""
    logger.debug("Creating file uploader")
    uploaded_file = st.file_uploader(
        "Upload staff locations CSV",
        type=["csv"],
        help="""
        Required columns:
        - staff_id (unique identifier)
        - name (staff name)
        - latitude (valid coordinate)
        - longitude (valid coordinate)
        - address (location description)
        """
    )
    return uploaded_file

def create_optimize_button():
    """Create the optimize routes button"""
    logger.debug("Creating optimize button")
    return st.button(" Optimize Routes", type="primary")

def display_staff_directory(staff_data):
    """Display staff directory with search functionality"""
    logger.debug("Displaying staff directory")
    st.subheader("Staff Directory")
    search_term = st.text_input("Search staff by name or address")
    
    display_df = staff_data[['name', 'address','latitude','longitude']].copy()
    if search_term:
        logger.debug(f"Filtering staff data with search term: {search_term}")
        mask = (
            display_df['name'].str.contains(search_term, case=False) |
            display_df['address'].str.contains(search_term, case=False)
        )
        display_df = display_df[mask]
    
    st.dataframe(display_df, height=300)

def display_route_details(routes, optimizer):
    """Display detailed route information"""
    logger.debug("Displaying route details")
    st.subheader("Route Details")
    
    metrics = {
        'total_distance': 0,
        'total_cost': 0,
        'total_duration': 0
    }
    
    for route_name, route in routes.items():
        with st.expander(f" {route_name}"):
            distance, cost = optimizer.calculate_route_metrics(route)
            metrics['total_distance'] += distance
            metrics['total_cost'] += cost
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Distance", f"{distance:.2f} km")
            with col2:
                st.metric("Cost", f"GHC{cost:.2f}")
            
            st.dataframe(
                pd.DataFrame(route)[['name', 'address', 'distance_to_office']],
                height=200
            )
    
    show_metrics_dashboard({
        'Total Distance': f"{metrics['total_distance']:.2f} km",
        'Total Cost': f"GHC{metrics['total_cost']:.2f}",
        'Number of Routes': len(routes),
        'Average Cost/Route': f"GHC{metrics['total_cost']/len(routes):.2f}"
    })
    
    return metrics 