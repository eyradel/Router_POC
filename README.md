# Ride Router - Staff Transport Optimization

A comprehensive staff transport route optimization application built with Streamlit, featuring SOM clustering algorithms and interactive mapping.

##  Project Structure

```
Router_poc/
├── app.py                          # Main application entry point
├── main.py                         # Original monolithic file (legacy)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .env                           # Environment variables (create this)
├── algorithms/                    # Algorithm implementations
│   ├── __init__.py
│   ├── som_cluster.py            # Self-Organizing Map clustering
│   └── route_optimizer.py        # Route optimization engine
├── maps/                          # Map visualization
│   ├── __init__.py
│   └── map_creator.py            # Interactive map creation
├── ui/                           # User interface components
│   ├── __init__.py
│   └── components.py             # UI components and styling
├── data/                         # Data handling
│   ├── __init__.py
│   └── sample_data.py           # Sample data generation
└── utils/                        # Utility functions
    ├── __init__.py
    └── metrics.py               # Metrics formatting and calculations
```

##  Features

- **SOM Clustering**: Self-Organizing Map algorithm for staff location clustering
- **Route Optimization**: Greedy algorithm for optimal route creation
- **Interactive Maps**: Folium-based maps with multiple layers and controls
- **Google Maps Integration**: Real-time route directions and distance calculations
- **Data Validation**: Comprehensive staff data validation and cleaning
- **Metrics Dashboard**: Detailed cost and distance analytics
- **Debug Mode**: Real-time logging and debugging tools
- **Responsive UI**: Modern, responsive web interface

##  Requirements

- Python 3.8+
- Google Maps API key (optional, for enhanced routing)
- All dependencies listed in `requirements.txt`

##  Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Router_poc
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```
   GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

##  Data Format

The application expects CSV files with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| staff_id | string | Unique identifier for each staff member |
| name | string | Staff member's name |
| latitude | float | Latitude coordinate (decimal degrees) |
| longitude | float | Longitude coordinate (decimal degrees) |
| address | string | Human-readable address description |

### Sample CSV Format:
```csv
staff_id,name,latitude,longitude,address
1,John Doe,5.5826,-0.1435,Adabraka
2,Jane Smith,5.5926,-0.1335,Osu
3,Bob Johnson,5.5726,-0.1535,Cantonments
```

##  Configuration

### Route Parameters

- **Cluster Grid Size**: Controls the number of potential clusters (2-5)
- **Cluster Radius**: Controls how spread out the clusters are (0.5-2.0)
- **Learning Rate**: Controls how quickly the clustering algorithm adapts (0.1-1.0)

### Application Settings

- **Max Passengers per Route**: 4 (configurable in code)
- **Min Passengers per Route**: 3 (configurable in code)
- **Cost per Kilometer**: GHC 2.5 (configurable in code)
- **Office Location**: Accra, Ghana (configurable in code)

##  Modular Architecture

### Algorithms Module (`algorithms/`)

- **SOMCluster**: Self-Organizing Map implementation for clustering
- **RouteOptimizer**: Route optimization engine with validation

### Maps Module (`maps/`)

- **MapCreator**: Interactive map creation with multiple layers
- Google Maps integration for route directions

### UI Module (`ui/`)

- **Components**: Reusable UI components and styling
- Responsive design with Bootstrap integration

### Data Module (`data/`)

- **Sample Data**: Generation of test data for development
- CSV template generation

### Utils Module (`utils/`)

- **Metrics**: Formatting and calculation utilities
- Summary generation for analytics

##  Usage

1. **Start the application**: Run `streamlit run app.py`
2. **Choose data input**: Upload CSV or use sample data
3. **Configure parameters**: Adjust clustering and optimization settings
4. **Optimize routes**: Click "Optimize Routes" to generate routes
5. **View results**: Interactive map and detailed metrics dashboard

##  Logging

The application includes comprehensive logging:

- **Log file**: `router_app.log`
- **Log levels**: INFO, DEBUG, WARNING, ERROR
- **Debug panel**: Real-time log viewing in the sidebar

##  Debugging

Enable debug mode in the sidebar to:
- View real-time log entries
- Monitor application performance
- Track user interactions
- Debug optimization issues

##  Customization

### Adding New Algorithms

1. Create new algorithm file in `algorithms/`
2. Implement required interface methods
3. Update `algorithms/__init__.py`
4. Import in `app.py`

### Customizing UI

1. Modify components in `ui/components.py`
2. Update CSS styles in the `load_css()` function
3. Add new UI elements as needed

### Extending Maps

1. Add new map features in `maps/map_creator.py`
2. Implement additional map layers
3. Customize marker styles and popups

##  API Reference

### RouteOptimizer

```python
optimizer = RouteOptimizer(
    office_location={'lat': 5.5826, 'lon': -0.1435},
    max_passengers=4,
    min_passengers=3,
    cost_per_km=2.5
)

# Validate data
valid_data = optimizer.validate_staff_data(df)

# Create clusters
clustered_data = optimizer.create_clusters(
    staff_data, 
    grid_size=3, 
    sigma=1.0, 
    learning_rate=0.5
)

# Optimize routes
routes = optimizer.optimize_routes(clustered_data)
```

### MapCreator

```python
map_creator = MapCreator(
    office_location={'lat': 5.5826, 'lon': -0.1435},
    google_maps_client=gmaps_client
)

# Create interactive map
map_obj = map_creator.create_map(routes)
```

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request
##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Support

For support and questions:
- Check the debug logs
- Review the documentation
- Open an issue on GitHub

##  Migration from Legacy

To migrate from the original `main.py`:

1. The new `app.py` provides the same functionality
2. All features are preserved in the modular structure
3. Enhanced logging and error handling
4. Better code organization and maintainability

The original `main.py` is kept for reference but `app.py` is the recommended entry point.
