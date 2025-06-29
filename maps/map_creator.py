import folium
import polyline
from folium.plugins import GroupedLayerControl
from datetime import datetime
import logging
import streamlit as st

logger = logging.getLogger(__name__)

class MapCreator:
    """
    Map creation and visualization for route display
    """
    
    def __init__(self, office_location, google_maps_client=None):
        """
        Initialize map creator
        
        Args:
            office_location (dict): Office coordinates {'lat': float, 'lon': float}
            google_maps_client: Google Maps API client
        """
        self.office_location = office_location
        self.gmaps = google_maps_client
        
    def get_route_directions(self, origin, destination, waypoints=None):
        """
        Get route directions using Google Maps Directions API
        
        Args:
            origin (str): Origin coordinates
            destination (str): Destination coordinates
            waypoints (list): List of waypoint coordinates
            
        Returns:
            dict: Route information including polyline, distance, and duration
        """
        if not self.gmaps:
            logger.warning("Google Maps client not available, skipping route directions")
            return None
            
        try:
            logger.debug(f"Getting directions from {origin} to {destination}")
            if waypoints:
                waypoints = [f"{point['lat']},{point['lng']}" for point in waypoints]
                directions = self.gmaps.directions(
                    origin,
                    destination,
                    waypoints=waypoints,
                    optimize_waypoints=True,
                    mode="driving",
                    departure_time=datetime.now()
                )
            else:
                directions = self.gmaps.directions(
                    origin,
                    destination,
                    mode="driving",
                    departure_time=datetime.now()
                )

            if directions:
                route = directions[0]
                route_polyline = route['overview_polyline']['points']
                duration = sum(leg['duration']['value'] for leg in route['legs'])
                distance = sum(leg['distance']['value'] for leg in route['legs'])
                
                logger.debug(f"Route directions obtained: distance={distance/1000:.2f}km, duration={duration/60:.0f}min")
                return {
                    'polyline': route_polyline,
                    'duration': duration,
                    'distance': distance,
                    'directions': directions
                }
            else:
                logger.warning("No directions returned from Google Maps API")
                return None
        except Exception as e:
            logger.error(f"Error getting directions: {str(e)}")
            st.error(f"Error getting directions: {str(e)}")
            return None

    def create_map(self, routes):
        """
        Create an interactive map with multiple layer controls and satellite imagery
        
        Args:
            routes (dict): Dictionary of routes to display
            
        Returns:
            folium.Map: Interactive map object
        """
        logger.info(f"Creating map with {len(routes)} routes")
        try:
            # Initialize base map
            m = folium.Map(
                location=[self.office_location['lat'], self.office_location['lon']],
                zoom_start=13,
                control_scale=True
            )
            
            # Add multiple tile layers
            folium.TileLayer(
                'cartodbpositron',
                name='Street Map',
                control=True
            ).add_to(m)
            
            folium.TileLayer(
                'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Satellite View',
                control=True
            ).add_to(m)
            
            # Create office marker group
            office_group = folium.FeatureGroup(name='Office Location', show=True)
            folium.Marker(
                [self.office_location['lat'], self.office_location['lon']],
                popup=folium.Popup(
                    'Main Office',
                    max_width=200
                ),
                icon=folium.Icon(
                    color='red',
                    icon='building',
                    prefix='fa'
                ),
                tooltip="Office Location"
            ).add_to(office_group)
            office_group.add_to(m)
            
            # Define colors for routes
            colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred',
                     'beige', 'darkblue', 'darkgreen', 'cadetblue']
            
            # Create route groups
            for route_idx, (route_name, group) in enumerate(routes.items()):
                color = colors[route_idx % len(colors)]
                route_group = folium.FeatureGroup(
                    name=f"{route_name}",
                    show=True,
                    control=True
                )
                
                logger.debug(f"Processing route {route_name} with {len(group)} passengers")
                
                # Prepare waypoints for Google Maps
                waypoints = [
                    {'lat': staff['latitude'], 'lng': staff['longitude']}
                    for staff in group
                ]
                
                # Get route from Google Maps
                route_data = self.get_route_directions(
                    f"{waypoints[0]['lat']},{waypoints[0]['lng']}",
                    f"{self.office_location['lat']},{self.office_location['lon']}",
                    waypoints[1:] if len(waypoints) > 1 else None
                )
                
                if route_data:
                    # Add route polyline
                    route_coords = polyline.decode(route_data['polyline'])
                    
                    # Create route path with popup
                    route_line = folium.PolyLine(
                        route_coords,
                        weight=4,
                        color=color,
                        opacity=0.8,
                        popup=folium.Popup(
                            f"""
                            <div style='font-family: Arial; font-size: 12px;'>
                                <b>{route_name}</b><br>
                                Distance: {route_data['distance']/1000:.2f} km<br>
                                Duration: {route_data['duration']/60:.0f} min<br>
                                Passengers: {len(group)}
                            </div>
                            """,
                            max_width=200
                        )
                    )
                    route_line.add_to(route_group)
                    
                    # Add staff markers
                    for idx, staff in enumerate(group, 1):
                        # Create detailed popup content
                        popup_content = f"""
                        <div style='font-family: Arial; font-size: 12px;'>
                            <b>{staff['name']}</b><br>
                            Address: {staff['address']}<br>
                            Stop #{idx}<br>
                            Distance to office: {staff['distance_to_office']:.2f} km<br>
                            Pick-up order: {idx} of {len(group)}
                        </div>
                        """
                        
                        # Add staff marker
                        folium.CircleMarker(
                            location=[staff['latitude'], staff['longitude']],
                            radius=8,
                            popup=folium.Popup(
                                popup_content,
                                max_width=200
                            ),
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.7,
                            weight=2,
                            tooltip=f"Stop #{idx}: {staff['name']}"
                        ).add_to(route_group)
                
                route_group.add_to(m)
            
            # Add layer controls
            folium.LayerControl(
                position='topright',
                collapsed=False,
                autoZIndex=True
            ).add_to(m)
            
            # Add fullscreen control
            folium.plugins.Fullscreen(
                position='topleft',
                title='Fullscreen',
                title_cancel='Exit fullscreen',
                force_separate_button=True
            ).add_to(m)
            
            # Add search control
            folium.plugins.Search(
                layer=office_group,
                search_label='name',
                position='topleft'
            ).add_to(m)
            
            logger.info("Map created successfully")
            return m
            
        except Exception as e:
            logger.error(f"Error creating map: {str(e)}")
            st.error(f"Error creating map: {str(e)}")
            return None 