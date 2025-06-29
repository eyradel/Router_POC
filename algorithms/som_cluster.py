import numpy as np
import numpy.random as rnd
import logging
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

class SOMCluster:
    """
    Enhanced Self-Organizing Map (SOM) clustering algorithm for staff location clustering
    with improved initialization, training, and convergence monitoring
    """
    
    def __init__(self, input_len, grid_size=3, sigma=1.0, learning_rate=0.5, 
                 initialization='random', random_state=None):
        """
        Initialize enhanced SOM cluster with parameters
        
        Args:
            input_len (int): Length of input vectors
            grid_size (int): Size of the SOM grid (grid_size x grid_size)
            sigma (float): Initial neighborhood radius
            learning_rate (float): Initial learning rate
            initialization (str): Weight initialization method ('random', 'pca', 'linear')
            random_state (int): Random seed for reproducibility
        """
        logger.info(f"Initializing Enhanced SOMCluster with grid_size={grid_size}, "
                   f"sigma={sigma}, learning_rate={learning_rate}, init={initialization}")
        
        self.grid_size = grid_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.input_len = input_len
        self.initialization = initialization
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
            rnd.seed(random_state)
            
        self.scaler = StandardScaler()
        self._init_weights()
        self.training_history = []
        
    def _init_weights(self):
        """Initialize SOM weights using different strategies"""
        logger.debug(f"Initializing SOM weights using {self.initialization} method")
        
        if self.initialization == 'random':
            self.weights = rnd.rand(self.grid_size, self.grid_size, self.input_len)
        elif self.initialization == 'linear':
            # Linear initialization along the first two principal components
            self.weights = np.zeros((self.grid_size, self.grid_size, self.input_len))
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    self.weights[i, j] = np.array([
                        (i / (self.grid_size - 1)) * 0.5 + 0.25,
                        (j / (self.grid_size - 1)) * 0.5 + 0.25,
                        0.5  # Default value for distance_to_office
                    ])
        elif self.initialization == 'pca':
            # PCA-based initialization (placeholder - will be set during training)
            self.weights = rnd.rand(self.grid_size, self.grid_size, self.input_len)
        else:
            raise ValueError(f"Unknown initialization method: {self.initialization}")
        
    def _pca_initialization(self, data):
        """Initialize weights using PCA of the data"""
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(data)
        
        # Create grid based on PCA components
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Map grid position to PCA space
                pca_pos = np.array([
                    (i / (self.grid_size - 1)) * (pca_data[:, 0].max() - pca_data[:, 0].min()) + pca_data[:, 0].min(),
                    (j / (self.grid_size - 1)) * (pca_data[:, 1].max() - pca_data[:, 1].min()) + pca_data[:, 1].min()
                ])
                
                # Transform back to original space
                self.weights[i, j] = pca.inverse_transform(pca_pos)
        
    def _neighborhood(self, c, sigma):
        """
        Calculate neighborhood function for a given center and sigma
        
        Args:
            c (tuple): Center coordinates (x, y)
            sigma (float): Neighborhood radius
            
        Returns:
            numpy.ndarray: Neighborhood matrix
        """
        if sigma <= 0:
            # If sigma is too small, only update the winning neuron
            neighborhood = np.zeros((self.grid_size, self.grid_size))
            neighborhood[c[0], c[1]] = 1.0
            return neighborhood
            
        d = 2 * sigma * sigma
        ax = np.arange(self.grid_size)
        xx, yy = np.meshgrid(ax, ax)
        return np.exp(-((xx-c[0])**2 + (yy-c[1])**2) / d)

    def find_winner(self, x):
        """
        Find the winning neuron (Best Matching Unit) for input x
        
        Args:
            x (numpy.ndarray): Input vector
            
        Returns:
            tuple: Coordinates of the winning neuron
        """
        diff = self.weights - x
        dist = np.sum(diff**2, axis=-1)
        return np.unravel_index(np.argmin(dist), dist.shape)
    
    def calculate_quantization_error(self, data):
        """
        Calculate quantization error (average distance to winning neurons)
        
        Args:
            data (numpy.ndarray): Input data
            
        Returns:
            float: Quantization error
        """
        total_error = 0
        for x in data:
            winner = self.find_winner(x)
            error = np.sum((x - self.weights[winner])**2)
            total_error += error
        return total_error / len(data)
    
    def train(self, data, epochs=2000, convergence_threshold=1e-6, 
              early_stopping_patience=50, verbose=True):
        """
        Train the SOM with given data with enhanced convergence monitoring
        
        Args:
            data (numpy.ndarray): Training data
            epochs (int): Number of training epochs
            convergence_threshold (float): Threshold for convergence
            early_stopping_patience (int): Number of epochs to wait for improvement
            verbose (bool): Whether to print training progress
        """
        logger.info(f"Starting enhanced SOM training with {epochs} epochs and {len(data)} data points")
        
        # PCA initialization if requested
        if self.initialization == 'pca':
            self._pca_initialization(data)
        
        # Normalize data if not already done
        if data.shape[1] == self.input_len:
            data_normalized = self.scaler.fit_transform(data)
        else:
            data_normalized = data
            
        best_error = float('inf')
        patience_counter = 0
        convergence_history = []
        
        for epoch in range(epochs):
            # Decay parameters over time
            sigma = self.sigma * np.exp(-epoch / (epochs / 3))
            lr = self.learning_rate * np.exp(-epoch / epochs)
            
            # Shuffle data for better convergence
            if epoch % 100 == 0:
                np.random.shuffle(data_normalized)
            
            for x in data_normalized:
                winner = self.find_winner(x)
                g = self._neighborhood(winner, sigma)
                
                # Update weights
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        self.weights[i,j] += lr * g[i,j] * (x - self.weights[i,j])
            
            # Calculate quantization error every 100 epochs
            if epoch % 100 == 0 or epoch == epochs - 1:
                error = self.calculate_quantization_error(data_normalized)
                convergence_history.append(error)
                
                if verbose and epoch % 500 == 0:
                    logger.info(f"Epoch {epoch}: Quantization error = {error:.6f}")
                
                # Check for convergence
                if error < best_error:
                    best_error = error
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch} due to no improvement")
                    break
                
                # Convergence check
                if len(convergence_history) > 1:
                    if abs(convergence_history[-1] - convergence_history[-2]) < convergence_threshold:
                        logger.info(f"Converged at epoch {epoch}")
                        break
        
        self.training_history = convergence_history
        logger.info(f"SOM training completed. Final quantization error: {best_error:.6f}")

    def get_cluster(self, x):
        """
        Get cluster assignment for input x
        
        Args:
            x (numpy.ndarray): Input vector
            
        Returns:
            int: Cluster ID
        """
        # Normalize input if scaler was fitted
        if hasattr(self.scaler, 'mean_'):
            x_normalized = self.scaler.transform(x.reshape(1, -1)).flatten()
        else:
            x_normalized = x
            
        winner = self.find_winner(x_normalized)
        return winner[0] * self.grid_size + winner[1]
    
    def get_cluster_centers(self):
        """
        Get the cluster centers (weight vectors)
        
        Returns:
            numpy.ndarray: Cluster centers
        """
        centers = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                center = self.weights[i, j].copy()
                # Denormalize if scaler was used
                if hasattr(self.scaler, 'mean_'):
                    center = self.scaler.inverse_transform(center.reshape(1, -1)).flatten()
                centers.append(center)
        return np.array(centers)
    
    def get_cluster_quality_metrics(self, data):
        """
        Calculate quality metrics for the clustering
        
        Args:
            data (numpy.ndarray): Input data
            
        Returns:
            dict: Quality metrics
        """
        if hasattr(self.scaler, 'mean_'):
            data_normalized = self.scaler.transform(data)
        else:
            data_normalized = data
            
        # Calculate quantization error
        quantization_error = self.calculate_quantization_error(data_normalized)
        
        # Calculate topographic error (fraction of data points whose two best matching units are not adjacent)
        topographic_error = 0
        for x in data_normalized:
            # Find two best matching units
            diff = self.weights - x
            dist = np.sum(diff**2, axis=-1)
            flat_dist = dist.flatten()
            best_indices = np.argsort(flat_dist)[:2]
            
            # Convert to 2D coordinates
            best_coords = [np.unravel_index(idx, dist.shape) for idx in best_indices]
            
            # Check if they are adjacent
            if not self._are_adjacent(best_coords[0], best_coords[1]):
                topographic_error += 1
                
        topographic_error /= len(data)
        
        return {
            'quantization_error': quantization_error,
            'topographic_error': topographic_error,
            'silhouette_score': self._calculate_silhouette_score(data_normalized)
        }
    
    def _are_adjacent(self, coord1, coord2):
        """Check if two coordinates are adjacent in the grid"""
        return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1]) == 1
    
    def _calculate_silhouette_score(self, data):
        """Calculate silhouette score for clustering quality"""
        try:
            from sklearn.metrics import silhouette_score
            clusters = [self.get_cluster(x) for x in data]
            return silhouette_score(data, clusters)
        except ImportError:
            logger.warning("sklearn.metrics not available for silhouette score")
            return None 