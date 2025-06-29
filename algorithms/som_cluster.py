import numpy as np
import numpy.random as rnd
import logging

logger = logging.getLogger(__name__)

class SOMCluster:
    """
    Self-Organizing Map (SOM) clustering algorithm for staff location clustering
    """
    
    def __init__(self, input_len, grid_size=3, sigma=1.0, learning_rate=0.5):
        """
        Initialize SOM cluster with parameters
        
        Args:
            input_len (int): Length of input vectors
            grid_size (int): Size of the SOM grid (grid_size x grid_size)
            sigma (float): Initial neighborhood radius
            learning_rate (float): Initial learning rate
        """
        logger.info(f"Initializing SOMCluster with grid_size={grid_size}, sigma={sigma}, learning_rate={learning_rate}")
        self.grid_size = grid_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.input_len = input_len
        self._init_weights()
        
    def _init_weights(self):
        """Initialize SOM weights randomly"""
        logger.debug("Initializing SOM weights")
        self.weights = rnd.rand(self.grid_size, self.grid_size, self.input_len)
        
    def _neighborhood(self, c, sigma):
        """
        Calculate neighborhood function for a given center and sigma
        
        Args:
            c (tuple): Center coordinates (x, y)
            sigma (float): Neighborhood radius
            
        Returns:
            numpy.ndarray: Neighborhood matrix
        """
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
    
    def train(self, data, epochs=2000):
        """
        Train the SOM with given data
        
        Args:
            data (numpy.ndarray): Training data
            epochs (int): Number of training epochs
        """
        logger.info(f"Starting SOM training with {epochs} epochs and {len(data)} data points")
        
        for epoch in range(epochs):
            # Decay parameters over time
            sigma = self.sigma * (1 - epoch/epochs)
            lr = self.learning_rate * (1 - epoch/epochs)
            
            for x in data:
                winner = self.find_winner(x)
                g = self._neighborhood(winner, sigma)
                
                # Update weights
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        self.weights[i,j] += lr * g[i,j] * (x - self.weights[i,j])
        
        logger.info("SOM training completed")

    def get_cluster(self, x):
        """
        Get cluster assignment for input x
        
        Args:
            x (numpy.ndarray): Input vector
            
        Returns:
            int: Cluster ID
        """
        winner = self.find_winner(x)
        return winner[0] * self.grid_size + winner[1] 