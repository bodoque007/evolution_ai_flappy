import numpy as np
from constants import *

class NeuralNetwork:
    def __init__(self, architecture=None, weights=None, biases=None):
        if architecture:
            self.weights = []
            self.biases = []
            for i in range(len(architecture) - 1):
                w = np.random.randn(architecture[i], architecture[i + 1])
                b = np.random.randn(architecture[i + 1])
                self.weights.append(w)
                self.biases.append(b)
        else:
            self.weights = weights
            self.biases = biases

    def predict(self, inputs):
        # Forward pass through all layers
        current_input = inputs
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = current_input @ w + b
            
            if i < len(self.weights) - 1:  # reLU
                current_input = np.maximum(0, z)
            else:  # sigmoid for output layer
                current_input = 1 / (1 + np.exp(-z))
        
        return current_input[0] > CROSSOVER_THRESHOLD

    @classmethod
    def crossover(cls, parent1, parent2, mutation_rate=MUTATION_RATE):
        """Create child network from two parents with flexible layer structure"""
        child_weights = []
        child_biases = []
        
        # Crossover for each layer
        for w1, w2, b1, b2 in zip(parent1.weights, parent2.weights, parent1.biases, parent2.biases):
            # Crossover weights
            w_mask = np.random.rand(*w1.shape) > CROSSOVER_THRESHOLD
            w_child = np.where(w_mask, w1, w2)
            
            # Crossover biases  
            b_mask = np.random.rand(*b1.shape) > CROSSOVER_THRESHOLD
            b_child = np.where(b_mask, b1, b2)
            
            # Mutate
            mutation_strength = MUTATION_STRENGTH
            w_child += (np.random.rand(*w_child.shape) < mutation_rate) * np.random.randn(*w_child.shape) * mutation_strength
            b_child += (np.random.rand(*b_child.shape) < mutation_rate) * np.random.randn(*b_child.shape) * mutation_strength
            
            child_weights.append(w_child)
            child_biases.append(b_child)
        
        return cls(weights=child_weights, biases=child_biases)