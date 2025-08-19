import numpy as np
import pygame
import random
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class FastFlappySimulation:
    
    def __init__(self, population_size=2000):
        self.population_size = population_size
        self.width = 800
        self.height = 600
        
        # Bird states: [y_pos, y_velocity, fitness, alive]
        self.birds = np.zeros((population_size, 4))
        self.birds[:, 0] = self.height // 2  # Initial y position, we don't care about x position as it's constant
        self.birds[:, 3] = 1  # All alive initially
        
        self.gravity = 800
        self.flap_strength = -300
        self.bird_width = 30
        self.bird_height = 30
        
        self.pipes = []  # List of [x, gap_y, gap_height, passed]
        self.pipe_speed = 200
        self.pipe_width = 60
        self.pipe_gap_height = 150
        
        # Neural networks for each bird (simplified)
        self.networks = self.create_population_networks()
        
        self.time = 0
        self.generation = 1
        