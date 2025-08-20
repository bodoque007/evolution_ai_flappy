import numpy as np
import pygame
import random
from typing import List, Tuple

GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLUE_SKY = (135, 206, 235)

class FastFlappySimulation:
    # BIG TODO: Change magic numbers
    def __init__(self, population_size=2000):
        self.population_size = population_size
        self.width = 800
        self.height = 600
        
        # Vectorized bird states: [y_pos, y_velocity, fitness, alive]
        self.birds = np.zeros((population_size, 4))
        self.birds[:, 0] = self.height // 2  # Initial y position, we don't care about x position as it's constant
        self.birds[:, 3] = 1  # All alive initially

        
        self.gravity = 800
        self.flap_strength = -300
        self.bird_width = 30
        self.bird_height = 30
        self.bird_x = 100

        self.pipes = []  # List of [x, gap_y, gap_height, passed]
        self.pipe_speed = 200
        self.pipe_width = 60
        self.pipe_gap_height = 150
        
        # Neural networks for each bird (simplified)
        self.networks = self.create_population_networks()
        
        self.time = 0
        self.generation = 1

    
    def create_population_networks(self):
        networks = []

        for _ in range(self.population_size):
            w1 = np.random.randn(4, 8)
            w2 = np.random.randn(8, 16)
            w3 = np.random.randn(16, 1)
            b1 = np.random.randn(8)
            b2 = np.random.randn(16)
            b3 = np.random.randn(1)
            networks.append((w1, b1, w2, b2, w3, b3))
        return networks

    def neural_network_decision(self, inputs, network):
        w1, b1, w2, b2, w3, b3 = network
        h1 = np.maximum(0, inputs @ w1 + b1)  # relu
        h2 = np.maximum(0, h1 @ w2 + b2) # relu

        output = 1 / (1 + np.exp(-(h2 @ w3 + b3)))  # Sigmoid
        return output[0] > 0.5

    def get_inputs_for_bird(self, bird_idx):
        """Get inputs for a specific bird and normalizes them"""
        if self.birds[bird_idx, 3] == 0:  # Bird is dead
            return np.zeros(4)
        
        y_pos = self.birds[bird_idx, 0]
        y_vel = self.birds[bird_idx, 1]
        
        # Find nearest pipe
        nearest_pipe = None
        min_distance = float('inf')
        
        for pipe in self.pipes:
            pipe_x, gap_y, gap_height, passed = pipe
            if pipe_x + self.pipe_width >= self.bird_x:
                distance = pipe_x - self.bird_x
                if distance < min_distance:
                    min_distance = distance
                    nearest_pipe = pipe
        
        pipe_x, gap_y, gap_height, passed = nearest_pipe # Assume there will always be a next pipe
        
        return np.array([
            y_pos / 600.0,  # Normalized y position
            y_vel / 400.0,  # Normalized velocity  
            (pipe_x - self.bird_x) / 800.0,  # Distance to pipe
            (gap_y + gap_height/2 - y_pos) / 300.0  # Gap center relative to bird
        ])
    


class GameVisualizer:
    """Pygame visualizer that can show the best bird from simulation"""
    
    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Flappy Bird Best AI playing")
        self.clock = pygame.time.Clock()
        
        # Single bird to visualize
        self.bird_y = self.height // 2
        self.bird_vel = 0
        self.bird_x = 100
        self.pipes = []
        self.pipe_speed = 200
        self.pipe_width = 60
        
        # Best network from training
        self.best_network = None
    
    def set_best_network(self, network):
        self.best_network = network
    
    def neural_network_decision(self, inputs, network):
        w1, b1, w2, b2, w3, b3 = network
        h1 = np.maximum(0, inputs @ w1 + b1)
        h2 = np.maximum(0, h1 @ w2 + b2)
        output = 1 / (1 + np.exp(-(h2 @ w3 + b3)))
        return output[0] > 0.5
    
    def run_visualization(self):
        """Run pygame visualization of the best bird"""
        if self.best_network is None:
            print("No network provided for visualization!")
            return
        
        running = True
        # Initialize with one pipe, not great. Maybe TODO: for change
        self.pipes = [[self.width, self.height//2, 150, False]]
        
        while running:
            dt = self.clock.tick(60) / 1000.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.bird_y = self.height // 2
                        self.bird_vel = 0
                        self.pipes = [[self.width, self.height//2, 150, False]]
            
            # AI decision
            if self.pipes:
                # Minimum distance considering only the pipes that have fully passed the bird.
                nearest_pipe = min((p for p in self.pipes if p[0] + self.pipe_width >= self.bird_x), 
                                 key=lambda p: p[0] - 100, default=None)
                if nearest_pipe:
                    inputs = np.array([
                        self.bird_y / 600.0,
                        self.bird_vel / 400.0,
                        (nearest_pipe[0] - 100) / 800.0,
                        (nearest_pipe[1] + 75 - self.bird_y) / 300.0
                    ])
                    
                    if self.neural_network_decision(inputs, self.best_network):
                        self.bird_vel = -300
            
            self.bird_vel += 800 * dt
            self.bird_y += self.bird_vel * dt
            
            for pipe in self.pipes:
                pipe[0] -= self.pipe_speed * dt
            
            self.pipes = [p for p in self.pipes if p[0] + 60 > 0]
            
            # Spawn pipes
            if not self.pipes or self.pipes[-1][0] < self.width - 300:
                gap_y = random.randint(100, self.height - 250)
                self.pipes.append([self.width, gap_y, 150, False])
            
            self.screen.fill(BLUE_SKY)
            
            pygame.draw.rect(self.screen, YELLOW, 
                           (100, self.bird_y, 30, 30))
            
            for pipe in self.pipes:
                x, gap_y, gap_height, _ = pipe
                # Top pipe
                pygame.draw.rect(self.screen, GREEN, 
                               (x, 0, 60, gap_y))
                # Bottom pipe  
                pygame.draw.rect(self.screen, GREEN, 
                               (x, gap_y + gap_height, 60, self.height))
                            
            pygame.display.flip()
        
        pygame.quit()


