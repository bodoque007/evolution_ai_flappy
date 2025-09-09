import pygame
import numpy as np
import random
from constants import *

class GameVisualizer:
    """Pygame visualizer that can show the best bird from simulation"""
    def __init__(self):
        pygame.init()
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # Single bird to visualize
        self.bird_y = BIRD_INITIAL_Y
        self.bird_vel = 0
        self.bird_x = BIRD_X_POSITION
        self.pipes = []
        self.pipe_speed = PIPE_SPEED
        self.pipe_width = PIPE_WIDTH

        self.best_network = None
        pygame.display.set_caption("Flappy Bird Best AI playing")
    
    def set_best_network(self, network):
        self.best_network = network
    
    def run_visualization(self):
        """Run pygame visualization of the best bird"""
        if self.best_network is None:
            print("No network provided for visualization!")
            return
        
        running = True
        # Initialize with one pipe
        self.pipes = [[self.width, BIRD_INITIAL_Y, PIPE_GAP_HEIGHT, False]]
        
        while running:
            dt = self.clock.tick(DEFAULT_FPS) / 1000.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.bird_y = BIRD_INITIAL_Y
                        self.bird_vel = 0
                        self.pipes = [[self.width, BIRD_INITIAL_Y, PIPE_GAP_HEIGHT, False]]
            
            # AI decision
            if self.pipes:
                nearest_pipe = min((p for p in self.pipes if p[0] + self.pipe_width >= self.bird_x), 
                                 key=lambda p: p[0] - self.bird_x, default=None)
                if nearest_pipe:
                    inputs = np.array([
                        self.bird_y / Y_POSITION_NORM,
                        self.bird_vel / VELOCITY_NORM,
                        (nearest_pipe[0] - BIRD_X_POSITION) / DISTANCE_NORM,
                        (nearest_pipe[1] + PIPE_GAP_HEIGHT/2 - self.bird_y) / GAP_RELATIVE_NORM
                    ])
                    
                    if self.best_network.predict(inputs):
                        self.bird_vel = FLAP_STRENGTH
            
            self.bird_vel += GRAVITY * dt
            self.bird_y += self.bird_vel * dt
            
            for pipe in self.pipes:
                pipe[0] -= self.pipe_speed * dt
            
            self.pipes = [p for p in self.pipes if p[0] + PIPE_WIDTH > 0]
            
            # Spawn pipes
            if not self.pipes or self.pipes[-1][0] < self.width - PIPE_SPAWN_DISTANCE:
                gap_y = random.randint(PIPE_MIN_GAP_Y, PIPE_MAX_GAP_Y)
                self.pipes.append([self.width, gap_y, PIPE_GAP_HEIGHT, False])
            
            self.screen.fill(BLUE_SKY)
            
            pygame.draw.rect(self.screen, YELLOW, 
                           (BIRD_X_POSITION, self.bird_y, BIRD_WIDTH, BIRD_HEIGHT))
            
            for pipe in self.pipes:
                x, gap_y, gap_height, _ = pipe
                pygame.draw.rect(self.screen, GREEN, (x, 0, PIPE_WIDTH, gap_y))
                pygame.draw.rect(self.screen, GREEN, (x, gap_y + gap_height, PIPE_WIDTH, self.height))
                
            bird_rect = pygame.Rect(BIRD_X_POSITION, self.bird_y, BIRD_WIDTH, BIRD_HEIGHT)

            # Check for collision with pipes
            for pipe in self.pipes:
                x, gap_y, gap_height, _ = pipe
                top_rect = pygame.Rect(x, 0, PIPE_WIDTH, gap_y)
                bottom_rect = pygame.Rect(x, gap_y + gap_height, PIPE_WIDTH, self.height)
                
                if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                    running = False
            
            if self.bird_y <= 0 or self.bird_y + BIRD_HEIGHT >= self.height:
                running = False
            
            pygame.display.flip()
        
        pygame.quit()