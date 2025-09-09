import numpy as np
import random
from constants import *
from neural_network import NeuralNetwork

class FastFlappySimulation:
    def __init__(self, population_size=DEFAULT_POPULATION_SIZE):
        self.population_size = population_size
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        
        # Vectorized bird states: [y_pos, y_velocity, fitness, alive]
        self.birds = np.zeros((population_size, 4))
        self.birds[:, 0] = BIRD_INITIAL_Y  # Initial y position, we don't care about x position as it's constant
        self.birds[:, 3] = 1  # All alive initially

        self.gravity = GRAVITY
        self.flap_strength = FLAP_STRENGTH
        self.bird_width = BIRD_WIDTH
        self.bird_height = BIRD_HEIGHT
        self.bird_x = BIRD_X_POSITION

        self.pipes = []  # List of [x, gap_y, gap_height, passed]
        self.pipe_speed = PIPE_SPEED
        self.pipe_width = PIPE_WIDTH
        self.pipe_gap_height = PIPE_GAP_HEIGHT
        
        # Neural networks for each bird
        self.networks = self.create_population_networks()
        
        self.time = 0
        self.generation = 1

    def create_population_networks(self):
        return [NeuralNetwork(architecture=NN_ARCHITECTURE) for _ in range(self.population_size)]

    def get_inputs_for_bird(self, bird_idx):
        """Get inputs for a specific bird and normalizes them"""
        if self.birds[bird_idx, 3] == 0:  # Bird is dead
            return np.zeros(NN_INPUT_SIZE)
        
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
        
        if nearest_pipe is None:
            # No pipes available, return default inputs
            return np.array([
                y_pos / Y_POSITION_NORM,
                y_vel / VELOCITY_NORM,
                1.0,  # Max distance
                0.0   # No gap to consider
            ])
        
        pipe_x, gap_y, gap_height, passed = nearest_pipe
        
        return np.array([
            y_pos / Y_POSITION_NORM,  # Normalized y position
            y_vel / VELOCITY_NORM,  # Normalized velocity  
            (pipe_x - self.bird_x) / DISTANCE_NORM,  # Distance to pipe
            (gap_y + gap_height/2 - y_pos) / GAP_RELATIVE_NORM  # Gap center relative to bird, normalized too
        ])
    
    def update(self, dt):
        # bird = [y_pos, y_velocity, fitness, alive]
        alive_mask = self.birds[:, 3] == 1
        alive_count = np.sum(alive_mask)
        
        if alive_count == 0:
            return False # No need to update
        
        # AI decisions for all alive birds
        for i in range(self.population_size):
            if alive_mask[i]:
                inputs = self.get_inputs_for_bird(i)
                should_flap = self.networks[i].predict(inputs)
                if should_flap:
                    self.birds[i, 1] = self.flap_strength
        
        # Physics update (vectorized)
        self.birds[alive_mask, 1] += self.gravity * dt # Gravity
        self.birds[alive_mask, 0] += self.birds[alive_mask, 1] * dt # Position
        
        # Update fitness (time survived + distance)
        self.birds[alive_mask, 2] += dt * SURVIVAL_POINTS_PER_SECOND  # Base survival points
        
        # Collision detection
        for i in range(self.population_size):
            if not alive_mask[i]:
                continue
                
            bird_y = self.birds[i, 0]
            
            if bird_y <= 0 or bird_y + self.bird_height >= self.height:
                self.birds[i, 3] = 0  # Kill bird
                continue
            
            # Pipe collision
            bird_x = BIRD_X_POSITION
            
            for pipe in self.pipes:
                pipe_x, gap_y, gap_height, passed = pipe
                if (pipe_x <= bird_x + self.bird_width and 
                    pipe_x + self.pipe_width >= bird_x):
                    
                    # Check collision with top and bottom pipe
                    if (bird_y < gap_y or 
                        bird_y + self.bird_height > gap_y + gap_height):
                        self.birds[i, 3] = 0  # Kill bird
                        break
                    
                    # Bird passed pipe
                    elif not passed and bird_x > pipe_x + self.pipe_width:
                        self.birds[i, 2] += PIPE_PASS_BONUS  # Bonus for passing pipe
                        pipe[3] = True  # Mark pipe as passed
        
        # Update pipes
        pipes_to_remove = []
        for i, pipe in enumerate(self.pipes):
            pipe[0] -= self.pipe_speed * dt  # Move pipe left
            if pipe[0] + self.pipe_width < 0:  # Off screen
                pipes_to_remove.append(i)
        
        for i in reversed(pipes_to_remove):
            self.pipes.pop(i)
        
        # Spawn new pipes
        if len(self.pipes) == 0 or self.pipes[-1][0] < self.width - PIPE_SPAWN_DISTANCE:
            gap_y = random.randint(PIPE_MIN_GAP_Y, PIPE_MAX_GAP_Y)
            self.pipes.append([self.width, gap_y, self.pipe_gap_height, False])
        
        self.time += dt
        return True
    
    def run_generation(self, max_time=DEFAULT_GENERATION_TIME):
        """Run one full generation"""
        self.time = 0
        
        # Reset birds
        self.birds[:, 0] = BIRD_INITIAL_Y  # Reset positions
        self.birds[:, 1] = 0  # Reset velocities
        self.birds[:, 2] = 0  # Reset fitness
        self.birds[:, 3] = 1  # All alive
        
        # Reset pipes and randomizes first pipe
        initial_gap_y = random.randint(PIPE_MIN_GAP_Y, PIPE_MAX_GAP_Y)
        self.pipes = [[self.width, initial_gap_y, self.pipe_gap_height, False]]
        
        # Run simulation
        dt = 1/DEFAULT_FPS
        while self.time < max_time:
            if not self.update(dt):  # All birds dead
                break
        
        return self.birds.copy()
    
    def evolve(self):
        """Create next generation through evolution"""
        # Sort birds by fitness
        fitness_indices = np.argsort(self.birds[:, 2])[::-1]
        
        # Select top percentage as parents
        elite_size = max(1, int(self.population_size * ELITE_PERCENTAGE))
        elite_indices = fitness_indices[:elite_size]
        
        # Create new networks
        new_networks = []
        
        # Keep best performers as they are
        for i in range(min(ELITE_SURVIVORS, elite_size)):
            new_networks.append(self.networks[elite_indices[i]])
        
        # Create rest through crossover and mutation
        while len(new_networks) < self.population_size:
            p1_idx = np.random.choice(elite_indices)
            p2_idx = np.random.choice(elite_indices)
            
            child_network = NeuralNetwork.crossover(
                self.networks[p1_idx], 
                self.networks[p2_idx]
            )
            new_networks.append(child_network)
        
        self.networks = new_networks
        self.generation += 1