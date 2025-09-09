import numpy as np
import pygame
import random

# Color constants
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLUE_SKY = (135, 206, 235)

# Game constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BIRD_WIDTH = 30
BIRD_HEIGHT = 30
BIRD_X_POSITION = 100

# Physics constants
GRAVITY = 800
FLAP_STRENGTH = -300
BIRD_INITIAL_Y = SCREEN_HEIGHT // 2

# Pipe constants
PIPE_WIDTH = 60
PIPE_GAP_HEIGHT = 150
PIPE_SPEED = 200
PIPE_SPAWN_DISTANCE = 300
PIPE_MIN_GAP_Y = 100
PIPE_MAX_GAP_Y = SCREEN_HEIGHT - 250

# Neural network constants
NN_INPUT_SIZE = 4
NN_HIDDEN1_SIZE = 8
NN_HIDDEN2_SIZE = 16
NN_OUTPUT_SIZE = 1
NN_ARCHITECTURE = [NN_INPUT_SIZE, NN_HIDDEN1_SIZE, NN_HIDDEN2_SIZE, NN_OUTPUT_SIZE]

# Evolution constants
DEFAULT_POPULATION_SIZE = 2000
ELITE_PERCENTAGE = 0.2  # Top 20% as parents
ELITE_SURVIVORS = 10    # Best performers to keep unchangeable
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.1
CROSSOVER_THRESHOLD = 0.5

# Simulation constants
DEFAULT_FPS = 60
DEFAULT_GENERATION_TIME = 30
SURVIVAL_POINTS_PER_SECOND = 100
PIPE_PASS_BONUS = 1000

# Constants to normalize NN inputs
Y_POSITION_NORM = 600.0
VELOCITY_NORM = 400.0
DISTANCE_NORM = 800.0
GAP_RELATIVE_NORM = 300.0

# Visualization constants
GENERATION_DISPLAY_INTERVAL = 5

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
        
        # Reset pipes and randomizes first pipe TODO: This is still kind of bad, but I am focusing on more important matters lol
        initial_gap_y = random.randint(PIPE_MIN_GAP_Y, PIPE_MAX_GAP_Y)
        self.pipes = [[self.width, initial_gap_y, self.pipe_gap_height, False]]
        
        # Run simulation
        dt = 1/DEFAULT_FPS  # Assume constant 60 FPS to emulate visualizer mechanics
        while self.time < max_time:
            if not self.update(dt):  # All birds dead
                break
        
        return self.birds.copy()  # Return final states after all birds died
    
    def evolve(self):
        """Create next generation through evolution"""
        # Sort birds by fitness
        fitness_indices = np.argsort(self.birds[:, 2])[::-1]  # Descending order, most fit to the end
        
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

class GameVisualizer:
    """Pygame visualizer that can show the best bird from simulation"""
    def __init__(self):
        pygame.init()
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Flappy Bird Best AI playing")
        self.clock = pygame.time.Clock()
        
        # Single bird to visualize
        self.bird_y = BIRD_INITIAL_Y
        self.bird_vel = 0
        self.bird_x = BIRD_X_POSITION
        self.pipes = []
        self.pipe_speed = PIPE_SPEED
        self.pipe_width = PIPE_WIDTH

        # Best network from training
        self.best_network = None
    
    def set_best_network(self, network):
        self.best_network = network
    
    def run_visualization(self):
        """Run pygame visualization of the best bird"""
        if self.best_network is None:
            print("No network provided for visualization!")
            return
        
        running = True
        # Initialize with one pipe, not great. Maybe TODO: for change
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
                # Minimum distance considering only the pipes that have fully passed the bird.
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
                # x direction grows right to left, pipe approach from the left
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
                # Top pipe
                pygame.draw.rect(self.screen, GREEN, 
                               (x, 0, PIPE_WIDTH, gap_y))
                # Bottom pipe  
                pygame.draw.rect(self.screen, GREEN, 
                               (x, gap_y + gap_height, PIPE_WIDTH, self.height))
                
            bird_rect = pygame.Rect(BIRD_X_POSITION, self.bird_y, BIRD_WIDTH, BIRD_HEIGHT)

            # Check for collision with pipes (not good implementation though)
            for pipe in self.pipes:
                x, gap_y, gap_height, _ = pipe
                top_rect = pygame.Rect(x, 0, PIPE_WIDTH, gap_y)
                bottom_rect = pygame.Rect(x, gap_y + gap_height, PIPE_WIDTH, self.height)
                
                if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                    print("Collision detected in visualizer!")
                    running = False
                    break
            
            # Also check for ground/ceiling collision
            if self.bird_y <= 0 or self.bird_y + BIRD_HEIGHT >= self.height:
                print("Hit ground/ceiling in visualizer!")
                running = False
            
            pygame.display.flip()
        
        pygame.quit()

def main():
    print("Training AI started...")
    sim = FastFlappySimulation(population_size=DEFAULT_POPULATION_SIZE)
    
    best_fitness_history = []
    
    for generation in range(50):
        final_states = sim.run_generation(max_time=DEFAULT_GENERATION_TIME)
        
        # Track progress
        best_fitness = np.max(final_states[:, 2])
        avg_fitness = np.mean(final_states[:, 2])
        best_fitness_history.append(best_fitness)
        
        print(f"Gen {generation:3d}: Best={best_fitness:6.1f}, Avg={avg_fitness:6.1f}")
        
        # Show visualization every N generations
        if generation % GENERATION_DISPLAY_INTERVAL == 0:
            best_bird_idx = np.argmax(final_states[:, 2])
            best_network = sim.networks[best_bird_idx]
            
            print(f"Showing best bird from generation {generation}...")
            visualizer = GameVisualizer()
            visualizer.set_best_network(best_network)
            visualizer.run_visualization()
        
        sim.evolve()
    
    print("Training complete!")

if __name__ == "__main__":
    main()