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
                should_flap = self.neural_network_decision(inputs, self.networks[i])
                if should_flap:
                    self.birds[i, 1] = self.flap_strength
        
        # Physics update (vectorized)
        self.birds[alive_mask, 1] += self.gravity * dt # Gravity
        self.birds[alive_mask, 0] += self.birds[alive_mask, 1] * dt # Position
        
        # Update fitness (time survived + distance)
        self.birds[alive_mask, 2] += dt * 100  # Base survival points
        
        # Collision detection
        for i in range(self.population_size):
            if not alive_mask[i]:
                continue
                
            bird_y = self.birds[i, 0]
            
            
            if bird_y <= 0 or bird_y + self.bird_height >= self.height:
                self.birds[i, 3] = 0  # Kill bird
                continue
            
            # Pipe collision
            bird_x = 100
            bird_rect = [bird_x, bird_y, self.bird_width, self.bird_height]
            
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
                        self.birds[i, 2] += 1000  # Bonus for passing pipe
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
        if len(self.pipes) == 0 or self.pipes[-1][0] < self.width - 300:
            gap_y = random.randint(100, self.height - 250)
            self.pipes.append([self.width, gap_y, self.pipe_gap_height, False])
        
        self.time += dt
        return True
    
    def run_generation(self, max_time=30):
        """Run one full generation"""
        self.time = 0
        
        # Reset birds
        self.birds[:, 0] = self.height // 2  # Reset positions
        self.birds[:, 1] = 0  # Reset velocities
        self.birds[:, 2] = 0  # Reset fitness
        self.birds[:, 3] = 1  # All alive
        
        # Reset pipes
        self.pipes = [[self.width, self.height//2, self.pipe_gap_height, False]]
        
        # Run simulation
        dt = 1/60  # Assume constant 60 FPS to emulate visualizer mechanics
        while self.time < max_time:
            if not self.update(dt):  # All birds dead
                break
        
        return self.birds.copy()  # Return final states after all birds died
    
    def evolve(self):
        """Create next generation through evolution"""
        # Sort birds by fitness
        fitness_indices = np.argsort(self.birds[:, 2])[::-1]  # Descending order
        
        # Select top 20% as parents
        elite_size = max(1, self.population_size // 5)
        elite_indices = fitness_indices[:elite_size]
        
        # Create new networks
        new_networks = []
        
        # Keep best performers
        for i in range(min(10, elite_size)):
            new_networks.append(self.networks[elite_indices[i]])
        
        # Create rest through crossover and mutation
        while len(new_networks) < self.population_size:
            p1_idx = np.random.choice(elite_indices)
            p2_idx = np.random.choice(elite_indices)
            
            child_network = self.crossover_networks(
                self.networks[p1_idx], 
                self.networks[p2_idx]
            )
            new_networks.append(child_network)
        
        self.networks = new_networks
        self.generation += 1
    
    def crossover_networks(self, parent1, parent2, mutation_rate=0.1):
        """Create child network from two parents"""
        """This method is terribly and of course the whole manual building of NNs does NOT escalate at all"""
        w1_p1, b1_p1, w2_p1, b2_p1, w3_p1, b3_p1 = parent1
        w1_p2, b1_p2, w2_p2, b2_p2, w3_p2, b3_p2 = parent2
        
        # Crossover
        mask1 = np.random.rand(*w1_p1.shape) > 0.5
        mask2 = np.random.rand(*w2_p1.shape) > 0.5
        mask3 = np.random.rand(*w3_p1.shape) > 0.5
        mask_b1 = np.random.rand(*b1_p1.shape) > 0.5
        mask_b2 = np.random.rand(*b2_p1.shape) > 0.5
        mask_b3 = np.random.rand(*b3_p1.shape) > 0.5
        
        
        w1_child = np.where(mask1, w1_p1, w1_p2)
        w2_child = np.where(mask2, w2_p1, w2_p2)
        w3_child = np.where(mask3, w3_p1, w3_p2)
        b1_child = np.where(mask_b1, b1_p1, b1_p2)
        b2_child = np.where(mask_b2, b2_p1, b2_p2)
        b3_child = np.where(mask_b3, b3_p1, b3_p2)
        
        # Mutation
        mutation_strength = 0.1
        w1_child += (np.random.rand(*w1_child.shape) < mutation_rate) * np.random.randn(*w1_child.shape) * mutation_strength
        w2_child += (np.random.rand(*w2_child.shape) < mutation_rate) * np.random.randn(*w2_child.shape) * mutation_strength
        w3_child += (np.random.rand(*w3_child.shape) < mutation_rate) * np.random.randn(*w3_child.shape) * mutation_strength
        b1_child += (np.random.rand(*b1_child.shape) < mutation_rate) * np.random.randn(*b1_child.shape) * mutation_strength
        b2_child += (np.random.rand(*b2_child.shape) < mutation_rate) * np.random.randn(*b2_child.shape) * mutation_strength
        b3_child += (np.random.rand(*b3_child.shape) < mutation_rate) * np.random.randn(*b3_child.shape) * mutation_strength
        
        return (w1_child, b1_child, w2_child, b2_child, w3_child, b3_child)


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
                # x direction grows right to left, pipe approach from the left
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
                
            
            bird_rect = pygame.Rect(100, self.bird_y, 30, 30)

            # Check for collision with pipes (not good implementation though)
            for pipe in self.pipes:
                x, gap_y, gap_height, _ = pipe
                top_rect = pygame.Rect(x, 0, 60, gap_y)
                bottom_rect = pygame.Rect(x, gap_y + gap_height, 60, self.height)
                
                if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                    print("Collision detected in visualizer!")
                    running = False
                    break

            # Also check for ground/ceiling collision
            if self.bird_y <= 0 or self.bird_y + 30 >= self.height:
                print("Hit ground/ceiling in visualizer!")
                running = False
            
            pygame.display.flip()
        
        pygame.quit()


def main():
    print("Training AI started...")
    sim = FastFlappySimulation(population_size=2000)
    
    best_fitness_history = []
    
    for generation in range(50):
        # Run fast simulation
        final_states = sim.run_generation(max_time=30)
        
        # Track progress
        best_fitness = np.max(final_states[:, 2])
        avg_fitness = np.mean(final_states[:, 2])
        best_fitness_history.append(best_fitness)
        
        print(f"Gen {generation:3d}: Best={best_fitness:6.1f}, Avg={avg_fitness:6.1f}")
        
        # Show visualization every 5 generations
        if generation % 5 == 0:
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