import numpy as np
from simulation import FastFlappySimulation
from visualizer import GameVisualizer
from constants import *

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