import pygame
import random
import sys
from bird import Bird, crossover
from pipe import Pipe
import numpy as np


class Game:
    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Flappy Bird ML")
        self.clock = pygame.time.Clock()
        self.ai_frame_counter = 0
        self.ai_decision_interval = 5
        self.birds = set(Bird(100, self.height // 2) for _ in range(70))
        self.dead_birds = []
        self.pipes = []
        
        self.pipe_speed = 200  # pixels/second
        self.pipe_spawn_timer = 0
        self.pipe_spawn_interval = 2.0  # seconds

        self.speed_increase_timer = 0
        # The two following lines mean that every 3 seconds, pipe speed increases by *1.1.
        self.speed_increase_interval = 3.0  # seconds
        self.speed_increase_factor = 1.1 #
        
        self.score = 0
        self.game_over = False
        
    def spawn_pipe(self):
        gap_y = random.randint(100, self.height - 250)
        pipe = Pipe(self.width, gap_y)
        self.pipes.append(pipe)
    
    def get_next_pipe(self, x_pos, pipes):
        next_pipe = None
        min_distance = float('inf')
        
        for pipe in pipes:
            if pipe.right > x_pos:
                distance = pipe.x - x_pos
                if distance < min_distance:
                    min_distance = distance
                    next_pipe = pipe
        
        return next_pipe
        
    def update(self, dt):
        if self.game_over:
            return
        
        birds_to_remove = set()
        pipes_to_remove = set()

        nearest_pipe = self.get_next_pipe(next(iter(self.birds)).x, self.pipes)
            

        for bird in self.birds:
            if self.ai_frame_counter % self.ai_decision_interval == 0 and nearest_pipe:
                inputs = [bird.y, bird.velocity_y, nearest_pipe.x - bird.x, (nearest_pipe.gap_y - bird.y)]
                bird.decide(inputs)
            
            bird.update(dt) 

            # If bird hits either ground or ceiling, it's done for.
            if bird.y <= 0 or bird.y + bird.height >= self.height or nearest_pipe.check_collision(bird.get_rect()):
                birds_to_remove.add(bird)
                self.dead_birds.append(bird)
        
        self.ai_frame_counter += 1
            
            
        for pipe in self.pipes:
            pipe.update(dt, self.pipe_speed)

                
            if pipe.is_off_screen():
                pipes_to_remove.add(pipe)

        
        if birds_to_remove:
            self.birds = set(bird for bird in self.birds if bird not in birds_to_remove)
    
        if pipes_to_remove:
            self.pipes = [pipe for pipe in self.pipes if pipe not in pipes_to_remove]
            
        # End game only if all birds are dead
        if len(self.birds) == 0:
            print(f"dead birds are {len(self.dead_birds)}")
            self.game_over = True
                
        self.pipe_spawn_timer += dt
        if self.pipe_spawn_timer >= self.pipe_spawn_interval:
            self.spawn_pipe()
            self.pipe_spawn_timer = 0
        
        # Increase speed periodically after speed_increase_timer seconds
        self.speed_increase_timer += dt
        if self.speed_increase_timer >= self.speed_increase_interval:
            self.pipe_speed *= self.speed_increase_factor
            self.speed_increase_timer = 0
            
    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w and self.game_over:
                    self.reset()
        return True
        
    def reset(self):
        self.birds = self.make_mutations(self.dead_birds[-5:], 70)
        self.ai_frame_counter = 0
        self.dead_birds = []
        self.pipes = []
        self.pipes.append(Pipe(self.width - 50, self.height // 2))
        self.pipe_speed = 200
        self.pipe_spawn_timer = 0
        self.speed_increase_timer = 0
        self.score = 0
        self.game_over = False
        
    def draw(self):
        self.screen.fill((135, 206, 235))  # Sky blue
        
        for bird in self.birds:
            bird.draw(self.screen)
        
        for pipe in self.pipes:
            pipe.draw(self.screen)
                    
        if self.game_over:
            font = pygame.font.Font(None, 36)
            game_over_text = font.render("GAME OVER", True, (255, 0, 0))
            text_rect = game_over_text.get_rect(center=(self.width//2, self.height//2))
            self.screen.blit(game_over_text, text_rect)
            
        pygame.display.flip()
        
    def run(self):
        running = True
        # TODO: Maybe fix, this is just so the bird already has a next_pipe during the first frame. Not so elegant at all.
        self.pipes.append(Pipe(self.width - 50, self.height // 2))
        while running:
            dt = self.clock.tick(30) / 1000.0
            
            running = self.handle_input()
            self.update(dt)
            self.draw()
            
        pygame.quit()
        sys.exit()


    def make_mutations(self, parents, population_size):
        new_population = set()
        
        # Create weights that favor later parents (higher scores)
        # Last parent gets highest weight, first gets lowest
        weights = np.arange(1, len(parents) + 1)  # [1, 2, 3, 4, 5] for 5 parents
        weights = weights / np.sum(weights)  # Normalize to probabilities
        
        while len(new_population) < population_size:
            p1, p2 = np.random.choice(parents, 2, replace=True, p=weights)
            child = crossover(p1.brain, p2.brain)
            new_population.add(Bird(100, self.height // 2, brain=child))
        
        return new_population
