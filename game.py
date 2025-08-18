import pygame
import random
import sys
from bird import Bird
from pipe import Pipe


class Game:
    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Flappy Bird ML")
        self.clock = pygame.time.Clock()
        
        self.bird = Bird(100, self.height // 2)
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
        
    def update(self, dt):
        if self.game_over:
            return
            
        self.bird.update(dt)
        
        # If bird hits either ground or ceiling, it's done for.
        if self.bird.y <= 0 or self.bird.y + self.bird.height >= self.height:
            self.game_over = True
            
        for pipe in self.pipes[:]:  # Copy list to avoid modification during iteration
            pipe.update(dt, self.pipe_speed)
            
            if pipe.check_collision(self.bird.get_rect()):
                self.game_over = True
                
            # Check if pipe passed (for scoring)
            if not pipe.passed and pipe.right < self.bird.x:
                pipe.passed = True
                self.score += 1
                
            if pipe.is_off_screen():
                self.pipes.remove(pipe)
                
        self.pipe_spawn_timer += dt
        if self.pipe_spawn_timer >= self.pipe_spawn_interval:
            self.spawn_pipe()
            self.pipe_spawn_timer = 0
        
        # Increases speed periodically after speed_increase_timer ms
        self.speed_increase_timer += dt
        if self.speed_increase_timer >= self.speed_increase_interval:
            self.pipe_speed *= self.speed_increase_factor
            self.speed_increase_timer = 0
            print(f"Speed increased to: {self.pipe_speed:.1f}")
            
    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE or event.key == pygame.K_w:
                    if not self.game_over:
                        self.bird.flap()
                    else:
                        self.reset()
        return True
        
    def reset(self):
        self.bird = Bird(100, self.height // 2)
        self.pipes = []
        self.pipe_speed = 200
        self.pipe_spawn_timer = 0
        self.speed_increase_timer = 0
        self.score = 0
        self.game_over = False
        
    def draw(self):
        self.screen.fill((135, 206, 235))  # Sky blue
        
        self.bird.draw(self.screen)
        
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
        while running:
            dt = self.clock.tick(60) / 1000.0
            
            running = self.handle_input()
            self.update(dt)
            self.draw()
            
        pygame.quit()
        sys.exit()
