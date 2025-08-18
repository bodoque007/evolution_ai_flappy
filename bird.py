import pygame


class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velocity_y = 0
        self.width = 30
        self.height = 30
        self.gravity = 800
        self.flap_strength = -300  # In pygame, y coordinate grows "downwards" thus the flap is negative, it pushes the bird upward
        
    def flap(self):
        self.velocity_y = self.flap_strength
        
    def update(self, dt):
        self.velocity_y += self.gravity * dt
        self.y += self.velocity_y * dt
        
    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)
        
    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 0), self.get_rect())
