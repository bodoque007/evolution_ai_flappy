import pygame
from tensorflow import keras
from tensorflow.keras import layers

class Bird:
    def __init__(self, x, y, brain=None):
        self.x = x
        self.y = y
        self.velocity_y = 0
        self.width = 30
        self.height = 30
        self.gravity = 800
        self.flap_strength = -300  # In pygame, y coordinate grows "downwards" thus the flap is negative, it pushes the bird upward
        self.brain = brain if brain is not None else create_brain()
        
    def flap(self):
        self.velocity_y = self.flap_strength
        
    def update(self, dt):
        self.velocity_y += self.gravity * dt # Gravity always acts. No matter what.
        self.y += self.velocity_y * dt
        
    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)
        
    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 0), self.get_rect())



def create_brain():
    model = keras.Sequential([
        layers.Input(shape=(4,)),          # 4 inputs
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid")  # output between 0 and 1
    ])
    return model