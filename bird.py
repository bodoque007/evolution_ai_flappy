import pygame
import numpy as np
from keras import Sequential, layers
import random
import tensorflow as tf
import os

# Optimize TensorFlow for CPU performance
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable Intel oneDNN optimizations
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count()) 

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

    def decide(self, inputs):
        # inputs = [y, velocity_y, dist_to_next_pipe, gap_height_of_next_pipe]
        x = np.array(inputs).reshape((1,4))
        decision = self.brain(x).numpy()[0][0]
        if decision > 0.5:
            self.flap()

# TODO: Modularize the following functions.

def create_brain():
    model = Sequential([
        layers.Input(shape=(4,)),
        layers.Dense(8, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    return model


def crossover(parent1_brain: Sequential, parent2_brain: Sequential, mutation_rate=0.1):
    parent1_weights = parent1_brain.get_weights()
    parent2_weights = parent2_brain.get_weights()
    child_weights = []

    for w1, w2 in zip(parent1_weights, parent2_weights):
        mask = np.random.rand(*w1.shape) > 0.5
        child = np.where(mask, w1, w2)

        mutation_mask = np.random.rand(*child.shape) < mutation_rate
        noise = np.random.randn(*child.shape) * random.uniform(-0.5, 0.5)
        child += (mutation_mask * noise)
    
    child_brain = create_brain()
    child_brain.set_weights(child_weights)
    return child_brain