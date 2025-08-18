import pygame


class Pipe:
    def __init__(self, x, gap_y, gap_height=150):
        self.x = x # Left edge of pipe
        self.gap_y = gap_y
        self.gap_height = gap_height
        self.width = 60
        self.passed = False
        self.right = self.x + self.width
        
    def update(self, dt, speed):
        self.x -= speed * dt
        
    def get_rects(self):
        screen_height = 600
        top_rect = pygame.Rect(self.x, 0, self.width, self.gap_y)
        bottom_rect = pygame.Rect(self.x, self.gap_y + self.gap_height, self.width, screen_height)
        return top_rect, bottom_rect
        
    def draw(self, screen):
        top_rect, bottom_rect = self.get_rects()
        pygame.draw.rect(screen, (0, 255, 0), top_rect)
        pygame.draw.rect(screen, (0, 255, 0), bottom_rect)
        
    def is_off_screen(self):
        return self.right < 0
        
    def check_collision(self, bird_rect):
        top_rect, bottom_rect = self.get_rects()
        # Uses Pygame built-in axis-aligned bounding box collision detection
        return bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect)
