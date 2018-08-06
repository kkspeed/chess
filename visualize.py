import pygame
import sys
from pygame.locals import *

WHITE = (255, 255, 255)
BOARD_COLOR = (0, 0, 0)

def draw_chess_board(surface):
    pygame.draw.line(surface, BOARD_COLOR, (100, 100), (200, 200), 4)


if __name__ == "__main__":
    pygame.init()
    surface = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Visualize")
    surface.fill(WHITE);
    while True:
        surface.fill(WHITE)
        draw_chess_board(surface)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.update()