import numpy as np
import pygame
import sys
import h5py
import textwrap
from pygame.locals import *
from chess_types import Board, Player
from encoder import SimpleEncoder
from agent import ExpCollector

WHITE = (255, 255, 255)
BOARD_COLOR = (0, 0, 0)

START = (40, 40)

TEXT_FONT = "KaiTi" if sys.platform == "win32" else "WenQuanYi Micro Hei"

def draw_chess_board(surface):
    for i in range(10):
        pygame.draw.line(surface, BOARD_COLOR, (START[1], START[0] + i * 50), (440, START[0] + i * 50), 4)
    for i in range(9):
        pygame.draw.line(surface, BOARD_COLOR, (START[1] + i * 50, START[0]), (START[1] + i * 50, 490), 4)
    pygame.draw.rect(surface, WHITE, (40, 240, 400, 50))
    pygame.draw.line(surface, BOARD_COLOR, (190, 40), (290, 140), 4)
    pygame.draw.line(surface, BOARD_COLOR, (190, 140), (290, 40), 4)
    pygame.draw.line(surface, BOARD_COLOR, (190, 390), (290, 490), 4)
    pygame.draw.line(surface, BOARD_COLOR, (190, 490), (290, 390), 4)

def draw_pieces(surface, font, board: Board):
    for piece in board.pieces:
        y = piece.pos.row * 50 + 40
        x = piece.pos.col * 50 + 40
        color = (255, 0, 0) if piece.color == Player.red else (0, 255, 0)
        text = font.render(str(piece), True, color, (255, 255, 255))
        textrect = text.get_rect()
        textrect.centerx = x
        textrect.centery = y
        surface.blit(text, textrect)
    
idx = 0
 
if __name__ == "__main__":
    pygame.init()

    board = Board()
    board.parse_from_string(textwrap.dedent("""\
        車馬象仕将仕象馬車
        .........
        .包.....包.
        卒.卒.卒.卒.卒
        .........
        .........
        兵.兵.兵.兵.兵
        .炮.....炮.
        .........
        车马相士帅士相马车"""))

    font = pygame.font.SysFont(TEXT_FONT, 40)
    surface = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Visualize")
    surface.fill(WHITE);
    collector = ExpCollector()
    encoder = SimpleEncoder()
    h5 = h5py.File(sys.argv[1], "r")
    collector.load(h5)
    while True:
        surface.fill(WHITE)
        board = encoder.decode(collector.inputs[idx])
        draw_chess_board(surface)
        draw_pieces(surface, font, board)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYUP and event.key == K_RIGHT:
                idx += 1
                idx %= len(collector.inputs)
            if event.type == KEYUP and event.key == K_LEFT:
                idx += len(collector.inputs) - 1
                idx %= len(collector.inputs)

        pygame.display.update()
