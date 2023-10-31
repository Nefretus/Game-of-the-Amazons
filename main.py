from enum import Enum
import numpy as np

class BoardSize(Enum):
    SixBySix = 6
    EightByEight = 8
    TenByTen = 10

class Square(Enum):
    WHITE = 0
    BLACK = 1
    EMPTY = 2
    BLOCKED = 3


class Game:
    def __init__(self, board_size):
        self.board = Board(board_size)

    def run(self):
        pass

class Board:
    def generate_start_config(self, board_size):
        config = [[Square.EMPTY for _ in range(board_size.value)] for _ in range(board_size.value)]
        white_coords = []
        black_coords = []
        if board_size == BoardSize.SixBySix:
            white_coords = ['d1', 'c6']
            black_coords = ['a4', 'f3']
        elif board_size == BoardSize.EightByEight:
            white_coords = ['d8', 'e1']
            black_coords = ['a5', 'h4']
        elif board_size == BoardSize.TenByTen:
            white_coords = ['a4', 'd1', 'g1', 'j4']
            black_coords = ['a7', 'd10', 'g10', 'j7']
        for white, black in zip(white_coords, black_coords):
            i, j = self.notation_to_index(white)
            config[i][j] = Square.WHITE
            i, j = self.notation_to_index(black)
            config[i][j] = Square.BLACK
        return config

    # converts notation to index, takes as input eg. 'a1' and returns (0, 0)
    def notation_to_index(self, coord):
        return (int(''.join(filter(str.isdigit, coord))) - 1, ord(''.join(filter(str.isalpha, coord))) - ord('a'))

    def print_board(self):
        def name_to_symbol(square):
            symbol = ' '
            if square == Square.WHITE:
                symbol = 'w'
            elif square == Square.BLACK:
                symbol = 'b'
            elif square == square == Square.BLOCKED:
                symbol = 'O'
            else:
                symbol = '-'
            return symbol
        print("     Black")
        tmp = "  " + " ".join(map(lambda x: chr(x + ord('a')), range(len(self.config))))
        print(tmp)
        for r in range(len(self.config) - 1, -1, -1):
            print(r + 1, " ".join(map(name_to_symbol, self.config[r])), r + 1)
        print(tmp)
        print("     White")

    def __init__(self, board_size):
        self.white_to_play = True
        self.config = self.generate_start_config(board_size)
        self.print_board()

if __name__ == '__main__':
    Game(BoardSize.SixBySix).run()
    import pygame

    # Constants
    WIDTH, HEIGHT = 640, 480
    ROWS, COLS = 8, 8
    SQUARE_SIZE = WIDTH // COLS

    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)


    def draw_board(screen):
        for row in range(ROWS):
            for col in range(COLS):
                color = WHITE if (row + col) % 2 == 0 else BLACK
                pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))


    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chessboard")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)  # Fill the screen with a white background
        draw_board(screen)
        pygame.display.flip()

    pygame.quit()
