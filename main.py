from enum import Enum
import numpy as np
import os


class BoardSize(Enum):
    SixBySix = 6
    EightByEight = 8
    TenByTen = 10


class Square(Enum):
    WHITE = 0  # there is white amazon on that square
    BLACK = 1  # there is black amazon on that square
    EMPTY = 2  # square is empty
    BLOCKED = 3  # square blocked


class Territory(Enum):
    WHITE = 0      # this territory belongs to black
    BLACK = 1      # this territory belongs to white
    NEUTRAL = 2    # shared territory
    DEADSPACE = 3  # this territory is closed and cannot be entered by any amazon
    OCCUPIED = 4   # this square is blocked or has amazon on it
    ANALYZED = 5   # this state is only active while searching for connected areas
    UNKNOWN = 6    # square not yet analyzed


class Game:
    def __init__(self, board_size):
        self.board = Board(board_size)

    def run(self):
        # human vs human
        pass


class Board:
    def __init__(self, board_size):
        self.white_to_play = True
        self.config = self.generate_start_config(board_size)
        self.print_board()

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

    # converts notation to index, takes as input eg. 'a1' and returns (0, 0), row are number and columns are letters
    def notation_to_index(self, coord):
        return (int(''.join(filter(str.isdigit, coord))) - 1), (ord(''.join(filter(str.isalpha, coord))) - ord('a'))

    def index_to_notation(self, index):
        return chr(ord('a') + index[1]) + str(index[0] + 1)

    def print_board(self):
        def name_to_symbol(square):
            symbol = ' '
            if square == Square.WHITE:
                symbol = 'w'
            elif square == Square.BLACK:
                symbol = 'b'
            elif square == Square.BLOCKED:
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

    # takes in move as two tuples containing indices in config list
    def is_valid_move(self, source, destination):
        source_row, source_col = source
        dest_row, dest_col = destination

        # source square has to be the one containing either black or white amazon
        if (self.white_to_play and self.config[source_row][source_col] != Square.WHITE) or (
                not self.white_to_play and self.config[source_row][source_col] != Square.BLACK):
            print('Invalid move, there is no piece on this square')
            return False

        # move can only be straight line
        delta_row = dest_row - source_row
        delta_col = dest_col - source_col
        max_abs_delta = max(abs(delta_row), abs(delta_col))

        # tanges 45 = 1 = x / y, zeby diagolnie lezaly na tej samej prostej kat musi być miedzy nimi 45 stopni
        if delta_row != 0 and delta_col != 0 and abs(delta_row / delta_col) != 1:
            print('Invalid move, targets are not on the straight line')
            return False

        # for vertical and horizontal
        if delta_row == 0 and delta_col == 0:
            print('Invalid move, cannot move to the same square')
            return False

        # get all the indices on the line
        for i in range(max_abs_delta + 1):
            current_row = source_row + (i * delta_row) // max_abs_delta
            current_col = source_col + (i * delta_col) // max_abs_delta
            if self.config[current_row][current_col] in [Square.BLOCKED, Square.WHITE, Square.BLACK]:
                print('Invalid move, you cannot cross or enter occupied square')
                return False
        return True

    def move_amazon(self, src, dest):
        self.config[dest[0]][dest[1]] = self.config[src[0]][src[1]]
        self.config[dest[0]][dest[1]] = Square.EMPTY

    def shot_arrow(self, dest):
        self.config[dest[0]][dest[1]] = Square.BLOCKED

    # wyjasnienie algorytmu https://www.baeldung.com/cs/flood-fill-algorithm
    def count_territory(self):
        def update_territory(territory):
            count = 0
            for row in range(size):
                for col in range(size):
                    if territory_map[row][col] == Territory.ANALYZED:
                        territory_map[row][col] = territory
                        count += 1
        def search_neighbours(start_row, start_col):
            queue = [(start_row, start_col)]
            encountered_square_states = {}
            while queue:
                row, col = queue.pop()
                territory_map[row][col] = Territory.ANALYZED
                for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    next_row, next_col = (row + i, col + j)
                    if next_row < 0 or next_row >= size or next_col < 0 or next_col >= size:
                        continue
                    if self.config[next_row][next_col] == Square.EMPTY and territory_map[next_row][next_col] == Territory.UNKNOWN:
                        territory_map[next_row][next_col] = Territory.ANALYZED
                        queue.append((next_col, next_col))
                    elif self.config[next_row][next_col] != Square.EMPTY:
                        encountered_square_states[self.config[next_row][next_col]] = 1
                        territory_map[next_row][next_col] = Territory.OCCUPIED
            if Square.WHITE in encountered_square_states and not Square.BLACK in encountered_square_states:
                return update_territory(Territory.WHITE), 0, 0
            elif Square.BLACK in encountered_square_states and not Square.WHITE in encountered_square_states:
                return 0, update_territory(Territory.BLACK), 0
            elif Square.WHITE in encountered_square_states and Square.BLACK in encountered_square_states:
                return 0, 0, update_territory(Territory.NEUTRAL)
            else:
                update_territory(Territory.DEADSPACE)
                return 0, 0, 0
        size = len(self.config)
        territory_map = [[Territory.UNKNOWN for _ in range(size)] for _ in range(size)]
        w_territory = b_territory = n_territory = 0
        for row in range(size):
            for col in range(size):
                if territory_map[row][col] == Territory.UNKNOWN and self.config[row][col] == Square.EMPTY:
                    w, b, n = search_neighbours(row, col)
                    w_territory += w
                    b_territory += b
                    n_territory += n
                elif territory_map[row][col] == Territory.UNKNOWN and self.config[row][col] != Square.EMPTY:
                    territory_map[row][col] = Territory.OCCUPIED
        if n_territory == 0:
            if w_territory > b_territory:
                return w_territory, -1
            else:
                return -1, b_territory
        else:
            return w_territory + n_territory, b_territory + n_territory

if __name__ == '__main__':
    Game(BoardSize.TenByTen).run()
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

    def get_piece_coords():
        white_queen_idx = 1
        black_queen_idx = 7
        cols = 6
        rows = 2
        rect = spritesheet.get_rect()
        w = rect.width // cols
        h = rect.height // rows
        coords = {
            Square.WHITE: (white_queen_idx % cols * w, white_queen_idx // cols * h, w, h),
            Square.BLACK: (black_queen_idx % cols * w, black_queen_idx // cols * h, w, h)
        }
        return coords

    def draw_pieces(screen):
        for row in range(ROWS):
            for col in range(COLS):
                screen.blit(spritesheet,  (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), get_piece_coords()[Square.WHITE])

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chessboard")
    spritesheet = pygame.image.load(os.path.join('res', 'pieces.png')).convert_alpha()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill(WHITE)  # Fill the screen with a white background
        draw_board(screen)
        draw_pieces(screen)
        pygame.display.update()
    pygame.quit()
