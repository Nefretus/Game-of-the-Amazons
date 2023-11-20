import copy
from enum import Enum
import numpy as np
import os
import pygame


class BoardSize(Enum):
    SixBySix = 6
    EightByEight = 8
    TenByTen = 10


class Player(Enum):
    WHITE = 0
    BLACK = 1

    @classmethod
    def other(cls, player):
        if player == cls.WHITE:
            return Player.BLACK
        elif player == cls.BLACK:
            return Player.WHITE
        else:
            return player


class Square(Enum):
    WHITE = 0  # there is white amazon on that square
    BLACK = 1  # there is black amazon on that square
    EMPTY = 2  # square is empty
    BLOCKED = 3  # square blocked


class Territory(Enum):
    WHITE = 0  # territory belongs to black
    BLACK = 1  # territory belongs to white
    NEUTRAL = 2  # shared territory
    DEADSPACE = 3  # territory is closed and cannot be entered by any amazon
    OCCUPIED = 4  # territory is blocked or has amazon on it
    ANALYZED = 5  # state is only active while searching for connected areas
    UNKNOWN = 6  # square not yet analyzed


class Game:
    def __init__(self, board_size, human=False):
        self.board = Board(board_size)
        self.running = True

        self.human = human
        if self.human:
            self.init_interface(board_size)

        self.target_square = None
        self.source_square = None
        # in case that shooting arrow fails, we need to return to starting position
        self.starting_square = None
        self.ending_square = None

        self.mouse_button_clicked = False
        self.first_move_done = False

    def __del__(self):
        if self.human:
            pygame.quit()

    def init_interface(self, board_size):
        pygame.init()
        self.screen = pygame.display.set_mode((self.board.board_size[0], self.board.board_size[1]))
        pygame.display.set_caption("Chessboard")
        self.spritesheet = pygame.image.load(os.path.join('res', 'pieces.png')).convert_alpha()

        # depends on the spritesheet format
        white_queen_idx = 1
        black_queen_idx = 7
        cols = 6
        rows = 2

        rect = self.spritesheet.get_rect()
        w = rect.width // cols
        h = rect.height // rows
        self.amazon_sprite_coords = {
            Square.WHITE: (white_queen_idx % cols * w, white_queen_idx // cols * h, w, h),
            Square.BLACK: (black_queen_idx % cols * w, black_queen_idx // cols * h, w, h)
        }

    def draw_board(self):
        WHITE = (255, 255, 255)
        BLACK = (33, 12, 125)
        for row in range(self.board.square_count):
            for col in range(self.board.square_count):
                color = WHITE if (row + col) % 2 == 0 else BLACK
                pygame.draw.rect(self.screen, color, (col * self.board.square_width, row * self.board.square_height,
                                                      self.board.square_width, self.board.square_height))

    def draw_pieces(self):
        for i in range(self.board.square_count):
            for j in range(self.board.square_count):
                queen = self.board.config[i][j]
                if queen in [Square.WHITE, Square.BLACK]:
                    sprite = self.spritesheet.subsurface(pygame.Rect(self.amazon_sprite_coords[queen]))
                    scaled_sprite = pygame.transform.smoothscale(sprite,
                                                                 (self.board.square_width, self.board.square_height))
                    self.screen.blit(scaled_sprite,
                                     (j * self.board.square_width, i * self.board.square_height,
                                      self.board.square_width, self.board.square_height))
                if queen == Square.BLOCKED:
                    pygame.draw.rect(self.screen,
                                     (0, 0, 0),
                                     (j * self.board.square_width, i * self.board.square_height,
                                      self.board.square_width, self.board.square_height))

    def update(self):
        self.move_piece()
        print(self.board.count_territory())

    def move_piece(self):
        # find out which player clicked on
        if self.mouse_button_clicked:
            position = pygame.mouse.get_pos()
            for i in range(self.board.square_count):
                for j in range(self.board.square_count):
                    rect = pygame.Rect(j * self.board.square_width, i * self.board.square_height,
                                       self.board.square_width, self.board.square_height)
                    if rect.collidepoint(position[0], position[1]):
                        if not self.first_move_done:
                            if self.source_square is not None:
                                self.target_square = (i, j)
                            else:
                                self.source_square = (i, j)
                        else:
                            # ending square is the one from the first move
                            self.source_square = self.ending_square
                            self.target_square = (i, j)
        # we want to recolor selected square
        if self.source_square is not None and self.target_square is None and not self.first_move_done:
            transparent_blue = (28, 21, 212, 170)
            surface = pygame.Surface((self.board.square_width, self.board.square_height), pygame.SRCALPHA)
            surface.fill(transparent_blue)
            self.screen.blit(surface, (
                self.source_square[1] * self.board.square_width, self.source_square[0] * self.board.square_height))
        elif self.target_square is not None:
            if self.board.is_valid_move(self.source_square, self.target_square):
                if not self.first_move_done:
                    self.board.move_amazon(self.source_square, self.target_square)
                    self.starting_square = self.source_square
                    self.ending_square = self.target_square
                    self.first_move_done = True
                else:
                    self.board.shoot_arrow(self.target_square)
                    self.board.white_to_play = not self.board.white_to_play
                    self.first_move_done = False
            elif self.first_move_done:
                # return to the starting square
                self.board.move_amazon(self.ending_square, self.starting_square)
                self.first_move_done = False
            self.target_square = None
            self.source_square = None
        self.mouse_button_clicked = False

    def run(self):
        # human vs human
        clock = pygame.time.Clock()
        while self.running:
            clock.tick(60)
            # get move
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.mouse_button_clicked = True
            # self.screen.fill(WHITE)  # Fill the screen with a white background
            self.draw_board()
            self.update()
            self.draw_pieces()
            pygame.display.update()
            # check whether the game ended
            # draw on the screen


class Board:
    def __init__(self, board_size):
        self.white_to_play = True
        # if this is false, player shoots the arrow, not moving amazon during his turn
        self.amazon_move = True
        self.square_count = board_size.value
        self.board_size = (480, 480)
        self.square_width = self.board_size[0] // self.square_count
        self.square_height = self.board_size[1] // self.square_count
        self.config = self.generate_start_config(board_size)

    # self.print_board()

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
    def is_valid_move(self, source, destination, omit_player_checking=False):
        source_row, source_col = source
        dest_row, dest_col = destination

        # source square has to be the one containing either black or white amazon
        # AI algorithm doesn't rely on internal white_to_play variable, so there is option to turn it off
        if not omit_player_checking:
            if (self.white_to_play and self.config[source_row][source_col] != Square.WHITE) or (
                    not self.white_to_play and self.config[source_row][source_col] != Square.BLACK):
                #   print('Invalid move, there is no piece on this square', source_row, source_col)
                return False

        # move can only be straight line
        delta_row = dest_row - source_row
        delta_col = dest_col - source_col
        max_abs_delta = max(abs(delta_row), abs(delta_col))

        # tanges 45 = 1 = x / y, zeby diagolnie lezaly na tej samej prostej kat musi być miedzy nimi 45 stopni
        if delta_row != 0 and delta_col != 0 and abs(delta_row / delta_col) != 1:
            # print('Invalid move, targets are not on the straight line')
            return False

        # for vertical and horizontal
        if delta_row == 0 and delta_col == 0:
            #  print('Invalid move, cannot move to the same square')
            return False

        # get all the indices on the line
        # zmienic zeby nie uwzgledniac source i destination
        for i in range(1, max_abs_delta + 1):
            current_row = source_row + (i * delta_row) // max_abs_delta
            current_col = source_col + (i * delta_col) // max_abs_delta
            if self.config[current_row][current_col] in [Square.BLOCKED, Square.WHITE, Square.BLACK]:
                # print('Invalid move, you cannot cross or enter occupied square')
                return False
        return True

    def move_amazon(self, src, dest):
        self.config[dest[0]][dest[1]] = self.config[src[0]][src[1]]
        self.config[src[0]][src[1]] = Square.EMPTY

    def shoot_arrow(self, dest):
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
            return count

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
                    if self.config[next_row][next_col] == Square.EMPTY and territory_map[next_row][
                        next_col] == Territory.UNKNOWN:
                        territory_map[next_row][next_col] = Territory.ANALYZED
                        queue.append((next_row, next_col))
                    elif self.config[next_row][next_col] != Square.EMPTY:
                        encountered_square_states[self.config[next_row][next_col]] = 1
                        territory_map[next_row][next_col] = Territory.OCCUPIED
            if Square.WHITE in encountered_square_states and Square.BLACK not in encountered_square_states:
                return update_territory(Territory.WHITE), 0, 0
            elif Square.BLACK in encountered_square_states and Square.WHITE not in encountered_square_states:
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
                elif territory_map[row][col] == Territory.UNKNOWN:
                    territory_map[row][col] = Territory.OCCUPIED
        if n_territory == 0:
            if w_territory > b_territory:
                return w_territory, -1
            else:
                return -1, b_territory
        else:
            return w_territory + n_territory, b_territory + n_territory


class BoardAI(Board):
    def __init__(self, board_size):
        super().__init__(board_size)

    # queenlike moves apply to amazon and arrow moves
    # input: starting square
    # output: possible ending squares
    def get_queenlike_moves(self, start):
        moves = []
        for i in range(len(self.config)):
            for j in range(len(self.config)):
                target = (i, j)
                if self.is_valid_move(start, target, omit_player_checking=True):
                    moves.append((start, target))
        return moves

    # input: player - white or black
    # output: (start, target, arrow) - all possible positions
    def get_possible_moves(self, player):
        output = []
        if player == Player.WHITE:
            color = Square.WHITE
        else:
            color = Square.BLACK
        for i in range(len(self.config)):
            for j in range(len(self.config)):
                if self.config[i][j] == color:
                    start = (i, j)
                    for _, amazon_target in self.get_queenlike_moves(start):
                        state = self.get_new_state_for_amazon_move(start, amazon_target)
                        for _, arrow_target in state.get_queenlike_moves(amazon_target):
                            output.append((start, amazon_target, arrow_target))
        return output

    # takes the amazon move and returns new board after this move is applied
    def get_new_state_for_amazon_move(self, amazon_start, amazon_target):
        board = copy.deepcopy(self)
        board.move_amazon(amazon_start, amazon_target)
        return board

    # takes the arrow move and returns new board after this move is applied
    def get_new_state_for_arrow_move(self, move):
        board = copy.deepcopy(self)
        board.shoot_arrow(move)
        return board

    # takes the full in the form of (start, end, arrow) and returns new board after this move is applied
    def get_new_state_for_full_move(self, move):
        board = copy.deepcopy(self)
        amazon_start, amazon_target, arrow = move
        board.move_amazon(amazon_start, amazon_target)
        board.shoot_arrow(arrow)
        return board

    # this is needed for comparing states
    def __eq__(self, other):
        for i in range(len(self.config)):
            for j in range(len(self.config)):
                if self.config[i][j] != other.config[i][j]:
                    return False
        return True

    def __hash__(self):
        return hash(str(self.config))


from datetime import datetime, timedelta
import random
from math import log, sqrt


class MonteCarloTreeSearch:
    def __init__(self, seconds=30):
        self.starting_board = BoardAI(BoardSize.SixBySix)

        # Larger values of C will encourage more exploration of the possibilities,
        # and smaller values will cause the AI to prefer concentrating on known good moves
        self.C = 1.4

        # idx 0 - wins
        # idx 1 - plays
        self.states = dict()

        # true - white player, false - player black
        # update the path during back propagation
        self.path = {
            Player.WHITE: [],
            Player.BLACK: []
        }
        self.player = Player.WHITE

        self.calculation_time = timedelta(seconds=seconds)
        begin = datetime.now()
        while datetime.now() - begin < self.calculation_time:
            self.path[Player.WHITE].clear()
            self.path[Player.BLACK].clear()
            self.simulate()
        print('siema')

    # def select_and_expand(self, init_state, player):
    #     max_score = None
    #     best_state = None
    #     moves = init_state.get_possible_moves(player)
    #     random.shuffle(moves)
    #     #print(len(self.path[player]))
    #     for move in moves:
    #         state = init_state.get_new_state_for_full_move(move)
    #         # we found unexplored node, expand on it
    #         if state not in self.states:
    #             self.path[player].append(state)
    #             self.states[state] = (0, 0)
    #             return state, Player.other(player)
    #         wins, plays = self.states[state]
    #         if max_score is None or wins / plays > max_score:
    #             max_score = (wins / plays) + (1.4 * sqrt(log(plays) / plays))
    #             best_state = state
    #     self.path[player].append(best_state)
    #     # all the moves for one player are evaluated, search for the other one
    #     return self.select_and_expand(best_state, Player.other(player))

    def select_and_expand(self, init_state, player):
        best_state = None
        next_states = [
            init_state.get_new_state_for_full_move(move)
            for move in init_state.get_possible_moves(player)
        ]
        if all(self.states.get(state) for state in next_states):
            log_total = log(sum(self.states[state][1] for state in next_states))
            value, best_state = max(
                ((self.states[state][0] / self.states[state][1]) +
                 self.C * sqrt(log_total / self.states[state][1]),
                 state)
                for state in next_states
            )
        else:
            best_state = random.choice(next_states)
            # we found unexplored node, expand on it
            if best_state not in self.states:
                self.path[player].append(best_state)
                self.states[best_state] = (0, 0)
                return best_state, Player.other(player)
        self.path[player].append(best_state)
        return self.select_and_expand(best_state, Player.other(player))

    def backpropagation(self, player, winner):
        for state in self.path[player]:
            if state in self.states:
                wins, plays = self.states[state]
                if winner:
                    wins += 1
                plays += 1
                self.states[state] = (wins, plays)

    def simulate(self):
        state, self.player = self.select_and_expand(self.starting_board, self.player)
        # play one game until the end
        while True:
            moves = state.get_possible_moves(self.player)
            if not moves:
                break
            state = state.get_new_state_for_full_move(random.choice(moves))
            self.path[self.player].append(state)
            self.player = Player.other(self.player)
        self.backpropagation(Player.other(self.player), True)
        self.backpropagation(self.player, False)


if __name__ == '__main__':
    print(Square.WHITE == Player.WHITE)
    MonteCarloTreeSearch(seconds=155)
    # Game(BoardSize.SixBySix, human=True).run()

# Cel na rano: zapis do pliku
