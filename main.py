import copy
from enum import Enum
import os


class BoardSize(Enum):
    FourByFour = 4
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

        # exiting flags
        self.running = True
        self.exit = False

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
            # game has ended, wait for the user to close the game
            while not self.exit:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.exit = True
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
                amazon = self.board.config[i][j]
                if amazon in [Square.WHITE, Square.BLACK]:
                    sprite = self.spritesheet.subsurface(pygame.Rect(self.amazon_sprite_coords[amazon]))
                    scaled_sprite = pygame.transform.smoothscale(sprite,
                                                                 (self.board.square_width, self.board.square_height))
                    self.screen.blit(scaled_sprite,
                                     (j * self.board.square_width, i * self.board.square_height,
                                      self.board.square_width, self.board.square_height))
                if amazon == Square.BLOCKED:
                    pygame.draw.rect(self.screen,
                                     (0, 0, 0),
                                     (j * self.board.square_width, i * self.board.square_height,
                                      self.board.square_width, self.board.square_height))

    def update(self):
        # for now, human will play as white and bot as black
        if self.board.white_to_play:
            #new_board = monte_carlo_decide(self.board, Player.WHITE)
            new_board = alpha_beta_decide(self.board, 3, Player.WHITE)
            self.board.update_config(new_board.config)
            self.board.white_to_play = False
        else:
            self.move_piece_human()
        w_terr, b_terr = self.board.count_territory()
        if w_terr == -1:
            print('Black won!')
            self.running = False
        if b_terr == -1:
            print('White won!')
            self.running = False

    def move_piece_human(self):
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
        clock = pygame.time.Clock()
        while not self.exit:
            while self.running:
                clock.tick(60)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        # additional flag to wait after the game ended, dirty i know...
                        self.exit = True
                        self.running = False
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        self.mouse_button_clicked = True
                self.draw_board()
                self.update()
                self.draw_pieces()
                pygame.display.update()
            #reset board
            self.reset()

    def reset(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.KEYDOWN:
                    self.board = Board(BoardSize.FourByFour)
                    self.running = True
                    return
                if event.type == pygame.QUIT:
                    self.exit = True
                    return

class Board:
    def __init__(self, board_size):
        self.white_to_play = True
        self.square_count = board_size.value
        self.board_size = (360, 360)
        self.square_width = self.board_size[0] // self.square_count
        self.square_height = self.board_size[1] // self.square_count
        self.config = self.generate_start_config(board_size)

    def generate_start_config(self, board_size):
        config = [[Square.EMPTY for _ in range(board_size.value)] for _ in range(board_size.value)]
        white_coords = []
        black_coords = []
        if board_size == BoardSize.FourByFour:
            white_coords = ['d2']
            black_coords = ['a3']
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

        # source square has to be the one containing either black or white amazon (depending who is about to make a move)
        # my AI algorithm doesn't keep track of internal white_to_play variable, so I made option to turn it off
        if not omit_player_checking:
            if (self.white_to_play and self.config[source_row][source_col] != Square.WHITE) or (
                    not self.white_to_play and self.config[source_row][source_col] != Square.BLACK):
                return False

        # move can only be over the straight line
        delta_row = dest_row - source_row
        delta_col = dest_col - source_col
        max_abs_delta = max(abs(delta_row), abs(delta_col))

        # or diagonally
        if delta_row != 0 and delta_col != 0 and abs(delta_row / delta_col) != 1:
            return False

        # cannot move to the same square
        if delta_row == 0 and delta_col == 0:
            return False

        # cannot cross or enter occupied square
        for i in range(1, max_abs_delta + 1):
            current_row = source_row + (i * delta_row) // max_abs_delta
            current_col = source_col + (i * delta_col) // max_abs_delta
            if self.config[current_row][current_col] in [Square.BLOCKED, Square.WHITE, Square.BLACK]:
                return False
        return True

    def move_amazon(self, src, dest):
        self.config[dest[0]][dest[1]] = self.config[src[0]][src[1]]
        self.config[src[0]][src[1]] = Square.EMPTY

    def shoot_arrow(self, dest):
        self.config[dest[0]][dest[1]] = Square.BLOCKED

    # more info: https://www.baeldung.com/cs/flood-fill-algorithm
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
                return w_territory - b_territory, -1
            else:
                return -1, b_territory - w_territory
        else:
            return w_territory + n_territory, b_territory + n_territory

    def update_config(self, other):
        for i in range(len(self.config)):
            for j in range(len(self.config)):
                self.config[i][j] = other[i][j]

class BoardAI(Board):
    def __init__(self, board_size, config=None):
        super().__init__(board_size)
        if config is not None:
            self.config = config

    # queenlike moves apply to amazon and arrow moves
    # input: starting square
    # output: list of possible ending squares
    def get_queenlike_moves(self, start):
        moves = []
        for i in range(len(self.config)):
            for j in range(len(self.config)):
                target = (i, j)
                if self.is_valid_move(start, target, omit_player_checking=True):
                    moves.append((start, target))
        return moves

    # input: player (Player.white or Player.black)
    # output: [(start, target, arrow)] - list of all possible positions
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

    # takes the full move in the form of (start, end, arrow) and returns new board after this move is applied
    def get_new_state_for_full_move(self, move):
        board = copy.deepcopy(self)
        amazon_start, amazon_target, arrow = move
        board.move_amazon(amazon_start, amazon_target)
        board.shoot_arrow(arrow)
        return board

    # this is needed for comparing states and saving data to file
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
import pickle
from math import log, sqrt


from collections import defaultdict

class MonteCarlo:
    def __init__(self, file_path, seconds=30):
        self.starting_board = BoardAI(BoardSize.FourByFour)
        nodes_path = file_path + 'nodes.pkl'
        wins_path = file_path + 'wins.pkl'
        plays_path = file_path + 'plays.pkl'

        # Larger values of C will encourage more exploration of the possibilities,
        # and smaller values will cause the AI to prefer concentrating on known good moves
        self.C = 1.4

        self.wins = defaultdict(int)
        self.plays = defaultdict(int)
        self.children = dict()

        if os.path.isfile(nodes_path):
            with open(nodes_path, 'rb') as f:
                self.children = pickle.load(f)
        if os.path.isfile(wins_path):
            with open(wins_path, 'rb') as f:
                self.wins = pickle.load(f)
        if os.path.isfile(plays_path):
            with open(plays_path, 'rb') as f:
                self.plays = pickle.load(f)
            print('Data read from file')

        self.calculation_time = timedelta(seconds=seconds)
        save_timer = begin = datetime.now()
        while datetime.now() - begin < self.calculation_time:
            self.do_rollout((Player.WHITE, self.starting_board))
            # after one minute of training save data to file
            if datetime.now() - save_timer > timedelta(seconds=60):
                print('Data serialized', datetime.now())
                with open(nodes_path, 'wb') as f:
                    pickle.dump(self.children, f)
                with open(wins_path, 'wb') as f:
                    pickle.dump(self.wins, f)
                with open(plays_path, 'wb') as f:
                    pickle.dump(self.plays, f)
                save_timer = datetime.now()

    #node is (player, state)
    def do_rollout(self, node):
        path = self.select(node)
        leaf = path[-1]
        self.expand(leaf)
        reward = self.simulate(leaf)
        self.backpropagate(path, reward)

    def choose(self, node):
        moves = node.get_possible_moves()
        if len(moves) == 0:
            raise RuntimeError(f"choose called on terminal node {node}")
        if node not in self.children:
            return random.choice([node.get_new_state_for_full_move() for move in moves])
        def score(n):
            if self.plays[n] == 0:
                return float("-inf")
            return self.wins[n] / self.plays[n]
        return max(self.children[node], key=score)

    def _uct_select(self, node):
        def uct(n):
            return self.wins[n] / self.plays[n] + self.C * sqrt(
                log(self.plays[node]) / self.plays[n]
            )
        return max(self.children[node], key=uct)

    def select(self, node):
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                return path
            unexplored = [child for child in self.children[node] if child not in self.children]
            if unexplored:
                player, state = random.choice(unexplored)
                path.append((player, state))
                return path
            node = self._uct_select(node)

    def simulate(self, node):
        while True:
            player, state = node
            moves = state.get_possible_moves(player)
            if len(moves) == 0:
                reward = 1
                return reward
            node = random.choice([(Player.other(player), state.get_new_state_for_full_move(move)) for move in moves])

    def expand(self, node):
        player, state = node
        if node in self.children:
            return # already expanded
        self.children[node] = [(Player.other(player), state.get_new_state_for_full_move(move)) for move in state.get_possible_moves(player)]

    def backpropagate(self, path, reward):
        for node in reversed(path):
            self.plays[node] += 1
            self.wins[node] += reward
            reward = 1 - reward

def alpha_beta_search(board, depth, player, alpha, beta):
    w_terr, b_terr = board.count_territory()
    b_terr = -b_terr # evaluate black territory as negative
    if depth == 0 or w_terr == -1 or b_terr == 1:
        return w_terr if player == Player.WHITE else b_terr, None
    min_eval = float("inf")
    max_eval = float("-inf")
    best_move = None
    if player == Player.WHITE:
        for move in board.get_possible_moves(player):
            new_board = board.get_new_state_for_full_move(move)
            eval, _ = alpha_beta_search(new_board, depth - 1, Player.BLACK, alpha, beta)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, max_eval)
            if beta <= alpha:
                break
    else:
        for move in board.get_possible_moves(player):
            new_board = board.get_new_state_for_full_move(move)
            eval, _ = alpha_beta_search(new_board, depth - 1, Player.WHITE, alpha, beta)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, min_eval)
            if beta <= alpha:
                break
    return max_eval if player == Player.WHITE else min_eval, best_move

def alpha_beta_decide(board, depth, player):
    boardAI = BoardAI(BoardSize.FourByFour, copy.deepcopy(board.config))
    _, best_move = alpha_beta_search(boardAI, depth, player, float("-inf"), float("inf"))
    return boardAI.get_new_state_for_full_move(best_move)

bot = MonteCarlo('TEST_', seconds=0)

import pygame

def monte_carlo_decide(board, player):
    boardAI = BoardAI(BoardSize.FourByFour, copy.deepcopy(board.config))
    moves = boardAI.get_possible_moves(player)
    if (player, boardAI) not in bot.children:
        return random.choice([boardAI.get_new_state_for_full_move(move) for move in boardAI.get_possible_moves(player)])
    def score(n):
        if bot.plays[n] == 0:
            return float("-inf")
        return bot.wins[n] / bot.plays[n]

    return max(bot.children[(player, boardAI)], key=score)[1]


if __name__ == '__main__':
    Game(BoardSize.FourByFour, human=True).run()
