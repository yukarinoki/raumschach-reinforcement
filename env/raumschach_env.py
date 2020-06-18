import sys
from copy import copy

import gym
import numpy as np
from gym import spaces, error, utils
from gym.utils import seeding


# _step(self, action)	    action を実行し、結果を返す
# _reset(self)	            状態を初期化し、初期の観測値を返す
# _render(self,             mode='human', close=False)	環境を可視化する
# _close(self)	            環境を閉じて後処理をする                        optional
# _seed(self, seed=None)	ランダムシードを固定する                        optional
# プロパティ
# action_space	行動(Action)の張る空間
# observation_space	観測値(Observation)の張る空間
# reward_range	報酬の最小値と最大値のリスト

pieces_to_ids = {
    'R1': 1, 'N1': 2, 'K': 1, 'N2': 3, 'R2': 4,
    'P1': 5, 'P2': 6, 'P3': 7, 'P4': 8, 'P5': 9,
    'U1': 10, 'B1': 11, 'Q': 12, 'U2': 13, 'B2': 14,
    'P7': 15, 'P2': 16, 'P8': 17, 'P9': 18, 'P10': 19,
    'r1': -1, 'n1': -2, 'k': -1, 'n2': -3, 'r2': -4,
    'p1': -5, 'p2': -6, 'p3': -7, 'p4': -8, 'p5': -9,
    'u1': -10, 'b1': -11, 'q': -12, 'u2': -13, 'b2': -14,
    'p7': -15, 'p2': -16, 'p8': -17, 'p9': -18, 'p10': -19,
    '.': 0
}
uniDict = {
	'p': "♙", 'r': "♖", 'n': "♘", 'b': "♗", 'k': "♔", 'q': "♕", 'u': "🦄",
	'P': "♟", 'R': "♜", 'N': "♞", 'B': "♝", 'K': "♚", 'Q': "♛" , 'U': "🦄",
	'.': '.'
}
def check_position_validity(l,r,c):
    arr = [0,1,2,3,4]
    return l in arr and r in arr and c in arr

def make_random_policy(np_random):
    def random_policy(state):
        opp_player = -1
        moves = RaumschachEnv.get_possible_moves(state, opp_player)
        # No moves left
        if len(moves) == 0:
            return 'resign'
        else:
            return np.random.choice(moves)
    return random_policy

class RaumschachEnv(gym.Env) :
    def __init__(self, player_color=1, opponent="random", log=True):
        self.moves_max = 149
        self.log = log
        
        # One action (for each board position) x (no. of pieces), 2xcastles, offer/accept draw and resign
        self.observation_space = spaces.Box(-20,20, (5,5,5)) # board 5x5x5
        self.action_space = spaces.Discrete(125*20 + 5)

        self.player = player_color # define player 
        self.opponent = opponent # define opponent

        # reset and build state
        # self._seed()
        self._reset()
        self._render()
   	
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        # Update the random policy if needed
        if isinstance(self.opponent, str):
            if self.opponent == 'random':
                self.opponent_policy = make_random_policy(self.np_random)
            elif self.opponent == 'none':
                self.opponent_policy = None
            else:
                raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))
        else:
            self.opponent_policy = self.opponent

        return [seed]

    def _reset(self):
		# reset pieces (pawns that became queen become pawns again)
        RaumschachEnv.ids_to_pieces = {v: k for k, v in pieces_to_ids.items()}
		# vars
        self.state = {}
        self.done = False
        self.current_player = 1
        self.saved_states = {}
        self.repetitions = 0     # 3 repetitions ==> DRAW
        # register king's and rooks' (and all other pieces) moves
        pieces = np.linspace(1, 20, 20, dtype=int)
        self.state['kr_moves'] = {**{p: 0 for p in pieces}, **{-p: 0 for p in pieces}} 
		# material captured
        self.state['captured'] = {1: [], -1: []}
		# current move
        self.state['on_move'] = 1
        # Board
        board = [[['R1', 'N1', 'K', 'N2', 'R2'], ['P1', 'P2', 'P3', 'P4', 'P5']] + [['.']*5] * 3]
        board += [[['U1', 'B1', 'Q', 'U1', 'B1'],  ['P1', 'P2', 'P3', 'P4', 'P5']] + [['.']*5] * 3]
        board += [[['.']*5] * 5]
        board += [[['.']*5] * 3 + [ ['p1', 'p2', 'p3', 'p4', 'p5'], ['u1', 'b1', 'q', 'u1', 'b1']]]
        board += [[['.']*5] * 3 + [ ['p1', 'p2', 'p3', 'p4', 'p5'], ['r1', 'n1', 'k', 'n2', 'r2']]]
        self.state['board'] = np.array([[[pieces_to_ids[x] for x in row] for row in layer] for layer in board])
        self.state['prev_board'] = copy(self.state['board'])
        return self.state

    def _render(self, mode='human', close=False):
        cb = self.state['board'] 
        layer_index_arr = ['A', 'B', 'C', 'D', 'E']
        row_index_arr = ['1', '2', '3', '4', '5']
        column_index_arr = ['a', 'b', 'c', 'd', 'e']

        for l in range(5):
            print(layer_index_arr[l])
            print('   ', end='')
            for c in range(5):  print(column_index_arr[c], end='  ')
            print(" ")
            for r in range(5):
                print(row_index_arr[r], end='  ')
                for c in range(5):  print(uniDict[RaumschachEnv.ids_to_pieces[cb[l][r][c]][0]], end='  ')
                print("\n")
            print("\n")   
        print("//// board rendering  ////")
    def get_possible_moves(self, state, player):
        moves = []
        for l in range(5):
            for r in range(5):
                for c in range(5):
                    pi = RaumschachEnv.ids_to_pieces[state["board"][l][r][c]][0]
                    if (pi.islower() and player == -1) or (pi.isupper() and player == 1):
                        tp = pi.upper()
                        if tp == 'K':
                            moves += generate_king_moves(l,r,c, state, player)
                        elif tp == 'Q':
                            moves += generate_queen_moves(l,r,c, state, player)
                        elif tp == 'R':
                            moves += generate_rook_moves(l,r,c, state, player)
                        elif tp == 'B':
                            moves += generate_bishop_moves(l,r,c, state, player)
                        elif tp == 'N':
                            moves += generate_knight_moves(l,r,c, state, player)
                        elif tp == 'U':
                            moves += generate_unicorn_moves(l,r,c, state, player)
                        elif tp == 'P':
                            moves += generate_pawn_moves(l,r,c, state, player)
        return moves

    def check_movable(self,l,r,c,state,player):
            return check_position_validity(l,r,c) and ((player < 0 and state["board"] >= 0) or (player > 0 and state["board"] <= 0))

    def check_takable_pawn(self,l,r,c,state,player):
        return check_position_validity(l,r,c) and (player < 0 and state["board"] > 0) or (player > 0 and state["board"] < 0)

    def check_movable_pawn(self,l,r,c,state,player):
        return check_position_validity(l,r,c) and state["board"] == 0

    def generate_king_moves(self, l, r, c, state, player):
        dif_array =   [(dx,dy,dz) for dx in range(-1,2) for dy in range(-1,2) for dz in range(-1,2)].remove((0,0,0))
        moves = []
        for (dx,dy,dz) in dif_array:
            if check_movable(l + dx, r + dy, c + dz, state, player):
                moves += [(state["board"][l][r][c], (l + dx, r + dy, c + dz))]
        return moves

    def generate_knight_moves(self, l,r,c,state, player):
        dif_array = [(0,2,3), (0,3,2), (0,-3,2), (0,2,-3), (0,3,-2), (0,-2,3), (0,-3,-2), (0,-2,-3),(2,0,3), (3,0,2), (-3,0,2), (2,0,-3), (3,0,-2), (-2,0,3), (-3,0,-2), (-2,0,-3), (2,3,0), (3,2,0), (-3,2,0), (2,-3,0), (3,-2,0), (-2,3,0), (-3,-2,0), (-2,-3,0)]
        moves = []
        for (dx,dy,dz) in dif_array:
            if check_movable(l + dx, r + dy, c + dz, state, player):
                moves += [(state["board"][l][r][c], (l + dx, r + dy, c + dz))]
        return moves

    def generate_pawn_moves(self, l,r,c,state, player):
        p = player
        adv = [(1,0,0), (0,1,0)]
        take = [(1,1,0), (0,1,1), (0,1,-1), (1,0,-1), (1,0,1)]
        moves = []
        for (dx,dy,dz) in adv:
            if check_movable_pawn(l + p*dx, r + p*dy, c + p*dz, state, player):
                moves += [(state["board"][l][r][c], (l + dx, r + dy, c + dz))]
        for (dx,dy,dz) in take:
            if check_takable_pawn(l + p*dx, r + p*dy, c + p*dz, state, player):
                moves += [(state["board"][l][r][c], (l + p*dx, r + p*dy, c + p*dz))]
        return moves

    def generate_queen_moves(self,l,r,c,state, player):
        dif_array =   [(dx,dy,dz) for dx in range(-1,2) for dy in range(-1,2) for dz in range(-1,2)].remove((0,0,0))
        moves = []
        for (dx,dy,dz) in dif_array:
            for k in range(1, 4):
                (nx, ny, nz) = (l + k*dx, r + k*dy,c + k*dz)
                if check_movable(nx,ny,nz,state, player):
                    moves += [(state["board"][nx][ny][nz], (l + dx, r + dy, c + dz))]
                elif not check_position_validity(nx,ny,nz):
                    break
        return moves

    def generate_rook_moves(self, l,r,c,state, player):
        dif_array = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        moves = []
        for (dx,dy,dz) in dif_array:
            for k in range(1, 4):
                (nx, ny, nz) = (l + k*dx, r + k*dy,c + k*dz)
                if check_movable(nx,ny,nz,state, player):
                    moves += [(state["board"][nx][ny][nz], (l + dx, r + dy, c + dz))]
                elif not check_position_validity(nx,ny,nz):
                    break
        return moves
    def generate_bishop_moves(self, l,r,c,state, player):
        dif_array = [(1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0), (1,0,1), (1,0,-1), (-1,0,1), (-1,0,-1), (0,1,1), (0,-1,1), (0,1,-1), (0,-1,-1)]
        moves = []
        for (dx,dy,dz) in dif_array:
            for k in range(1, 4):
                (nx, ny, nz) = (l + k*dx, r + k*dy,c + k*dz)
                if check_movable(nx,ny,nz,state, player):
                    moves += [(state["board"][nx][ny][nz], (l + dx, r + dy, c + dz))]
                elif not check_position_validity(nx,ny,nz):
                    break
        return moves

    def generate_unicorn_moves(self, l,r,c,state, player):
        dif_array = [(1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1), (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1)]
        moves = []
        for (dx,dy,dz) in dif_array:
            for k in range(1, 4):
                (nx, ny, nz) = (l + k*dx, r + k*dy,c + k*dz)
                if check_movable(nx,ny,nz,state, player):
                    moves += [(state["board"][nx][ny][nz], (l + dx, r + dy, c + dz))]
                elif not check_position_validity(nx,ny,nz):
                    break
        return moves

