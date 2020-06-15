import sys

import gym
import numpy as np
from gym import spaces, error, utils


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
        self._seed()
        self._reset()
    
    def _seed(self):

    def _reset(self):

    
