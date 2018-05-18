from learning.online import explorers

class Parameters:
    # Main Parameters
    games_to_play = 2
    max_moves = 4
    # starting_position = '6k1/4Rppp/8/8/8/8/5PPP/6K1 w - - 0 1'

    # MCTS Parameters
    exploration_alg = explorers.Epsilon()
    exploration_rate = 1
    uct_max_depth = 12
    uct_sims = 5
    tree_sims = 1

    # Adversary Parameters
    a_verbose = True
    a_depth = None
    a_think = 1
    stockfish_nodes=None

    # Agent Parameters
    end_games={'mate_in_1':['6k1/4Rppp/8/8/8/8/5PPP/6K1 w - - 0 1',
                            '8/1p6/kp6/1p6/8/8/5PPP/5RK1 w - - 0 1',
                            'R7/4kp2/5N2/4P3/8/8/8/6K1 w - - 0 1',
                            '5r2/1b2Nppk/8/2R5/8/8/5PPP/6K1 w - - 0 1',
                            '6rk/6pp/8/6N1/8/8/8/7K w - - 0 1'],
               'mate_in_2':['2r1r1k1/5ppp/8/8/Q7/8/5PPP/4R1K1 w - - 0 1',
                            '5r1k/1b2Nppp/8/2R5/4Q3/8/5PPP/6K1 w - - 0 1',
                            '5rk1/1b3ppp/8/2RN4/8/8/2Q2PPP/6K1 w - - 0 1',
                            '6rk/6pp/6q1/6N1/8/7Q/6PP/6K1 w - - 0 1',
                            '3r3k/1p1b1Qbp/1n2B1p1/p5N1/Pq6/8/1P4PP/R6K w - - 0 1'],
               'mate_in_3':['6k1/3qb1pp/4p3/ppp1P3/8/2PP1Q2/PP4PP/5RK1 w - - 0 1',
                            '5r1b/2R1R3/P4r2/2p2Nkp/2b3pN/6P1/4PP2/6K1 w - - 0 1',
                            '2b1Q3/1kp5/p1Nb4/3P4/1P5p/p6P/K3R1P1/5q2 w - - 0 1',
                            '5rk1/1b3ppp/8/2RN4/8/8/2Q2PPP/6K1 w - - 0 1',
                            '1r5k/6pp/2pr4/P1Q3bq/1P2Bn2/2P5/5PPP/R3NRK1 b - - 0 1',
                            '5rk1/1R2R1pp/8/8/8/8/8/1K6 w - - 0 1',
                            ],
               'mate_in_4':['r2q1rk1/pbp3pp/1p1b4/3N1p2/2B5/P3PPn1/1P3P1P/2RQK2R w K - 0 1'],
               'mate_in_5':['5rk1/1R1R1p1p/4N1p1/p7/5p2/1P4P1/r2nP1KP/8 w - - 0 1'],
               'mate_in_6':['r4rk1/2R5/1n2N1pp/2Rp4/p2P4/P3P2P/qP3PPK/8 w - - 0 1',
                            'r1k4r/ppp1bq1p/2n1N3/6B1/3p2Q1/8/PPP2PPP/R5K1 w - - 0 1']}

    def update(self):
        self.games_to_play = 50