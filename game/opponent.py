import chess.uci

class Adversary:
    """
    The opponent for which our agent will play against
    """
    def __init__(self, verbose, search_depth, max_think=None, nodes = None):
        """
        :param verbose: Bool. Should print outs to the console concerning the opponents thinking be made.
        :param search_depth: The max depth for which Stockfish should conduct MCTS
        :param max_think: The maximum amount of time (milliseconds) that Stockfish should spend planning a move
        """
        self.engine = self.build_engine()
        self.verbosity = verbose
        self.depth = search_depth
        self.max_think = max_think
        self.nodes = nodes

    def build_engine(self):
        """
        Instantiate the Stockfish engine and pass it the board details so that it can later perform assessments on the
        current board's state
        :return: Engine object
        """
        engine = chess.uci.popen_engine("stockfish/src/stockfish")
        engine.uci()
        info_handler = chess.uci.InfoHandler()
        engine.info_handlers.append(info_handler)
        return engine

    def get_engine_name(self):
        """
        Print out the engine's name
        :return: Str
        """
        print('Opponent: {}'.format(self.engine.name))

    def reset(self):
        """
        Reset the Engine's memory. This should be done prior to every new game.
        :return:
        """
        self.engine.ucinewgame()
        if self.verbosity:
            print('Game Reset')

    def initialise_engine(self, board):
        """
        Pass the board's initial state to the engine. Needed when the engine plays white?
        :param board: A board object
        :return:
        """
        self.engine.position(board)

    def update_position(self, board):
        """
        Give the engine the games current state
        :param board: board objec
        :return:
        """
        self.engine.position(board)

    def calculate(self, current_board):
        """
        Run MCTS to calculate the next best move
        :param current_board: board object
        :return: A move
        """
        self.update_position(current_board.board)
        if self.max_think is not None:
            move, _ = self.engine.go(ponder=False, movetime=self.max_think)
        elif self.depth is not None:
            move, _ = self.engine.go(ponder=False, depth=self.depth)
        elif self.nodes is not None:
            move, _ = self.engine.go(ponder=False, nodes = self.nodes)
        else:
            move, _ = self.engine.go()
        return move
