import pandas as pd
import numpy as np
import chess

# %%
# import csv
puzzle_data = pd.read_csv("./data/ml-data-test.csv")

# %% initialize features and labels

PAWN_PLANE = 0
ROOK_PLANE = 1
KNIGHT_PLANE = 2
BISHOP_PLANE = 3
QUEEN_PLANE = 4
KING_PLANE = 5

FEATURES = 6 * 8 * 8  # 6 unique piece types each on an 8x8 integer board; 1 for white, -1 for black
OUTPUTS = 57  # total number of puzzle categories

WHITE = True
BLACK = False

puzzle_categories = [   "advancedPawn",
    "advantage",
    "anastasiaMate",
    "arabianMate",
    "attackingF2F7",
    "attraction",
    "backRankMate",
    "bishopEndgame",
    "bodenMate",
    "capturingDefender",
    "castling",
    "clearance",
    "coercion",
    "crushing",
    "defensiveMove",
    "discoveredAttack",
    "deflection",
    "doubleBishopMate",
    "doubleCheck",
    "dovetailMate",
    "equality",
    "enPassant",
    "exposedKing",
    "fork",
    "hangingPiece",
    "hookMate",
    "interference",
    "intermezzo",
    "kingsideAttack",
    "knightEndgame",
    "long",
    "mate",
    "mateIn5",
    "mateIn4",
    "mateIn3",
    "mateIn2",
    "mateIn1",
    "oneMove",
    "overloading",
    "pawnEndgame",
    "pin",
    "promotion",
    "queenEndgame",
    "queensideAttack",
    "quietMove",
    "rookEndgame",
    "queenRookEndgame",
    "sacrifice",
    "short",
    "simplification",
    "skewer",
    "smotheredMate",
    "trappedPiece",
    "underPromotion",
    "veryLong",
    "xRayAttack",
    "zugzwang"
]



# %% intialize featurize function
def featurize_board(board_fen, rotate=False):
    board_array = np.zeros((6, 8, 8), dtype='int8')
    f = -1 # file (column)
    r = 0 # rank (row)
    for c in board_fen:
        f += 1
        if c == 'P' or c == 'p':
            board_array[PAWN_PLANE, r, f] = 1 if c == 'P' else -1
        elif c == 'R' or c == 'r':
            board_array[ROOK_PLANE, r, f] = 1 if c == 'R' else -1
        elif c == 'N' or c == 'n':
            board_array[KNIGHT_PLANE, r, f] = 1 if c == 'N' else -1
        elif c == 'B' or c == 'b':
            board_array[BISHOP_PLANE, r, f] = 1 if c == 'B' else -1
        elif c == 'Q' or c == 'q':
            board_array[QUEEN_PLANE, r, f] = 1 if c == 'Q' else -1
        elif c == 'K' or c == 'k':
            board_array[KING_PLANE, r, f] = 1 if c == 'K' else -1
        elif c == '/':
            assert f == 8
            f = -1
            r += 1
        elif c == ' ':
            break
        else: # a number indicating 1 or more blank squares
            f += int(c) - 1
    # TODO: add parsing for castling availability
    if rotate:
        for p in range(6): # all planes
            for i in range(4): # first half of the ranks
                for j in range(8): # all files
                    temp = board_array[p, i, j]
                    board_array[p, i, j] = -board_array[p, 7-i, 7-j]
                    board_array[p, 7-i, 7-j] = -temp
    return board_array

# %% fix fens (need to play first move)
input_fens = puzzle_data["FEN"]
correct_moves = puzzle_data["MOVES"]
df = pd. DataFrame([1,FEATURES])
input_fens_corrected = []
ii = 0
for fen in input_fens:
    board = chess.Board(fen)
    initial_move = correct_moves[ii][0:4]
    move = chess.Move.from_uci(initial_move)
    board.push(move)
    fen = board.fen()
    input_fens_corrected.append(fen)
    ii += 1
    if ii % 10000 == 0:
        print(ii)
input_fens = input_fens_corrected

# %% calculate features
# need to re-write this
i = 0
input_list = []
for fen in input_fens:
    input_features = featurize_board(fen)
    # input_dict = dict(enumerate(input_features.flatten(), 1))
    input_list.append(input_features)
    #input_series = pd.Series(input_features.ravel())
    #df = df.append(input_series, ignore_index=True)
    i += 1
    if i % 100000 == 0:
        print(i)

# df_final = pd.DataFrame.from_dict(input_list)

# %% write to csv
compression_opts = dict(method='zip',
                        archive_name='featuresCNN.csv')
df_final.to_csv('features.zip',index=False,compression=compression_opts)