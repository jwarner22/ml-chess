import pandas as pd
import numpy as np
import chess

# %%
# import csv
# puzzle_data = pd.read_csv("./data/ml-data-test.csv")
fen_data = pd.read_csv('./fens_raw_expanded.zip')
# move_data = pd.read_csv('./moves_raw.zip')
# %% initialize features and labels

PAWN_PLANE = 0
ROOK_PLANE = 1
KNIGHT_PLANE = 2
BISHOP_PLANE = 3
QUEEN_PLANE = 4
KING_PLANE = 5

FEATURES = 6 * 8 * 8  # 6 unique piece types each on an 8x8 integer board; 1 for white, -1 for black
OUTPUTS = 43  # total number of puzzle categories

WHITE = True
BLACK = False

puzzle_categories_old = ["advancedPawn",
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

puzzle_categories = ["advancedPawn",
    "attackingF2F7",
    "attraction",
    "backRankMate",
    "bishopEndgame",
    "capturingDefender",
    "castling",
    "clearance",
    "coercion",
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
    "interference",
    "intermezzo",
    "kingsideAttack",
    "knightEndgame",
    "mate",
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
    "simplification",
    "skewer",
    "smotheredMate",
    "trappedPiece",
    "underPromotion",
    "xRayAttack"]


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
    return board_array.reshape((1, FEATURES))

# %% fix fens (need to play first move)
input_fens = np.array(fen_data)
correct_moves = np.array(move_data)
df = pd. DataFrame([1,FEATURES])
input_fens_corrected = []
ii = 0
for fen in input_fens:
    board = chess.Board(fen[0])
    initial_move = correct_moves[ii][0][0:4]
    move = chess.Move.from_uci(initial_move)
    board.push(move)
    fen = board.fen()
    rotate = board.turn
    input_fens_corrected.append(fen)
    ii += 1
    if ii % 100000 == 0:
        print(ii)
input_fens = input_fens_corrected

# %% fix expanded fens
input_fens = np.array(fen_data)
input_fens_listed = []
for fen in input_fens:
    input_fens_listed.append(fen[0])
input_fens = input_fens_listed
del input_fens_listed
del fen_data

# %% calculate features
# input_features = np.zeros((len(input_fens),FEATURES))
i = 0
input_list = []
for fen in input_fens:
    board = chess.Board(fen)
    rotate = board.turn
    input_features = featurize_board(fen,rotate)
    input_list.append(input_features.flatten())
    # input_dict = dict(enumerate(input_features.flatten(), 1))
    # input_list.append(input_dict)
    #input_series = pd.Series(input_features.ravel())
    #df = df.append(input_series, ignore_index=True)
    i += 1
    if i % 100000 == 0:
        print(i)
    if i > 4000000:
        break


# %% save to dataframe
# df_final = pd.DataFrame.from_dict(input_list)
df_final = pd.DataFrame(np.vstack(input_list))
# %% write to csv
compression_opts = dict(method='zip',
                        archive_name='features_big')
df_final.to_csv('features_big.zip',index=False,compression=compression_opts)