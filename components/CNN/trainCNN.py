import pandas as pd
# import numpy as np
import tensorflow as tf
# %%
# import tensorflow.keras
# import tensorflow
# import CUDA
from tensorflow import keras

# %%
# import keras.layers as layers
# import keras.models as models
#from keras.layers import Activation, Sequential

#from keras.models import Dense

# Use this to prevent 100% GPU memory usage

tf.config.list_physical_devices('GPU')

# %%
# features = pd.read_csv('./featuresCNN.zip')
labels = pd.read_csv('../../labels.zip')

# %% get features

import pandas as pd
import numpy as np
import chess

# %%
# import csv
puzzle_data = pd.read_csv('../../data/ml-data-test.csv')# , names=['FEN','MOVES','RATING','RATINDDEVIATION','POP','NBPLAYS','GAMEURL'])# pd.read_csv("../data/ml-data-test.csv")

# %% initialize features and labels

PAWN_PLANE = 0
ROOK_PLANE = 1
KNIGHT_PLANE = 2
BISHOP_PLANE = 3
QUEEN_PLANE = 4
KING_PLANE = 5
COLOR_PLANE = 6

FEATURES = 7 * 8 * 8  # 6 unique piece types each on an 8x8 integer board; 1 for white, -1 for black
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

def featurize_board_wcolors(board_fen, rotate=False):
    board_array = np.zeros((7, 8, 8), dtype='int8')
    f = -1 # file (column)
    r = 0 # rank (row)
    end = False
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
            end = True
        else: # a number indicating 1 or more blank squares
            if c == 'w' or c == 'b' and end:
                board_array[COLOR_PLANE, r, f] = 1 if c == 'w' else -1
                break
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
    input_features = featurize_board_wcolors(fen)
    # input_dict = dict(enumerate(input_features.flatten(), 1))
    input_list.append(input_features)
    #input_series = pd.Series(input_features.ravel())
    #df = df.append(input_series, ignore_index=True)
    i += 1
    if i % 100000 == 0:
        print(i)

features = np.array(input_list)

# %% create model

def create_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=features.shape[1:], data_format='channels_first'))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Conv2D(32, (3, 3), padding='same'))
    model.add(keras.layers.Activation('relu'))
    keras.layers.BatchNormalization()
    model.add(keras.layers.MaxPooling2D(pool_size=(3,3)))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(512))
    model.add(keras.layers.Activation('relu'))
    keras.layers.BatchNormalization()
    model.add(keras.layers.Dense(57))
    model.add(keras.layers.Activation('sigmoid'))

    compile_model(model)
    return model


def compile_model(model):
    # loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, # adadelta
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    #was binary_crossentropy
def train_model(model):
    batch_size = 512
    epochs = 100
    save_filename = 'model_test'
    model.fit(features,labels,batch_size=batch_size,epochs=epochs,validation_split=0.1,use_multiprocessing=True)
    save_model(model,save_filename)


def save_model(model, filename):
    filename = 'model/' + filename
    model.save_weights(filename + '.h5')
    model_json = model.to_json()
    with open(filename + '.json', 'w') as json_file:
        json_file.write(model_json)

def eval_model(model):
    _,accuracy = model.evaluate(features,labels)
    print('Accuracy: %.2f' % (accuracy*100))

# %% create model
model = create_model()

# %% train
train_model(model)

# %% save model
save_model(model,'modelv1')

# %% evaluate
eval_model(model)