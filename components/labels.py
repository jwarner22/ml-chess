import pandas as pd
import numpy as np

# %%
# import csv
# puzzle_data = pd.read_csv("./data/ml-data-test.csv")

puzzle_data = pd.read_csv('./moves_raw_expanded.zip')

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


# task is to:
# 1. identify categories associated with each fen
# 2. assign value of 1 to each matching index
# i.e. [0,1,0,0] => [pin,verLong,xRayAttack,zugzwang]
# 3. create df of vectorize training labels
# 4. save df to csv for future use

# %%
output_vector = np.zeros([1, len(puzzle_categories)])
def check_motifs(row):
    #label_list = []
    label_vector = np.zeros((1, len(puzzle_categories)))
    for item in row:
        i=0
        label_vector_sparse = []
        for index in puzzle_categories:
            if index == item:
                label_vector_sparse.append(1)
            else:
                label_vector_sparse.append(0)
            i += 1
        # label_dict = dict(enumerate(label_vector.flatten(), 1))
        #label_list.append(label_dict)
        label_vector = np.add(label_vector,label_vector_sparse)
    return label_vector

# test_check_motifs = check_motifs(puzzle_categories)
# print(test)

label_series = []
for index,row in puzzle_data.iterrows():
    label_series.append(check_motifs(row).flatten())
    if index % 100000 == 0:
        print(index)
    if index >= 4000000:
        break

# %% convert list to dataframe
training_labels = pd.DataFrame(np.vstack(label_series))

# %% save to csv
compression_opts = dict(method='zip',
                        archive_name='labels_big.csv')
training_labels.to_csv('labels_big.zip',index=False,compression=compression_opts)



