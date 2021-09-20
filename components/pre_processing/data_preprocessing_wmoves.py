# %% imports
import pandas as pd
import numpy as np

source_data = pd.read_csv('./data/lichess_db_puzzle.csv.bz2') # ,names=['ID','FEN','MOVES','RATING','RATINGDEVIATION','RATINGDEVIATIOM2','POP','NBPLAYS','THEMES','GAMEURL'])

# %% create dataframe w column names
df = pd.DataFrame(source_data,columns=['ID','FEN','MOVES','RATING','RATINGDEVIATION','RATINGDEVIATION2','POP','NBPLAYS','THEMES','GAMEURL'])

# %% parse themes
themes = source_data.iloc[:,7]
themes = np.array(themes)
theme_list = []
for theme in themes:
    split = theme.split()
    while len(split) < 12:
        split.append('nan')
    theme_list.append(split)

# %% create dataframe from list of lists
df_theme = pd.DataFrame(np.vstack(theme_list))

# %% write themes to csv for labelling
compression_opts = dict(method='zip',
                        archive_name='labels_raw.csv')
df_theme.to_csv('labels_raw.zip',index=False,compression=compression_opts)

# %% create raw feature data
fens = source_data.iloc[:,1]
fens = np.array(fens)

# %%
df_fens = pd.DataFrame(np.vstack(fens))


compression_opts = dict(method='zip',
                        archive_name='fens_raw.csv')
df_fens.to_csv('fens_raw.zip',index=False,compression=compression_opts)

# %% get moves
moves = source_data.iloc[:,2]
moves = np.array(moves)

# %%
df_moves = pd.DataFrame(np.vstack(moves))

compression_opts = dict(method='zip',
                        archive_name='moves_raw.csv')
df_moves.to_csv('moves_raw.zip',index=False,compression=compression_opts)

# %% import chess
import chess

# %% split moves
moves_split = []
for move in moves:
    move_split = move.split()
    moves_split.append(move_split)

# %% fix fens

fens_corrected = []
ii = 0
for fen in fens:
    board = chess.Board(fen)
    initial_move = moves[0][0:4]
    move = chess.Move.from_uci(initial_move)
    board.push(move)
    fen = board.fen()
    rotate = board.turn
    fens_corrected.append(fen)
    ii += 1
    if ii % 100000 == 0:
        print(ii)
fens = fens_corrected

# %% generate new fens and append with labels
fens = fens.tolist()
# %%
for i in range(0, len(moves_split)-1):
    moves = moves_split[i]
    fen = fens[i]
    board = chess.Board(fen)
    for index, move in enumerate(moves):
        next_move = chess.Move.from_uci(move)
        board.push(next_move)
        if index % 2 != 0:
            new_fen = board.fen()
            fens.append(new_fen)
            theme_list.append(theme_list[i])
    if i % 100000 == 0:
        print(i)

# %% save outputs
df_fens_expanded = pd.DataFrame(fens)
compression_opts = dict(method='zip',
                        archive_name='moves_raw.csv')
df_fens_expanded.to_csv('fens_raw_expanded.zip',index=False,compression=compression_opts)
# %%
df_themes_expanded = pd.DataFrame(theme_list)
df_themes_expanded.to_csv('moves_raw_expanded.zip',index=False,compression=compression_opts)