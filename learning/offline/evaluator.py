
# coding: utf-8

# In[12]:


import chess 
import chess.pgn
import chess.polyglot
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
import re
import pickle
import time
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf


# # Data Collection
# We'll first gather the data needed to train our neural network. Using LiChess' Database, we can collect 19.9 million real online chess games. The ratings of players on LiChess is approximately normally distributed with mean 1500, so we'll remove any games whereby one or more of the player's holds a GLICKO rating less than 2000 to ensure that we are only using data from *good* chess players.
# 
# The board is represented within each game as a FEN string, so we'll need to parse that into a neural network friendly format. There are a number of ways of representing the board, however, we're going to keep things relatively simple here and represent the board as a a Zobrist Hash of the original board. The reseult of the game is just encoded as 1 for a white win, 2 for a black win and 3 for a draw; these values are purely arbitary as we'll one-hot encode them prior to training a network.

# In[32]:


pgn = open('data/2018_08.pgn')


def get_notation():
    result = {}
    notation = list('eprnbqkPRNBQK')
    i = 0
    for piece in notation:
        result[piece] = i
        i += 1
    return result


class Training:
    def __init__(self):
        self.X = []
        self.y = []


def parse_result(result):
    if result == '1-0':
        return 0
    elif result == '1-0':
        return 1
    else:
        return 2


def fen_expander(board_string, to_move_dict, notation_dict):
    fen_string = board_string.split(' ')
    # Get the next player to move
    to_move = to_move_dict[fen_string[1]]
    board_string = fen_string[0]
    numbers = list(set(re.findall('\d', board_string)))
    for number in numbers:
        board_string = board_string.replace(str(number), 'e' * int(number))
    board_string = board_string.replace('/', '')
    final_string = [notation_dict[x] for x in list(board_string)]
    final_string.append(to_move)
    return final_string


def get_X_y(current_game, training_obj):
    # Extract the current board
    board = current_game.board()
    # Parse and store the result
    result = parse_result(current_game.headers['Result'])
    training_obj.y.append([result] * len(list(current_game.main_line())))
    # Get each board state of the game
    for move in current_game.main_line():
        board.push(move)
        expanded_fen = fen_expander(board.fen(), to_move, notation_dict)
        training_obj.X.append(expanded_fen)

def nested_counter(item):
    if type(item) == list:
        return sum(nested_counter(subitem) for subitem in item)
    else:
        return 1
        
to_move = {'w': 0, 'b': 1}
notation_dict = get_notation()

i = 0
max_games = 1000000
max_training = 1000000
board_states = []
results = Training()

start = time.time()
while True and i < max_games and len(results.X)<max_training:
    try:
        game = chess.pgn.read_game(pgn)
        if int(game.headers['BlackElo']) > 1500 and int(game.headers['WhiteElo']) > 1500:
            get_X_y(game, results)
    except (KeyboardInterrupt, SystemExit):
        raise
    i += 1

with open('data/training_sma.obj', 'wb') as outfile:
    pickle.dump(results, outfile)


# # Defining the model
# 
# We now have an object containing features are results so it's time to train the network. Although DeepMind use convolutional neural nets, we're just going to use a standard neural network with dropout layers to prevent overfitting.

# In[33]:


# Load object
with open(r"data/training_sma.obj", "rb") as input_file:
    data_obj = pickle.load(input_file)

print('Feature Length: {}'.format(len(data_obj.X)))
print('Label Length: {}'.format(len(data_obj.y[0])))

# One-hot encode labels
labels = [label for item in data_obj.y for label in item]
labels_enc = to_categorical(np.array(labels))

# Put features into matrix form
features = np.array(data_obj.X)

# Split into test/train sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_enc, test_size=0.25, random_state=123)


# With the models architecture defined, we just need to define the optimiser and loss function before we can train the model.

# In[ ]:


def build_cnn(input_dim, num_classes):
    model = Sequential()
    model.add(Conv1D(32, strides=2, activation='relu', kernel_size=3, kernel_initializer='he_normal', input_shape=(None, input_dim)))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.25))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def build_ann(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(units = 120, activation = "relu", input_dim = input_dim, kernel_initializer = "normal"))
    model.add(Dense(units = 50, activation = "relu", kernel_initializer='normal'))
    model.add(Dense(units = 150, activation = "relu", input_dim = input_dim, kernel_initializer = "normal"))
    model.add(Dense(units = 80, activation = "relu", kernel_initializer='normal'))
    model.add(Dense(units = num_classes, activation = "softmax", kernel_initializer='normal'))
    model.compile(loss = keras.losses.categorical_crossentropy, 
                 optimizer=keras.optimizers.Adam(),
                 metrics=['accuracy'])
    return model

ann_model = build_ann(X_train.shape[1], y_train.shape[1])
ann_model.fit(X_train, y_train, epochs = 150, batch_size=32)
accuracy = ann_model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: {}'.format(accuracy))
# cnn_model = build_cnn(X_train.shape[1], y_train.shape[1])
# cnn_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)


# In[173]:


torch_bool = False
if torch_bool:
    # Define Network
    class Evaluator(nn.Module):
        def __init__(self, class_count=3, widths = [123]):
            super(Evaluator, self).__init__()
            self.layer1 = nn.Sequential(
                # nn.Conv1d(65, widths[0], kernel_size=5, stride=1, padding=2),
                nn.Linear(65, widths[0]), 
                nn.ReLU(inplace=True))
                # nn.MaxPool1d(kernel_size=2, stride=2))
            self.layer2 = nn.Sequential(
                #nn.Conv1d(widths[0], 32, kernel_size=5, stride=1, padding=2),
                nn.Linear(widths[0], 3),
                nn.ReLU(inplace=True))
                #nn.MaxPool1d(kernel_size=2, stride=2))
            self.fc = nn.Softmax(dim=1)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            return out

    # Detect device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initiate model
    model = Evaluator().to(device)    

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epoch_count = 500

    # Train the model
    total_step = X_train.shape[0]
    for epoch in range(epoch_count):
        X = torch.autograd.Variable(torch.cuda.FloatTensor(X_train), requires_grad=True)
        y = torch.autograd.Variable(torch.cuda.LongTensor(y_train)).view(-1)
        # Forward pass
        outputs = model(X)
        loss = criterion(X, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    #get prediction
    X = torch.autograd.Variable(torch.cuda.FloatTensor(X_test))
    y = torch.autograd.Variable(torch.cuda.LongTensor(y_test))
    out = model(X)
    _, predicted = torch.max(out.data, 1)

    #get accuration
    print('Accuracy of the network %d %%' % (100 * torch.sum(y==predicted) / y.shape[0]))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

