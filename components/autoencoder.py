import pandas as pd
import numpy as np
import tensorflow as tf
# %%
from tensorflow import keras

# %%
features = pd.read_csv('./features_big.zip')

# %%
# labels_raw = pd.read_csv('./labels.zip')


# %% shuffle training data

idx = np.random.permutation(features.index)
features.reindex(idx, axis=0)
# labels_raw.reindex(idx,axis=0)

# %%
# select label to train
# label_transform = labels_raw.iloc[:,11] #next 7
# labels = label_transform


# %% create model

def create_model():
    # weights = np.load('layer1_weights.npy')
    model = keras.Sequential([
        keras.layers.Dense(384, input_shape=(384,)),
        keras.layers.Activation('relu'),
        keras.layers.Dense(256),
        keras.layers.Activation('relu'),
        keras.layers.Dense(384),
        keras.layers.Activation('sigmoid')]) #sigmoid
    compile_model(model)
    return model


def compile_model(model):
    # loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-1,
    #decay_steps=10000,
    #decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)
    loss = 'mean_squared_error' #'binary_crossentropy'
    metrics = [
        keras.metrics.Precision(name='precision'),
        #keras.metrics.TruePositives(name='tp'),
        #keras.metrics.FalsePositives(name='fp'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        #keras.metrics.TrueNegatives(name='tn'),
        #keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.AUC(name='auc')
    ]
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

def train_model(model):
    batch_size = 512
    epochs = 20
    save_filename = 'model_test'
    # labels = features for autoencoder
    model.fit(features,features,batch_size=batch_size,epochs=epochs,validation_split=0.1,use_multiprocessing=True)
    save_model(model,save_filename)


def save_model(model, filename):
    filename = 'model/' + filename
    model.save_weights(filename + '.h5')
    model_json = model.to_json()
    with open(filename + '.json', 'w') as json_file:
        json_file.write(model_json)

def eval_model(model):
    _,accuracy = model.evaluate(features,features)
    print('Accuracy: %.2f' % (accuracy*100))

# %% create model
model = create_model()

# %% train
train_model(model)

# %% save model
save_model(model,'../modelv3')

# %% evaluate
eval_model(model)
