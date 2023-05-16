import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import palettable
from pathlib import Path
import glob
import os
import scipy.ndimage as ndimage
from scipy import signal
import statsmodels.api as smapi
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from time import time

#Specify your data path here
data_dir = Path('/rds/general/user/ys5320/home/Classification')

movements = ['HC_90','HC_45','Leg_extension_sit','Leg_presses_liftup', 'Half_squat']
#movements = ['HC_45','HC_90']

#Specify your data path here
data_dir = Path('/rds/general/user/ys5320/home/Classification')

movements = ['HC_90','HC_45','Leg_extension_sit','Leg_presses_liftup', 'Half_squat']
#movements = ['HC_45','HC_90']

n_move = []
file_path = list()
tcs = list()
target = list()
csv_all = []
for i in range(len(movements)):
    sub_tc = list()
    move = movements[i]
    os.chdir(Path(data_dir, move))
    extension = 'csv'
    csv_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    csv_all.extend(csv_filenames)
    n_move.append(len(csv_filenames))
    file_path.extend(csv_filenames)

    target.extend(i for j in range(n_move[i]))

    for k in range(n_move[i]):
        df = pd.read_csv(csv_filenames[k], index_col = False, header = None)
        
        values = np.array(df.values)
        values = values[:5700,1:]
        #print(values.shape)
        sub_tc.append(values)
    #print(df.shape)    
    tcs.extend(sub_tc)   
    
stacked_arr = np.stack([arr for arr in tcs])

from sklearn.model_selection import train_test_split
X = stacked_arr
y = np.array(target)
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify = y, test_size = 0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

import keras
from tensorflow.keras import layers
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

input_shape = X_train.shape[1:]
n_classes = y.max() + 1
model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),
    metrics=["sparse_categorical_accuracy"],
)
model.summary()

callbacks = [keras.callbacks.ModelCheckpoint(
        "transformer_best_model.h5", save_best_only=True, monitor="val_loss",
        save_freq = 60
                              ),
             keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
)

model.evaluate(X_test, y_test, verbose=1)

y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, y_pred.argmax(axis = 1))
print(matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred.argmax(axis = 1)))

print(history.history['loss'])
print(history.history['val_loss'])

print(history.history['sparse_categorical_accuracy'])
print(history.history['val_sparse_categorical_accuracy'])