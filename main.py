import pandas as pd
import numpy as np
import sys
import os
sys.path.append('../')
from src.preprocessing.text import *

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tf.keras.preprocessing.text import Tokenizer
# from tf.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Sequential, layers, regularizers

import ipdb

train_data = pd.read_csv('data/train.csv')
test_data  = pd.read_csv('data/test.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')

def clean_wrapper(text): 
	text = remove_url(text)
	text = remove_html(text)
	text = remove_emoji(text)
	text = remove_punctuation(text)
	return text


train_data['text'] = train_data['text'].apply(lambda x : clean_wrapper(x))
test_data['text'] = test_data['text'].apply(lambda x : clean_wrapper(x))


sent_data = train_data.text.values
labels_data = train_data.target.values
sent_submission = test_data.text.values


tokenizer = Tokenizer()
tokenizer.fit_on_texts(sent_data)

X_train = tokenizer.texts_to_sequences(sent_data)
# X_test = tokenizer.texts_to_sequences(sent_test)
X_submission = tokenizer.texts_to_sequences(sent_submission)

y_train = labels_data
# y_test = labels_test

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

# print(sent_train[2])
# print(X_train[2])

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
# X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
X_submission = pad_sequences(X_submission, padding='post', maxlen=maxlen)

# print(X_train[0, :])

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	x = range(1, len(acc) + 1)

	plt.figure(figsize=(12, 5))
	plt.subplot(1, 2, 1)
	plt.plot(x, acc, 'b', label='Training acc')
	plt.plot(x, val_acc, 'r', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.subplot(1, 2, 2)
	plt.plot(x, loss, 'b', label='Training loss')
	plt.plot(x, val_loss, 'r', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()


ROOT_DIR = "."
IMAGES_PATH = os.path.join(ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_name, tight_layout=True, fig_extension="png", resolution=300):
	path = os.path.join(IMAGES_PATH, fig_name + "." + fig_extension)
	print("Saving figure", fig_name)
	if tight_layout:
		plt.tight_layout()
	plt.savefig(path, format=fig_extension, dpi=resolution)


# ###################### Keras ######################

# embedding_dim = 100
# drop_out_prob = 0.3

# model = Sequential()
# model.add(layers.Embedding(input_dim=vocab_size,
#                            output_dim=embedding_dim,
#                            input_length=maxlen))
# model.add(layers.GlobalAveragePooling1D())
# # model.add(layers.Dense(30, activation='relu'))
# model.add(layers.Dense(64, activation='relu',
#                 kernel_regularizer=regularizers.l2(0.01),
#                 activity_regularizer=regularizers.l1(0.01)))
# model.add(layers.Dropout(drop_out_prob))
# model.add(layers.Dense(10, activation='relu'))
# model.add(layers.Dropout(drop_out_prob))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# model.summary()

# # # # checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

# history = model.fit(X_train, y_train,
#                     epochs=20,
#                     verbose=1,
#                     validation_split=0.1,
#                     batch_size=100)
# plot_history(history)
# save_fig("keras")


# ###################### Keras (RNN) ######################

# embedding_dim = 100
# drop_out_prob = 0.3

# model = Sequential()
# model.add(layers.Embedding(input_dim=vocab_size, 
#                            output_dim=embedding_dim, 
#                            input_length=maxlen))
# model.add(layers.LSTM(64, activation='tanh', dropout = 0.3, return_sequences=True))
# # model.add(layers.GlobalAveragePooling1D())
# model.add(layers.GlobalMaxPool1D())
# model.add(layers.Dense(64, activation='relu',
#                 kernel_regularizer=regularizers.l2(0.01),
#                 activity_regularizer=regularizers.l1(0.01)))
# model.add(layers.Dropout(drop_out_prob))
# model.add(layers.Dense(10, activation='relu'))
# model.add(layers.Dropout(drop_out_prob))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# model.summary()

# # # # checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

# history = model.fit(X_train, y_train,
#                     epochs=20,
#                     verbose=1,
#                     validation_split=0.2,
#                     batch_size=100)
# plot_history(history)
# save_fig("keras_RNN")

# ###################### GloVe ######################

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

embedding_dim = 100
embedding_matrix = create_embedding_matrix(
    'glove.twitter.27B.100d.txt',
    tokenizer.word_index, embedding_dim)

drop_out_prob = 0.3

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           weights = [embedding_matrix],
                           input_length=maxlen, 
                           trainable=True))
# model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalAveragePooling1D())
# model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(64, activation='relu',
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
model.add(layers.Dropout(drop_out_prob))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# ipdb.set_trace()
# print()

history = model.fit(X_train, y_train,
                    epochs=20,
                    verbose=1,
                    validation_split=0.1,
                    batch_size=1000, use_multiprocessing=True)
plot_history(history)
save_fig("GloVe")

# ###################### GloVe (RNN) ######################

# def create_embedding_matrix(filepath, word_index, embedding_dim):
#     vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
#     embedding_matrix = np.zeros((vocab_size, embedding_dim))

#     with open(filepath) as f:
#         for line in f:
#             word, *vector = line.split()
#             if word in word_index:
#                 idx = word_index[word] 
#                 embedding_matrix[idx] = np.array(
#                     vector, dtype=np.float32)[:embedding_dim]

#     return embedding_matrix

# embedding_dim = 100
# embedding_matrix = create_embedding_matrix(
#     'glove.twitter.27B.100d.txt',
#     tokenizer.word_index, embedding_dim)

# drop_out_prob = 0.3

# model = Sequential()
# model.add(layers.Embedding(input_dim=vocab_size,
#                            output_dim=embedding_dim,
#                            weights = [embedding_matrix],
#                            input_length=maxlen, 
#                            trainable=False))
# # model.add(layers.Conv1D(128, 5, activation='relu'))
# model.add(layers.LSTM(64, activation='tanh', dropout = 0.3, return_sequences=True))
# # model.add(layers.GlobalAveragePooling1D())
# model.add(layers.GlobalMaxPool1D())
# model.add(layers.Dense(64, activation='relu',
#                 kernel_regularizer=regularizers.l2(0.01),
#                 activity_regularizer=regularizers.l1(0.01)))
# model.add(layers.Dropout(drop_out_prob))
# model.add(layers.Dense(10, activation='relu'))
# model.add(layers.Dropout(drop_out_prob))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# model.summary()

# # ipdb.set_trace()
# # print()

# history = model.fit(X_train, y_train,
#                     epochs=20,
#                     verbose=1,
#                     validation_split=0.1,
#                     batch_size=100, use_multiprocessing=True)
# plot_history(history)
# save_fig("GloVe_RNN")


# ###################### Universial Sentence Enccoder ######################
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

module_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/5'
embed = hub.KerasLayer(module_url, trainable=True, name='USE_embedding')

# embed = hub.KerasLayer(module_url)

def build_model(embed):
	drop_out_prob = 0.3

	model = Sequential([
		layers.Input(shape=[], dtype=tf.string),
		embed,
		# layers.Dense(256, activation='relu'),
		# layers.BatchNormalization(),
		# layers.Dropout(0.5),
		# layers.Dense(128, activation='relu'),
		# layers.BatchNormalization(),
		# layers.Dropout(0.5),
		# layers.Dense(1, activation='sigmoid')
		
		# New
		layers.Dense(64, activation='relu',
		                kernel_regularizer=regularizers.l2(0.01),
		                activity_regularizer=regularizers.l1(0.01)),
		layers.BatchNormalization(),
		layers.Dropout(drop_out_prob),
		layers.Dense(10, activation='relu'),
		layers.BatchNormalization(),
		layers.Dropout(drop_out_prob),
		layers.Dense(1, activation='sigmoid')
	])
	# adam_optimizer = Adam(lr = 0.0005, beta_1 = 0.9, beta_2 = 0.999)
	# model.compile(optimizer = "adam", lr=0.0005, loss='binary_crossentropy', metrics=['accuracy'])
	model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])
	
	return model

model = build_model(embed)
model.summary()


checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

history = model.fit(sent_data, labels_data, validation_split=0.2, epochs=20, callbacks=[checkpoint], batch_size=32)

plot_history(history)
save_fig("Universal_Sentence_Encoder")



# ###################### Universial Sentence Enccoder (RNN) ######################
# import tensorflow as tf
# import tensorflow_hub as hub
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.optimizers import Adam

# module_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/5'
# embed = hub.KerasLayer(module_url, trainable=False, name='USE_embedding')

# # embed = hub.KerasLayer(module_url)

# def build_model(embed):
# 	drop_out_prob = 0.3

# 	model = Sequential([
# 		layers.Input(shape=[], dtype=tf.string),
# 		embed,
# 		# layers.Dense(256, activation='relu'),
# 		# layers.BatchNormalization(),
# 		# layers.Dropout(0.5),
# 		# layers.Dense(128, activation='relu'),
# 		# layers.BatchNormalization(),
# 		# layers.Dropout(0.5),
# 		# layers.Dense(1, activation='sigmoid')
		
# 		# New
# 		layers.Dense(64, activation='relu',
# 		                kernel_regularizer=regularizers.l2(0.01),
# 		                activity_regularizer=regularizers.l1(0.01)),
# 		layers.BatchNormalization(),
# 		layers.Dropout(drop_out_prob),
# 		layers.Dense(10, activation='relu'),
# 		layers.BatchNormalization(),
# 		layers.Dropout(drop_out_prob),
# 		layers.Dense(1, activation='sigmoid')
# 	])
# 	# adam_optimizer = Adam(lr = 0.0005, beta_1 = 0.9, beta_2 = 0.999)
# 	# model.compile(optimizer = "adam", lr=0.0005, loss='binary_crossentropy', metrics=['accuracy'])
# 	model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

	
# 	return model

# model = build_model(embed)
# model.summary()


# checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

# history = model.fit(sent_data, labels_data, validation_split=0.2, epochs=20, callbacks=[checkpoint], batch_size=32)

# plot_history(history)
# save_fig("Universal_Sentence_Encoder_RNN")


### Keras Built-in & GloVe ###
y_submission = model.predict(X_submission)
test_data['target'] = (y_submission > 0.5).astype(int)
submission = test_data[['id', 'target']]
submission.to_csv('submission.csv', index=False)

# ipdb.set_trace()
# print()


# ### Universal ###
# model.load_weights('model.h5')
# test_pred = model.predict(sent_submission)

# test_data['target'] = test_pred.round().astype(int)
# submission = test_data[['id', 'target']]
# submission.to_csv('submission.csv', index=False)