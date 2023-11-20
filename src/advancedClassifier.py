# LSTM for sequence classification in the IMDB dataset
import tensorflow as tf
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Normalization, Dropout, BatchNormalization
from keras.preprocessing import sequence
from keras.layers import Conv1D
from keras.optimizers import SGD
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.utils import to_categorical

from labelling import *
from encoder import *
from split_data import get_train_test

#####################################################################
#
#   isn't done yet, dont know how far I'm gonna go with this one
#
##############################################################################







# fix random seed for reproducibility
tf.random.set_seed(7)
# load the dataset but only keep the top n words, zero the rest

# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
X_train, X_test, y_train, y_test = get_train_test('basic_split-small')

encoder = NGramEncoder(2, name_flag='-small').load() # make('basic_split-small').save()
X_train = encoder.encode(X_train)
X_test  = encoder.encode(X_test)
# top_words = len(encoder.API_calls_set)

# truncate and pad input sequences
# max_review_length = 1000
# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

labeller = SubsetLabeller().load()
label_count = len(labeller.subsets)
y_train = to_categorical(labeller.label(y_train), num_classes=label_count)
y_test  = to_categorical(labeller.label(y_test), num_classes=label_count)

# create the model
model = Sequential()


model.add(Dense(1000, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.5))
model.add(Normalization())

model.add(Dense(1000, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.5))
model.add(Normalization())

model.add(Dense(label_count, activation='softmax'))
opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# model.build(len(X_train[0]))
# print(model.summary())
model.fit(X_train, y_train, epochs=48, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))
