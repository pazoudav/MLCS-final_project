# LSTM for sequence classification in the IMDB dataset
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers import Conv1D, Conv2D, Reshape, Activation, Add
from keras.optimizers import SGD, Adam
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.utils import to_categorical
from joblib import dump, load
from basicClassifiers import eval_classifier

from labelling import *
from encoder import *
from split_data import get_train_test

from keras.layers import Layer

class Residual(Layer):
    def __init__(self, channels_in,kernel,**kwargs):
        super(Residual, self).__init__(**kwargs)
        self.channels_in = channels_in/2
        self.channel_out = channels_in
        self.kernel = kernel
        self.conv0 = Conv1D( self.channel_out,
                             1,
                             padding="same",
                             trainable=False)
        self.conv1 = Conv1D( self.channel_out,
                             self.kernel,
                             padding="same")
        self.conv2 = Conv1D( self.channel_out,
                             self.kernel,
                             padding="same")
        self.norm1 = BatchNormalization()
        self.norm2 = BatchNormalization()
        self.actv1 = Activation("linear", trainable=False)
        self.actv2 = Activation("relu")
        self.actv3 = Activation("relu")
        
        

    def call(self, x):
        first_layer =   self.conv0(x)
        
        x =             self.conv1(first_layer)
        x =             self.norm1(x)
        x =             self.actv2(x)

        x =             self.conv2(x)
        x =             self.norm2(x)
        x =             self.actv3(x)
        
        x =             Dropout(0.5)(x)
        residual =      Add()([x, first_layer])
        return residual

    def compute_output_shape(self, input_shape):
        return input_shape*2


def CNN():
    length = 4096
    enc = OneHotEncoder(max_len=length).load() # .make(split='basic_split-small').save() # 
    print('enc loaded')
    lab = SubsetLabeller().load()
    print('lab loaded')
    label_count = len(lab.subsets)

    X_train, X_test, y_train, y_test_ = get_train_test('basic_split-small')
    nb_epochs = 100
    batch_sz = 32

    print('data loaded')
    print('encoding train data')
    X_train = enc.encode(X_train[:]) 
    y_train = lab.label(y_train[:])   
    print('encoding test data')
    X_test = enc.encode(X_test[:]) 
    y_test_ = lab.label(y_test_[:]) 

    y_train = to_categorical(y_train, num_classes=label_count)
    y_test  = to_categorical(y_test_,  num_classes=label_count)
    

    model = Sequential()
    n = 2
    model.add(Conv2D(1, (n, enc.call_count), activation='relu', kernel_initializer='he_uniform',input_shape=(enc.max_len, enc.call_count,1)))
    model.add(Reshape((length+1-n, 1)))

    model.add(Residual(2,4))
    model.add(MaxPooling1D(2))
    model.add(Residual(4,8))
    model.add(MaxPooling1D(2))
    model.add(Residual(8,16))
    model.add(MaxPooling1D(2))
    model.add(Residual(16,32))
    model.add(MaxPooling1D(2))
    model.add(Residual(32,64))
    model.add(MaxPooling1D(2))
    model.add(Residual(64,128))
    model.add(MaxPooling1D(2))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
   
    model.add(Dense(label_count, activation='softmax'))
    # opt = SGD(learning_rate=0.005, momentum=0.9)
    opt = Adam()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())    
    model.fit(X_train, y_train, epochs=nb_epochs, batch_size=batch_sz) 
    dump(model, 'model.clf')
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    eval_classifier(model, X_test, y_test_, conf_mat=True, y_pred=y_pred)
    

if __name__ == '__main__':
    CNN()

    