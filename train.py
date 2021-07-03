from time import time
import pandas as pd

import matplotlib

matplotlib.use('Agg')
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPool1D, Dense, Dropout

from util import make_w2v_embeddings
from util import split_and_zero_padding
from util import ManDist


class SentencesEngine:
    def __init__(self):
        TRAIN_CSV = './data/train.csv'
        self.train_df = pd.read_csv(TRAIN_CSV)
        self.train_df = pd.read_csv(TRAIN_CSV)
        for q in ['question1', 'question2']:
            self.train_df[q + '_n'] = self.train_df[q]
            embedding_dim = 300
        self.max_seq_length = 20
        use_w2v = True
        SE = SentencesEngine()
        self.train_df, self.embeddings = make_w2v_embeddings(SE.train_df, embedding_dim=embedding_dim, empty_w2v=not use_w2v)
        validation_size = int(len(self.train_df) * 0.1)
        training_size = len(self.train_df) - validation_size


    def train(self): 
        X = self.train_df[['question1_n', 'question2_n']]
        Y = self.train_df['is_duplicate']
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

        X_train = split_and_zero_padding(X_train, self.max_seq_length)
        X_validation = split_and_zero_padding(X_validation, self.max_seq_length)

        # Convert labels to their numpy representations
        Y_train = Y_train.values
        Y_validation = Y_validation.values

        gpus = 2
        batch_size = 1024 * gpus
        n_epoch = 50
        n_hidden = 50

        # Define the shared model
        x = Sequential()
        x.add(Embedding(len(self.embeddings), 300,
                        weights=[self.embeddings], input_shape=(self.max_seq_length,), trainable=False))

        x.add(LSTM(n_hidden))

        shared_model = x

        # The visible layer
        left_input = Input(shape=(self.max_seq_length,), dtype='int32')
        right_input = Input(shape=(self.max_seq_length,), dtype='int32')

        # Pack it all up into a Manhattan Distance model
        malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
        model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])
        print(model.summary())
        model = tf.keras.utils.multi_gpu_model(model, gpus=1)
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
        model.summary()
        shared_model.summary()

        # Start trainings
        training_start_time = time()
        self.trained_model = model.fit([X_train['left'], X_train['right']], Y_train,
                                   batch_size=batch_size, epochs=n_epoch,
                                   validation_data=([X_validation['left'], X_validation['right']], Y_validation))
        training_end_time = time()
        print("Training time finished.\n%d epochs in %12.2f" % (n_epoch,
                                                                training_end_time - training_start_time))

        model.save('./data/SiameseLSTM.h5')

    def plot_performance(self):
        plt.subplot(211)
        plt.plot(self.trained_model.history['acc'])
        plt.plot(self.trained_model.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot loss
        plt.subplot(212)
        plt.plot(self.trained_model.history['loss'])
        plt.plot(self.trained_model.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')

        plt.tight_layout(h_pad=1.0)
        plt.savefig('./data/history-graph.png')

        print(str(self.trained_model.history['val_acc'][-1])[:6] +
              "(max: " + str(max(self.trained_model.history['val_acc']))[:6] + ")")
        print("Done.")


