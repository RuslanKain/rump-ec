import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Flatten, RepeatVector, Bidirectional
from tensorflow.keras.optimizers import Adam
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.model_selection import KFold


class Encoder_Decoder:

    def __init__(self):
        pass

    # splits dataset for train - test
    def split_dataset(self, data):
        train, test =  data[0:int(len(data)*0.7)], data[int(len(data)*0.7):]

        return train, test

    # creates sequences
    def split_sequence(self, seq, steps, out):
        X, Y = list(), list()
        for i in range(len(seq)):
            end = i + steps
            outi = end + out
            if outi > len(seq)-1:
                break
            seqx, seqy = seq[i:end], seq[end:outi]
            X.append(seqx)
            Y.append(seqy)
        return np.array(X), np.array(Y)

    def split_test_train_seq(self, train, test, in_steps, out, features):

        try:
            if 'state' in train.columns:
                train = train.drop(columns=['state'])
                test = test.drop(columns=['state'])
        except:
            pass
        X_train, Y_train = self.split_sequence(train, in_steps, out)
        X_test, Y_test = self.split_sequence(test, in_steps, out)
        X_train = X_train.reshape((Y_train.shape[0], X_train.shape[1], features))
        Y_train = Y_train.reshape((Y_train.shape[0], Y_train.shape[1], features))
        X_test = X_test.reshape((Y_test.shape[0], X_test.shape[1], features))
        Y_test = Y_test.reshape((Y_test.shape[0], Y_test.shape[1], features))
  
        return X_train, Y_train, X_test, Y_test

    def bid_lstm_model(self, in_steps, out, features, learning_rate, multiplier=1):

        """Model Architecture:
        1. Bidirectional LSTM layer with 256 units with relu activation
        2. LSTM layer with variable units in multiples of 128 
        3. RepeatVector layer with out steps
        4. LSTM layer with variable units in multiples of 256 
        5. Bidirectional LSTM layer with variable units in multiples of 128 
        6. LSTM layer with variable units in multiples of 64 
        7. TimeDistributed Dense layer with features units

        """

        model = Sequential()
        model.add(Bidirectional(LSTM(256*multiplier, activation='relu', input_shape=(in_steps, features),return_sequences=True), name='bidirectional_lstm1'))
        model.add(LSTM(128*multiplier, activation='relu', name='lstm1'))
        model.add(RepeatVector(out, name='repeat_vector'))
        model.add(LSTM(256*multiplier, activation='relu', return_sequences=True, name='lstm2'))
        model.add(Bidirectional(LSTM(128*multiplier, activation='relu',return_sequences=True), name='bidirectional_lstm2'))
        model.add(TimeDistributed(Dense(64*multiplier, activation='relu'), name='time_distributed_dense1'))
        model.add(TimeDistributed(Dense(features), name='time_distributed_dense2'))
        adam = Adam(learning_rate=learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return model

    def cnn_lstm_model(self, in_steps, out, features, learning_rate):

        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=(in_steps,features), name='conv1d_1'))
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu', name='conv1d_2'))
        model.add(MaxPooling1D(pool_size=2, name='max_pooling1d_1'))
        model.add(Flatten(name='flatten_1'))
        model.add(RepeatVector(out, name='repeat_vector_1'))
        model.add(LSTM(200, activation='relu', return_sequences=True, name='lstm_1'))
        model.add(TimeDistributed(Dense(100, activation='relu'), name='time_distributed_1'))
        model.add(TimeDistributed(Dense(features), name='time_distributed_2'))
        adam = Adam(learning_rate=learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return model

    def bid_model(self, in_steps, out, features, learning_rate):

        model = Sequential()
        model.add(Bidirectional(LSTM(180, activation='relu', input_shape=(in_steps, features))))
        model.add(RepeatVector(out))
        model.add(Bidirectional(LSTM(180, activation='relu',return_sequences=True)))
        model.add(TimeDistributed(Dense(64, activation='relu')))
        model.add(TimeDistributed(Dense(features)))
        adam = Adam(learning_rate=learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return model

    # produces time-step specific predictions
    def column(matrix, i):
            rows = []
            for row in matrix: 
                for r in row:
                    rows.append(r[i])
            return rows

    def cross_validate(self, model, X, y, n_splits=5, h=1):
        kf = KFold(n_splits=n_splits)
        rmse_scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            predictions = []
            for i in range(0, len(X_test), h):
                prediction = model.predict(X_test[i:i+h])
                predictions.append(prediction)
            a_predictions = np.concatenate(predictions, axis=0)
            rmse = mean_squared_error(a_predictions, y_test, squared=False)
            rmse_scores.append(rmse)
        return np.mean(rmse_scores)
