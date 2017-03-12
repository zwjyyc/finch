from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense
from keras.optimizers import Adam


def build_keras_lstm(input_size, time_steps, cell_size, output_size):
    model = Sequential()
    model.add(LSTM(
        input_dim = input_size,
        input_length = time_steps,
        output_dim = cell_size,
        unroll = True,
        dropout_W = 0.3,
    ))
    model.add(Dense(output_size))
    model.add(Activation('softmax'))
    adam = Adam(lr=1e-4)
    model.compile(optimizer = adam,
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'],
    )
    return model
# end function build_keras_lstm
