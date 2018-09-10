from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, ZeroPadding2D, Activation, BatchNormalization

from encoder import TOTAL_MOVES

STATE_SHAPE = (1, 10, 9)


def create_model():
    state = Input(shape=STATE_SHAPE, name='state')
    state_encoding = ZeroPadding2D((3, 3), input_shape=STATE_SHAPE, data_format='channels_first')(state)
    state_encoding = Conv2D(32, kernel_size=(3, 3),
                            # input_shape=STATE_SHAPE,
                            data_format='channels_first')(state_encoding)
    state_encoding = BatchNormalization(axis=1)(state_encoding)
    state_encoding = Activation('relu')(state_encoding)
    state_encoding = Dropout(rate=0.6)(state_encoding)
    state_encoding = ZeroPadding2D((2, 2), data_format='channels_first')(state_encoding)
    state_encoding = Conv2D(64, (3, 3))(state_encoding)
    state_encoding = BatchNormalization(axis=1)(state_encoding)
    state_encoding = Activation('relu')(state_encoding)
    state_encoding = MaxPooling2D(pool_size=(2, 2))(state_encoding)
    state_encoding = Dropout(rate=0.6)(state_encoding)
    state_encoding = Flatten()(state_encoding)
    state_encoding = Dense(1024, activation='relu')(state_encoding)
    state_encoding = Dropout(rate=0.6)(state_encoding)
    state_encoding = Dense(256, activation='relu')(state_encoding)
    state_encoding = Dropout(rate=0.6)(state_encoding)

    policy_hidden_layer = Dense(128, activation='relu')(state_encoding)
    policy_output = Dense(TOTAL_MOVES, activation='softmax')(
        policy_hidden_layer)

    value_hidden_layer = Dense(64, activation='relu')(state_encoding)
    value_output = Dense(1, activation='tanh')(value_hidden_layer)

    model = Model(inputs=[state], outputs=[policy_output, value_output])
    
    model.compile(optimizer='sgd', loss=['categorical_crossentropy', 'mse'], loss_weights=[1.0, 0.5])
    return model


if __name__ == '__main__':
    model = create_model()
    model.summary()
