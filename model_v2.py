from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Embedding, concatenate, \
    Conv2D, MaxPooling2D, Input

from encoder import TOTAL_MOVES

STATE_SHAPE = (10, 9, 1)
ACTION_EMBEDDING_SIZE = 20

def create_model():
    # cnn to encode chess state
    state = Input(shape=STATE_SHAPE, name='state')
    state_encoding = Conv2D(32, kernel_size=(3, 3), padding='same',
                            activation='relu', input_shape=STATE_SHAPE,
                            data_format='channels_first')(state)
    state_encoding = Dropout(rate=0.6)(state_encoding)
    state_encoding = Conv2D(64, (3, 3), activation='relu')(state_encoding)
    state_encoding = MaxPooling2D(pool_size=(2, 2))(state_encoding)
    state_encoding = Flatten()(state_encoding)

    # embedding layer to encode action
    action = Input(shape=(TOTAL_MOVES,), dtype='int32', name='action')
    action_embedding = Flatten()(Embedding(TOTAL_MOVES, ACTION_EMBEDDING_SIZE,
                              input_length=1)(action))

    # input for the model
    input_x = concatenate([state_encoding, action_embedding])
    input_x = Dense(128, activation='relu')(input_x)
    input_x = Dense(32, activation='relu')(input_x)

    # output
    # !!! note that the reward should be scaled into the range [0,1],
    # since we are using sigmoid activation for the output.
    output = Dense(1, activation="sigmoid", name="reward")(input_x)

    # model.summary()
    model = Model(inputs=[state, action], outputs=[output])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    model = create_model()
