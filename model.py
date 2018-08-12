from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from encoder import TOTAL_MOVES 

def create_model(input_shape, output_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), 
        activation='relu', input_shape=input_shape, data_format='channels_first'))
    model.add(Dropout(rate=0.6))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.6))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.6))
    model.add(Dense(output_shape[0], activation='softmax'))
    # model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    model = create_model((1, 10, 9), (TOTAL_MOVES, 1))
