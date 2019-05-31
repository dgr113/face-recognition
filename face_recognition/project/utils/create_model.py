# coding: utf-8

from typing import Tuple
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Convolution2D, ZeroPadding2D, Activation




def create_model_1(path: str, input_shape: Tuple[int, ...], n_classes: int) -> None:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape, data_format='channels_last'))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    model_json = model.to_json()
    with open(path, 'w', encoding='utf-8') as f:
        f.write(model_json)




def create_model_2(path: str, input_shape: Tuple[int, ...], n_classes: int) -> None:
    model = Sequential()

    model.add(Conv2D(32, 3, activation='relu', input_shape=input_shape, data_format='channels_last'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(12, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(600, activation='relu'))
    model.add(Dense(60, activation='relu'))

    model.add(Dense(n_classes, activation='softmax'))

    model_json = model.to_json()
    with open(path, 'w', encoding='utf-8') as f:
        f.write(model_json)





def main():
    create_model_1('../../_instance/config/model_config.json', (32, 32, 1), 2)





if __name__ == "__main__":
    main()
