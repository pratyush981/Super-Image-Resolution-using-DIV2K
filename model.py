import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D

def create_model():
    model = Sequential()

    model.add(Conv2D(64, (9, 9), padding='same', activation='relu', input_shape=(128, 128, 3)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(3, (5, 5), padding='same', activation='sigmoid'))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    model = create_model()
    model.summary()
    model.save('super_resolution_model.h5')
