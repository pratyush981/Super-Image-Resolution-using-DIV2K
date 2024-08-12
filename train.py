import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')

model = load_model('super_resolution_model.h5')
model.compile(optimizer=Adam(), loss='mean_squared_error')

history = model.fit(X_train, Y_train, epochs=10, batch_size=16, validation_split=0.1)
model.save('super_resolution_trained_model.h5')
