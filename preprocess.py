import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Function to load and preprocess images
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size, color_mode='rgb')
    img_array = img_to_array(img) / 255.0
    return img_array


def load_images_from_folder(folder, target_size=(128, 128)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = load_and_preprocess_image(img_path, target_size)
        if img is not None:
            images.append(img)
    return np.array(images)

if __name__ == "__main__":
    hr_folder = 'C:/Users/praty.PRATPC/PycharmProjects/SUPERImageRESOLUTION/.venv/DIV2K_train_HR'
    X_train = load_images_from_folder(hr_folder, target_size=(128, 128))
    Y_train = load_images_from_folder(hr_folder, target_size=(512, 512))

    np.save('X_train.npy', X_train)
    np.save('Y_train.npy', Y_train)
