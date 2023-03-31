import tensorflow as tf


def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D(size=(2, 2)),
        tf.keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D(size=(2, 2)),
        tf.keras.layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')
    ])

    return model


