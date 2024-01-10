from datetime import datetime

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from src.utils import app_logger


class Autoencoder:
    def __init__(self, input_dim, encoding_dim):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        app_logger.info("Creating Autoencoder model.")

    def create_callback(self):
        # define our early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.0001,
            patience=10,
            verbose=1,
            mode="min",
            restore_best_weights=True,
        )

        save_model = tf.keras.callbacks.ModelCheckpoint(
            filepath="data_generated/evaluation/autoencoder/autoencoder_best_weights.hdf5",
            save_best_only=True,
            monitor="val_loss",
            verbose=0,
            mode="min",
        )

        # callbacks argument only takes a list
        cb = [early_stop, save_model]

        return cb

    def build_encoder(self):
        encoder = Sequential()
        encoder.add(
            Dense(self.input_dim, activation="elu", input_shape=(self.input_dim,))
        )
        encoder.add(Dense(16, activation="elu"))
        encoder.add(Dense(8, activation="elu"))
        encoder.add(Dense(4, activation="elu"))
        encoder.add(Dense(self.encoding_dim, activation="elu"))
        return encoder

    def build_decoder(self):
        decoder = Sequential()
        decoder.add(Dense(4, activation="elu", input_shape=(self.encoding_dim,)))
        decoder.add(Dense(8, activation="elu"))
        decoder.add(Dense(16, activation="elu"))
        decoder.add(Dense(self.input_dim, activation="elu"))
        return decoder

    def compile(self):
        self.autoencoder = Sequential([self.encoder, self.decoder])
        self.autoencoder.compile(
            optimizer="adam", loss="mse", metrics=["acc"]
        )  # TODO: loss="binary_crossentropy" check this

    def train(self, X_train, X_val, epochs, batch_size):
        cb = self.create_callback()

        history = self.autoencoder.fit(
            X_train,
            X_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb,
            validation_data=(X_val, X_val),
        )

    def encode(self, x):
        return self.encoder.predict(x)

    def decode(self, encoded_data):
        return self.decoder.predict(encoded_data)
