import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.layers import BatchNormalization, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class LSTMForecaster:
    """
    LSTM-based forecaster for electricity consumption
    Supports both unidirectional and bidirectional architectures
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.history = None
        self.input_shape = None
        self.is_trained = False

        # LSTM parameters from config
        lstm_config = config['forecasting']['lstm']
        self.architecture = lstm_config.get('architecture', 'bidirectional')
        self.layers = lstm_config.get('layers', [64, 32, 16])
        self.dropout = lstm_config.get('dropout', [0.3, 0.3, 0.3])
        self.batch_size = lstm_config.get('batch_size', 32)
        self.epochs = lstm_config.get('epochs', 100)
        self.learning_rate = lstm_config.get('learning_rate', 0.001)
        self.lookback = lstm_config.get('lookback_window', 12)
        self.horizon = lstm_config.get('forecast_horizon', 12)

    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        self.input_shape = input_shape

        if self.architecture == 'bidirectional':
            model = self._build_bidirectional_lstm()
        elif self.architecture == 'encoder_decoder':
            model = self._build_encoder_decoder()
        else:
            model = self._build_standard_lstm()

        # Compile model
        def weighted_mae(y_true, y_pred):
            # Weighted MAE that penalizes errors on low consumption more heavily

            mae = tf.abs(y_true - y_pred)

            # Higher weight for low consumption values (under 400 kWh)
            weights = tf.where(y_true < 400, 2.0, 1.0)  # Double the error for low usage

            # Even higher weight for very low consumption (under 200 kWh)
            weights = tf.where(y_true < 200, 3.0, weights)

            return tf.reduce_mean(mae * weights)

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=weighted_mae,
            metrics=['mae']
        )

        logger.info(f"Built {self.architecture} LSTM model")
        logger.info(f"Input shape: {input_shape}")
        logger.info(f"Total parameters: {model.count_params():,}")

        self.model = model
        return model

    def _build_standard_lstm(self) -> tf.keras.Model:
        # Build standard LSTM (unidirectional)
        model = Sequential()

        # First LSTM layer
        model.add(LSTM(
            self.layers[0],
            return_sequences=len(self.layers) > 1,
            input_shape=self.input_shape
        ))
        model.add(Dropout(self.dropout[0]))
        model.add(BatchNormalization())

        # Hidden layers
        for i in range(1, len(self.layers) - 1):
            model.add(LSTM(
                self.layers[i],
                return_sequences=True
            ))
            model.add(Dropout(self.dropout[i]))
            model.add(BatchNormalization())

        # Last LSTM layer (no return sequences)
        if len(self.layers) > 1:
            model.add(LSTM(self.layers[-1]))
            model.add(Dropout(self.dropout[-1]))

        # Output layer
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.horizon))

        return model

    def _build_bidirectional_lstm(self) -> tf.keras.Model:
        # Build bidirectional LSTM
        model = Sequential()

        # First bidirectional layer
        model.add(Bidirectional(
            LSTM(self.layers[0], return_sequences=len(self.layers) > 1),
            input_shape=self.input_shape
        ))
        model.add(Dropout(self.dropout[0]))
        model.add(BatchNormalization())

        # Hidden bidirectional layers
        for i in range(1, len(self.layers) - 1):
            model.add(Bidirectional(
                LSTM(self.layers[i], return_sequences=True)
            ))
            model.add(Dropout(self.dropout[i]))
            model.add(BatchNormalization())

        # Last LSTM layer
        if len(self.layers) > 1:
            model.add(Bidirectional(LSTM(self.layers[-1])))
            model.add(Dropout(self.dropout[-1]))

        # Output layers
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.horizon))

        return model

    def _build_encoder_decoder(self) -> tf.keras.Model:
        # Build encoder-decoder architecture for seq2seq
        # Encoder
        encoder_inputs = Input(shape=self.input_shape)
        encoder = LSTM(self.layers[0], return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(self.horizon, self.input_shape[1]))
        decoder_lstm = LSTM(self.layers[0], return_sequences=True)
        decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)

        # Output
        decoder_dense = TimeDistributed(Dense(1))
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        return model

    def prepare_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_seq, y_seq = [], []

        for i in range(len(X) - self.lookback - self.horizon + 1):
            X_seq.append(X[i:i + self.lookback])
            y_seq.append(y[i + self.lookback:i + self.lookback + self.horizon])

        return np.array(X_seq), np.array(y_seq)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:

        if self.model is None:
            # Infer input shape from data
            if len(X_train.shape) == 2:
                # Need to create sequences
                X_train, y_train = self.prepare_sequences(X_train, y_train)
                if X_val is not None:
                    X_val, y_val = self.prepare_sequences(X_val, y_val)

            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        # Train
        validation_data = (X_val, y_val) if X_val is not None else None

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0
        )

        self.is_trained = True
        logger.info(f"Model trained for {len(self.history.history['loss'])} epochs")
        logger.info(f"Final loss: {self.history.history['loss'][-1]:.4f}")

        return self.history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Prepare sequences if needed
        if len(X.shape) == 2:
            # Create sequences
            X_seq = []
            for i in range(len(X) - self.lookback + 1):
                X_seq.append(X[i:i + self.lookback])
            X = np.array(X_seq)

        return self.model.predict(X, verbose=0)