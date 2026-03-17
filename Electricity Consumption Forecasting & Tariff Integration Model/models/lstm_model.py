"""
LSTM Model for Consumption Forecasting
Primary forecasting method for Component 3
"""

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
        """
        Initialize LSTM forecaster

        Args:
            config: Configuration dictionary
        """
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
        """
        Build LSTM model based on configuration

        Args:
            input_shape: (sequence_length, n_features)

        Returns:
            Compiled Keras model
        """
        self.input_shape = input_shape

        if self.architecture == 'bidirectional':
            model = self._build_bidirectional_lstm()
        elif self.architecture == 'encoder_decoder':
            model = self._build_encoder_decoder()
        else:
            model = self._build_standard_lstm()

        # Compile model
        def weighted_mae(y_true, y_pred):
            """
            Weighted MAE that penalizes errors on low consumption more heavily
            """
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
        """Build standard LSTM (unidirectional)"""
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
        """Build bidirectional LSTM"""
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
        """Build encoder-decoder architecture for seq2seq"""
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
        """
        Prepare sequences for LSTM training

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            X_seq, y_seq ready for LSTM
        """
        X_seq, y_seq = [], []

        for i in range(len(X) - self.lookback - self.horizon + 1):
            X_seq.append(X[i:i + self.lookback])
            y_seq.append(y[i + self.lookback:i + self.lookback + self.horizon])

        return np.array(X_seq), np.array(y_seq)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train LSTM model

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Training history
        """
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
        """
        Generate predictions

        Args:
            X: Input features

        Returns:
            Predictions
        """
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

    def forecast(self, user_data: Dict, features: Dict) -> Dict:
        """
        Generate forecast for a single user
        Now correctly aligns months based on user's last month
        """
        if not self.is_trained:
            logger.warning("Model not trained, cannot generate LSTM forecast")
            return None

        try:
            # Extract features for prediction
            X = features.get('lstm_input')
            if X is None:
                logger.error("No LSTM input features available")
                return None

            # Get user's last month to align forecast
            user_months = user_data.get('consumption_months', {})
            if user_months:
                last_user_month = max(user_months.keys())
                last_user_year = user_data.get('year', 2025)  # You might need to pass this
            else:
                last_user_month = 12
                last_user_year = 2025

            logger.info(f"User's last month: {last_user_month}/{last_user_year}")

            # Generate prediction
            y_pred = self.predict(X)

            # Get last prediction (most recent)
            if len(y_pred.shape) > 1:
                forecast_values = y_pred[-1]
            else:
                forecast_values = y_pred

            # Ensure we have 12 months
            if len(forecast_values) < 12:
                forecast_values = np.pad(forecast_values, (0, 12 - len(forecast_values)), 'edge')
            elif len(forecast_values) > 12:
                forecast_values = forecast_values[:12]

            # Calculate starting month (next month after user's last)
            start_month = (last_user_month % 12) + 1
            start_year = last_user_year + (1 if start_month == 1 else 0)

            # Create properly aligned monthly forecast
            monthly_forecast = {}
            monthly_details = []

            for i in range(12):
                current_month = ((start_month - 1 + i) % 12) + 1
                current_year = start_year + ((start_month - 1 + i) // 12)

                monthly_forecast[current_month] = float(forecast_values[i])

                monthly_details.append({
                    'month': current_month,
                    'month_name': self._get_month_name(current_month),
                    'year': current_year,
                    'consumption_kwh': round(forecast_values[i], 1),
                    'confidence': self._estimate_confidence(),
                    'season': self._get_sri_lanka_season(current_month)
                })

            # Calculate statistics
            values = list(monthly_forecast.values())
            annual_total = sum(values)
            annual_avg = annual_total / 12
            peak_month = max(range(1, 13), key=lambda m: monthly_forecast[m])

            forecast_stats = {
                'annual_total': annual_total,
                'annual_average': annual_avg,
                'peak_month': peak_month,
                'peak_consumption': monthly_forecast[peak_month],
                'overall_confidence': self._estimate_confidence(),
                'forecast_period': {
                    'start': f"{self._get_month_name(start_month)} {start_year}",
                    'end': f"{self._get_month_name(((start_month + 10) % 12) + 1)} {start_year + (1 if start_month > 1 else 0)}"
                }
            }

            # Simple uncertainty (20% for LSTM)
            uncertainty = {}
            for month in range(1, 13):
                val = monthly_forecast[month]
                uncertainty[month] = {
                    'lower_bound': val * 0.8,
                    'upper_bound': val * 1.2,
                    'std_dev': val * 0.2
                }

            result = {
                'forecast': {
                    'monthly_values': monthly_forecast,
                    'monthly_details': monthly_details,
                    'uncertainty_ranges': uncertainty,
                    'statistics': forecast_stats
                },
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'user_months_provided': len(user_months),
                    'user_last_month': f"{last_user_month}/{last_user_year}",
                    'forecast_start': f"{self._get_month_name(start_month)} {start_year}",
                    'forecast_method': 'lstm',
                    'architecture': self.architecture,
                    'confidence': self._estimate_confidence()
                }
            }

            return result

        except Exception as e:
            logger.error(f"Error in LSTM forecast: {e}")
            return None

    def _get_month_name(self, month_num: int) -> str:
        """Get month name from number"""
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        return month_names[month_num - 1] if 1 <= month_num <= 12 else f"Month {month_num}"

    def _estimate_confidence(self) -> float:
        """Estimate confidence based on training history"""
        if not self.history:
            return 0.5

        # Use validation loss if available, otherwise training loss
        if 'val_loss' in self.history.history:
            final_loss = self.history.history['val_loss'][-1]
        else:
            final_loss = self.history.history['loss'][-1]

        # Convert loss to confidence (lower loss = higher confidence)
        # Assuming typical loss range 0-1000 kWh
        confidence = max(0.3, min(0.95, 1.0 - (final_loss / 1000)))
        return float(confidence)

    def _create_forecast_result(self, monthly_forecast: Dict[int, float],
                               confidence: float,
                               user_data: Dict) -> Dict:
        """Create forecast result dictionary"""
        values = list(monthly_forecast.values())
        annual_total = sum(values)
        annual_avg = annual_total / 12
        peak_month = max(range(1, 13), key=lambda m: monthly_forecast[m])

        forecast_stats = {
            'annual_total': annual_total,
            'annual_average': annual_avg,
            'peak_month': peak_month,
            'peak_consumption': monthly_forecast[peak_month],
            'overall_confidence': confidence
        }

        # Simple uncertainty (20% for LSTM)
        uncertainty = {}
        for month in range(1, 13):
            val = monthly_forecast[month]
            uncertainty[month] = {
                'lower_bound': val * 0.8,
                'upper_bound': val * 1.2,
                'std_dev': val * 0.2
            }

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        monthly_details = []
        for month in range(1, 13):
            monthly_details.append({
                'month': month,
                'month_name': month_names[month - 1],
                'consumption_kwh': round(monthly_forecast[month], 1),
                'confidence': confidence,
                'season': self._get_sri_lanka_season(month)
            })

        result = {
            'forecast': {
                'monthly_values': monthly_forecast,
                'monthly_confidence': {m: confidence for m in range(1, 13)},
                'monthly_details': monthly_details,
                'uncertainty_ranges': uncertainty,
                'statistics': forecast_stats
            },
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'user_months_provided': len(user_data.get('consumption_months', {})),
                'forecast_method': 'lstm',
                'architecture': self.architecture,
                'confidence': confidence
            }
        }

        return result

    def _get_sri_lanka_season(self, month: int) -> str:
        """Get Sri Lankan season for month"""
        if month in [12, 1, 2]:
            return "NE Monsoon"
        elif month in [3, 4]:
            return "Dry Season"
        elif month in [5, 6, 7, 8, 9]:
            return "SW Monsoon"
        else:
            return "Dry Season"

    def save(self, filepath: Union[str, Path]):
        """
        Save model and scalers

        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        self.model.save(str(filepath))

        # Save scalers and metadata
        metadata = {
            'architecture': self.architecture,
            'layers': self.layers,
            'dropout': self.dropout,
            'lookback': self.lookback,
            'horizon': self.horizon,
            'input_shape': self.input_shape,
            'is_trained': self.is_trained
        }

        metadata_path = filepath.with_suffix('.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: Union[str, Path]):
        """
        Load saved model with proper custom objects
        """
        filepath = Path(filepath)
        print(f"🔍 Attempting to load model from: {filepath}")
        print(f"   File exists: {filepath.exists()}")

        # Define custom objects for loading
        from tensorflow.keras.losses import MeanAbsoluteError
        from tensorflow.keras.metrics import MeanAbsoluteError as MAEMetric
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

        custom_objects = {
            'mae': MAEMetric(),
            'loss': MeanAbsoluteError(),
            'MeanAbsoluteError': MeanAbsoluteError,
            'Adam': Adam,
            'LSTM': LSTM,
            'Dense': Dense,
            'Dropout': Dropout,
            'BatchNormalization': BatchNormalization
        }

        try:
            # Load with custom objects
            self.model = load_model(
                str(filepath),
                custom_objects=custom_objects,
                compile=True  # Keep compilation for metrics
            )
            print(f"✅ Model loaded successfully with custom objects")
        except Exception as e:
            print(f"⚠️ Error loading with custom objects: {e}")
            print("Attempting to load with default settings...")

            # Try loading with default settings
            self.model = load_model(str(filepath), compile=False)
            print(f"✅ Model loaded with compile=False")

            # Recompile the model
            self.model.compile(
                optimizer='adam',
                loss='mae',
                metrics=['mae']
            )
            print(f"✅ Model recompiled successfully")

        # Load metadata
        metadata_path = filepath.with_suffix('.pkl')
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            self.architecture = metadata.get('architecture', 'bidirectional')
            self.layers = metadata.get('layers', [64, 32, 16])
            self.dropout = metadata.get('dropout', [0.3, 0.3, 0.3])
            self.lookback = metadata.get('lookback', 12)
            self.horizon = metadata.get('horizon', 12)
            self.input_shape = metadata.get('input_shape')
            self.is_trained = metadata.get('is_trained', True)
            print(f"   Metadata loaded, is_trained: {self.is_trained}")
        else:
            self.is_trained = True
            print(f"   No metadata found, assuming model is trained")

        print(f"✅ Model ready for predictions")