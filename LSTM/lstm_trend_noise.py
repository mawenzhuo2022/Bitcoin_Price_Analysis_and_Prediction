import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras import models, layers, optimizers, callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from itertools import product


class ImprovedTrendPredictor:
    def __init__(self, price_column, trend_threshold=0.01):
        self.price_column = price_column
        self.trend_threshold = trend_threshold
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.history = None
        self.classes_ = None

    def calculate_trends(self, prices, volumes):
        price_changes = np.diff(prices) / prices[:-1]
        volume_changes = np.diff(volumes) / volumes[:-1]

        momentum = price_changes * np.sign(volume_changes)
        window = 2
        smoothed_changes = np.convolve(momentum, np.ones(window) / window, mode='valid')

        trends = []
        trends.extend(['sideways'] * (len(momentum) - len(smoothed_changes) + 1))

        for change in smoothed_changes:
            if abs(change) < self.trend_threshold:
                trends.append('sideways')
            elif change > 0:
                trends.append('up')
            else:
                trends.append('down')

        self.classes_ = np.unique(trends)
        return trends

    def prepare_data(self, data, lookback=3):
        df = data.copy()

        df['returns'] = df[self.price_column].pct_change()
        df['log_returns'] = np.log(df[self.price_column] / df[self.price_column].shift(1))
        df['volatility'] = df['returns'].rolling(window=lookback).std()

        df['volume_change'] = df['Volume'].pct_change()
        df['volume_ma'] = df['Volume'].rolling(window=lookback).mean()
        df['relative_volume'] = df['Volume'] / df['volume_ma']

        df['rsi'] = self._calculate_rsi(df[self.price_column], periods=lookback)
        df['macd'], df['macd_signal'] = self._calculate_macd(df[self.price_column])

        df['price_range'] = (df['High'] - df['Low']) / df[self.price_column]
        df['gap'] = df['Open'] - df[self.price_column].shift(1)

        df = df.fillna(method='bfill')

        feature_columns = ['returns', 'log_returns', 'volatility',
                           'volume_change', 'relative_volume',
                           'rsi', 'macd', 'macd_signal',
                           'price_range', 'gap']

        scaled_features = self.scaler.fit_transform(df[feature_columns])

        trends = self.calculate_trends(df[self.price_column].values, df['Volume'].values)
        encoded_trends = self.label_encoder.fit_transform(trends)

        X, y = [], []
        for i in range(lookback, len(scaled_features)):
            X.append(scaled_features[i - lookback:i])
            y.append(encoded_trends[i])

        return np.array(X), np.array(y)

    def _calculate_rsi(self, prices, periods=14):
        deltas = np.diff(prices)
        gain = (deltas >= 0).astype(float) * deltas
        loss = (deltas < 0).astype(float) * (-deltas)

        avg_gain = np.concatenate(([np.nan], np.convolve(gain, np.ones(periods) / periods, mode='valid')))
        avg_loss = np.concatenate(([np.nan], np.convolve(loss, np.ones(periods) / periods, mode='valid')))

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return np.concatenate(([np.nan] * (len(prices) - len(rsi)), rsi))

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def create_model(self, input_shape, num_classes, lstm_units=[128, 64], dropout_rates=[0.3, 0.2],
                     learning_rate=0.001):
        model = models.Sequential([
            layers.LSTM(lstm_units[0], return_sequences=True, input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rates[0]),
            layers.LSTM(lstm_units[1]),
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(dropout_rates[1]),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train, y_train, epochs=50, batch_size=64, validation_split=0.2,
              lstm_units=[128, 64], dropout_rates=[0.3, 0.2], learning_rate=0.001):
        num_classes = len(self.classes_)

        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))

        self.model = self.create_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            num_classes=num_classes,
            lstm_units=lstm_units,
            dropout_rates=dropout_rates,
            learning_rate=learning_rate
        )

        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-6
            )
        ]

        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks_list,
            class_weight=class_weight_dict,
            verbose=1
        )
        return self.history

    def grid_search(self, X_train, y_train, X_val, y_val, param_grid=None):
        if param_grid is None:
            param_grid = {
                'lstm_units': [[64, 32], [128, 64], [256, 128]],
                'dropout_rates': [[0.2, 0.1], [0.3, 0.2], [0.4, 0.3]],
                'learning_rates': [0.001, 0.0005, 0.0001],
                'batch_sizes': [32, 64]
            }

        best_val_accuracy = float('-inf')
        best_params = None
        results = []

        total_combinations = (len(param_grid['lstm_units']) *
                              len(param_grid['dropout_rates']) *
                              len(param_grid['learning_rates']) *
                              len(param_grid['batch_sizes']))

        print(f"\nPerforming Grid Search ({total_combinations} combinations)...")

        for params in product(param_grid['lstm_units'],
                              param_grid['dropout_rates'],
                              param_grid['learning_rates'],
                              param_grid['batch_sizes']):
            lstm_units, dropout_rates, lr, batch_size = params

            model = self.create_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                num_classes=len(self.classes_),
                lstm_units=lstm_units,
                dropout_rates=dropout_rates,
                learning_rate=lr
            )

            early_stopping = callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            )

            class_weights = class_weight.compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weight_dict = dict(enumerate(class_weights))

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=30,
                batch_size=batch_size,
                callbacks=[early_stopping],
                class_weight=class_weight_dict,
                verbose=0
            )

            val_accuracy = max(history.history['val_accuracy'])
            results.append({
                'params': {
                    'lstm_units': lstm_units,
                    'dropout_rates': dropout_rates,
                    'learning_rate': lr,
                    'batch_size': batch_size
                },
                'val_accuracy': val_accuracy
            })

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_params = results[-1]['params']

            print(f"Tested configuration: {results[-1]['params']}")
            print(f"Validation accuracy: {val_accuracy:.4f}")
            print("-" * 50)

        results.sort(key=lambda x: x['val_accuracy'], reverse=True)
        print("\nTop 3 configurations:")
        for i, result in enumerate(results[:3], 1):
            print(f"\n{i}. Validation Accuracy: {result['val_accuracy']:.4f}")
            print("Parameters:", result['params'])

        return best_params, results

    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return np.argmax(predictions, axis=1)

    def plot_results(self, y_true, y_pred, dates):
        plt.figure(figsize=(15, 12))

        # Plot 1: Confusion Matrix with reordered classes
        plt.subplot(3, 1, 1)
        reordered_classes = ['up', 'sideways', 'down']  # New order

        # Reorder confusion matrix
        old_order = list(self.classes_)
        reorder_idx = [old_order.index(cls) for cls in reordered_classes]
        conf_matrix = confusion_matrix(y_true, y_pred)
        conf_matrix = conf_matrix[reorder_idx][:, reorder_idx]

        plt.imshow(conf_matrix, cmap='Blues')
        plt.title('Confusion Matrix')
        tick_marks = np.arange(len(reordered_classes))
        plt.xticks(tick_marks, reordered_classes, rotation=45)
        plt.yticks(tick_marks, reordered_classes)

        for i in range(len(reordered_classes)):
            for j in range(len(reordered_classes)):
                plt.text(j, i, conf_matrix[i, j],
                         ha="center", va="center")

        # Plot 2: Training History
        plt.subplot(3, 1, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot 3: Trend Comparison with reordered y-axis
        plt.subplot(3, 1, 3)

        # Create mapping for reordering
        reordered_classes = ['up', 'sideways', 'down']
        class_to_num = {cls: i for i, cls in enumerate(reordered_classes)}

        # Convert trends to numeric values based on new order
        actual_trends = [class_to_num[trend] for trend in self.label_encoder.inverse_transform(y_true)]
        pred_trends = [class_to_num[trend] for trend in self.label_encoder.inverse_transform(y_pred)]

        plt.plot(dates, actual_trends, label='Actual Trend', marker='o', markersize=4)
        plt.plot(dates, pred_trends, label='Predicted Trend', marker='x', markersize=4)
        plt.title('Trend Prediction Comparison')
        plt.xlabel('Date')
        plt.ylabel('Trend')
        plt.yticks(range(len(reordered_classes)), reordered_classes)
        plt.ylim(-0.5, 2.5)  # Set limits to ensure proper spacing
        plt.legend()
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

def load_and_preprocess_data(file_path):
    try:
        data = pd.read_csv(file_path)
        if 'Start' not in data.columns:
            raise ValueError("Data must contain 'Start' column for dates")
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column for prices")
        if 'Volume' not in data.columns:
            raise ValueError("Data must contain 'Volume' column")

        data['Start'] = pd.to_datetime(data['Start'])
        data = data.sort_values('Start')
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None


def train_with_grid_search(data, price_column='Close', lookback=3):
    predictor = ImprovedTrendPredictor(price_column)
    X, y = predictor.prepare_data(data, lookback=lookback)

    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    test_dates = data['Start'].values[train_size + val_size + lookback:]

    best_params, results = predictor.grid_search(X_train, y_train, X_val, y_val)

    print("\nTraining final model with best parameters...")
    predictor.train(
        X_train, y_train,
        epochs=50,
        batch_size=best_params['batch_size'],
        lstm_units=best_params['lstm_units'],
        dropout_rates=best_params['dropout_rates'],
        learning_rate=best_params['learning_rate'],
        validation_split=0.2
    )

    predictions = predictor.predict(X_test)

    print("\nFinal Model Performance:")
    print(classification_report(y_test, predictions, target_names=predictor.classes_))

    predictor.plot_results(y_test, predictions, test_dates)

    return predictor, best_params, results


if __name__ == "__main__":
    print("Bitcoin Price Trend Prediction")
    print("-" * 30)

    file_path = "../data/Bitcoin_Price/price_sentiment_data.csv"
    data = load_and_preprocess_data(file_path)

    if data is not None:
        lookbacks = [1, 3, 5, 7, 10, 15]
        best_overall = {
            'lookback': None,
            'params': None,
            'accuracy': float('-inf')
        }

        for lookback in lookbacks:
            print(f"\nTesting lookback period: {lookback}")
            print("-" * 30)

            predictor, best_params, results = train_with_grid_search(data, lookback=lookback)

            best_accuracy = max(result['val_accuracy'] for result in results)
            if best_accuracy > best_overall['accuracy']:
                best_overall['accuracy'] = best_accuracy
                best_overall['params'] = best_params
                best_overall['lookback'] = lookback

        print("\nBest Overall Configuration:")
        print(f"Lookback period: {best_overall['lookback']}")
        print(f"Best accuracy: {best_overall['accuracy']:.4f}")
        print("Parameters:", best_overall['params'])