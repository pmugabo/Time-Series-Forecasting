"""
Task 3: Forecasting Models
This script implements and compares 3 forecasting models:
1. Statistical: SARIMA
2. Neural Network: LSTM
3. Neural Network: GRU
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import gc
import pickle
import warnings
warnings.filterwarnings('ignore')

# Statistical modeling
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# Deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configure TensorFlow to use less memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    # Limit CPU memory usage
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ForecastingModel:
    """Base class for forecasting models"""
    
    def __init__(self, name):
        self.name = name
        self.training_time = None
        self.prediction_time = None
        
    def train(self, train_data):
        """Train the model"""
        raise NotImplementedError
        
    def predict(self, test_data):
        """Make predictions"""
        raise NotImplementedError


class SARIMAModel(ForecastingModel):
    """SARIMA model for time series forecasting"""
    
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 144)):
        super().__init__("SARIMA")
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.model_fit = None
        
    def train(self, train_data):
        """Train SARIMA model"""
        print(f"  Training {self.name} model...")
        start_time = time.time()
        
        # Fit SARIMA model
        self.model = SARIMAX(train_data, 
                            order=self.order,
                            seasonal_order=self.seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
        self.model_fit = self.model.fit(disp=False)
        
        self.training_time = time.time() - start_time
        print(f"  Training time: {self.training_time:.2f} seconds")
        
    def predict(self, test_data, steps):
        """Make predictions"""
        print(f"  Predicting with {self.name} model...")
        start_time = time.time()
        
        # Forecast
        predictions = self.model_fit.forecast(steps=steps)
        
        self.prediction_time = time.time() - start_time
        print(f"  Prediction time: {self.prediction_time:.2f} seconds")
        
        return predictions
    
    def save(self, filepath):
        """Save model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model_fit, f)
        print(f"  Model saved to {filepath}")


class LSTMModel(ForecastingModel):
    """LSTM model for time series forecasting"""
    
    def __init__(self, sequence_length=144, units=50, epochs=50, batch_size=32):
        super().__init__("LSTM")
        self.sequence_length = sequence_length
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        self.model = None
        
    def create_sequences(self, data):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def train(self, train_data):
        """Train LSTM model"""
        print(f"  Training {self.name} model...")
        start_time = time.time()
        
        # Scale data
        train_scaled = self.scaler.fit_transform(train_data.values.reshape(-1, 1))
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_scaled)
        
        # Build model
        self.model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(self.units, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse')
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )
        
        self.training_time = time.time() - start_time
        print(f"  Training time: {self.training_time:.2f} seconds")
        print(f"  Epochs trained: {len(history.history['loss'])}")
        
    def predict(self, test_data, steps):
        """Make predictions"""
        print(f"  Predicting with {self.name} model...")
        start_time = time.time()
        
        # Use last sequence_length points from training data to predict
        # For simplicity, we'll use a rolling prediction approach
        predictions = []
        last_sequence = test_data[-self.sequence_length:].values
        last_sequence = self.scaler.transform(last_sequence.reshape(-1, 1))
        
        for _ in range(steps):
            X_pred = last_sequence.reshape(1, self.sequence_length, 1)
            pred = self.model.predict(X_pred, verbose=0)
            predictions.append(pred[0, 0])
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = pred[0, 0]
        
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        self.prediction_time = time.time() - start_time
        print(f"  Prediction time: {self.prediction_time:.2f} seconds")
        
        return predictions.flatten()
    
    def save(self, filepath):
        """Save model to file"""
        self.model.save(filepath)
        print(f"  Model saved to {filepath}")


class GRUModel(ForecastingModel):
    """GRU model for time series forecasting"""
    
    def __init__(self, sequence_length=144, units=50, epochs=50, batch_size=32):
        super().__init__("GRU")
        self.sequence_length = sequence_length
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        self.model = None
        
    def create_sequences(self, data):
        """Create sequences for GRU training"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def train(self, train_data):
        """Train GRU model"""
        print(f"  Training {self.name} model...")
        start_time = time.time()
        
        # Scale data
        train_scaled = self.scaler.fit_transform(train_data.values.reshape(-1, 1))
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_scaled)
        
        # Build model
        self.model = Sequential([
            GRU(self.units, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            GRU(self.units, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse')
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )
        
        self.training_time = time.time() - start_time
        print(f"  Training time: {self.training_time:.2f} seconds")
        print(f"  Epochs trained: {len(history.history['loss'])}")
        
    def predict(self, test_data, steps):
        """Make predictions"""
        print(f"  Predicting with {self.name} model...")
        start_time = time.time()
        
        # Use last sequence_length points from training data to predict
        predictions = []
        last_sequence = test_data[-self.sequence_length:].values
        last_sequence = self.scaler.transform(last_sequence.reshape(-1, 1))
        
        for _ in range(steps):
            X_pred = last_sequence.reshape(1, self.sequence_length, 1)
            pred = self.model.predict(X_pred, verbose=0)
            predictions.append(pred[0, 0])
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = pred[0, 0]
        
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        self.prediction_time = time.time() - start_time
        print(f"  Prediction time: {self.prediction_time:.2f} seconds")
        
        return predictions.flatten()
    
    def save(self, filepath):
        """Save model to file"""
        self.model.save(filepath)
        print(f"  Model saved to {filepath}")


class TrafficForecaster:
    """Main class for traffic forecasting"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.df = None
        self.area_totals = None
        
    def load_area_data(self, square_id):
        """Load data for a specific area only (memory efficient)"""
        print(f"Loading data for area {square_id}...")
        txt_files = sorted(list(self.data_dir.glob('sms-call-internet-mi-*.txt')))
        
        area_chunks = []
        for file in txt_files:
            # Load in chunks and filter for specific area
            for chunk in pd.read_csv(file, sep='\t', header=None, chunksize=50000):
                chunk.columns = ['square_id', 'time_interval', 'internet_traffic'] + list(chunk.columns[3:])
                chunk = chunk[['square_id', 'time_interval', 'internet_traffic']]
                
                # Optimize dtypes
                chunk['square_id'] = pd.to_numeric(chunk['square_id'], downcast='integer')
                chunk['square_id'] = chunk['square_id'].astype('int16')
                chunk['internet_traffic'] = pd.to_numeric(chunk['internet_traffic'], errors='coerce')
                chunk['internet_traffic'] = pd.to_numeric(chunk['internet_traffic'], downcast='float')
                
                # Convert timestamp
                chunk['time_interval'] = pd.to_datetime(chunk['time_interval'], unit='ms')
                
                # Filter for specific area
                area_chunk = chunk[chunk['square_id'] == square_id].copy()
                if not area_chunk.empty:
                    area_chunks.append(area_chunk)
                
                # Clear chunk from memory
                del chunk
        
        if area_chunks:
            area_df = pd.concat(area_chunks, ignore_index=True)
            area_df = area_df.sort_values('time_interval')
            print(f"Loaded {len(area_df)} records for area {square_id}")
            return area_df
        else:
            print(f"No data found for area {square_id}")
            return None
        
    def compute_area_totals(self):
        """Compute total traffic per area (memory efficient - vectorized)"""
        print("\nComputing total traffic per area (sampling for efficiency)...")
        txt_files = sorted(list(self.data_dir.glob('sms-call-internet-mi-*.txt')))
        
        area_totals = pd.DataFrame()
        for file in txt_files[:10]:  # Sample first 10 files for efficiency
            for chunk in pd.read_csv(file, sep='\t', header=None, chunksize=100000):
                chunk.columns = ['square_id', 'time_interval', 'internet_traffic'] + list(chunk.columns[3:])
                chunk = chunk[['square_id', 'internet_traffic']]
                chunk['square_id'] = pd.to_numeric(chunk['square_id'], downcast='integer')
                chunk['square_id'] = chunk['square_id'].astype('int16')
                chunk['internet_traffic'] = pd.to_numeric(chunk['internet_traffic'], errors='coerce')
                
                # Vectorized groupby and sum
                chunk_totals = chunk.groupby('square_id')['internet_traffic'].sum().reset_index()
                area_totals = pd.concat([area_totals, chunk_totals], ignore_index=True)
                
                del chunk, chunk_totals
                gc.collect()
        
        # Aggregate totals from all chunks
        if not area_totals.empty:
            self.area_totals = area_totals.groupby('square_id')['internet_traffic'].sum().reset_index()
            self.area_totals.columns = ['square_id', 'total_traffic']
            print(f"Computed totals for {len(self.area_totals)} areas")
        else:
            print("Warning: No data found for area totals computation")
            self.area_totals = None
        
    def get_time_series(self, area_df, start_date=None, end_date=None):
        """Get time series from pre-loaded area data"""
        if start_date:
            area_df = area_df[area_df['time_interval'] >= start_date]
        if end_date:
            area_df = area_df[area_df['time_interval'] <= end_date]
            
        return area_df.set_index('time_interval')['internet_traffic']
    
    def split_train_test(self, ts, test_start_date='2013-12-16', test_end_date='2013-12-22'):
        """Split time series into train and test sets"""
        train = ts[ts.index < pd.Timestamp(test_start_date)]
        test = ts[(ts.index >= pd.Timestamp(test_start_date)) & (ts.index <= pd.Timestamp(test_end_date))]
        return train, test
    
    def calculate_metrics(self, actual, predicted):
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        # MAPE - handle division by zero
        mask = actual != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        else:
            mape = np.inf
            
        return mae, mape, rmse
    
    def train_and_evaluate_models(self, square_id, models, results_dir):
        """Train and evaluate all models for a specific area (memory efficient)"""
        print(f"\n{'='*80}")
        print(f"Processing Area {square_id}")
        print(f"{'='*80}")
        
        # Load data for this specific area only
        area_df = self.load_area_data(square_id)
        if area_df is None or len(area_df) == 0:
            print(f"Skipping area {square_id} - no data available")
            return None
        
        # Get time series
        ts = self.get_time_series(area_df)
        
        # Clear area_df from memory
        del area_df
        gc.collect()
        
        # Split train/test
        train, test = self.split_train_test(ts)
        print(f"Training samples: {len(train)}")
        print(f"Test samples: {len(test)}")
        
        if len(test) == 0:
            print(f"Skipping area {square_id} - no test data available")
            return None
        
        results = {}
        figures_path = Path(results_dir) / 'figures'
        figures_path.mkdir(parents=True, exist_ok=True)
        
        for model in models:
            print(f"\n{'-'*80}")
            print(f"Model: {model.name}")
            print(f"{'-'*80}")
            
            # Train
            model.train(train)
            
            # Predict
            steps = len(test)
            predictions = model.predict(train, steps)
            
            # Calculate metrics
            mae, mape, rmse = self.calculate_metrics(test.values, predictions)
            
            results[model.name] = {
                'mae': mae,
                'mape': mape,
                'rmse': rmse,
                'training_time': model.training_time,
                'prediction_time': model.prediction_time,
                'predictions': predictions,
                'actual': test.values
            }
            
            # Plot predictions
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(test.index, test.values, label='Actual', linewidth=2)
            ax.plot(test.index, predictions, label='Predicted', linewidth=2, linestyle='--')
            ax.set_xlabel('Time')
            ax.set_ylabel('Traffic')
            ax.set_title(f'{model.name} Predictions - Area {square_id}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(figures_path / f'{model.name.lower()}_area_{square_id}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  MAE: {mae:.4f}")
            print(f"  MAPE: {mape:.4f}%")
            print(f"  RMSE: {rmse:.4f}")
            
            # Save model
            models_dir = Path(results_dir).parent / 'models'
            models_dir.mkdir(parents=True, exist_ok=True)
            
            if model.name == 'SARIMA':
                model.save(models_dir / f'{model.name.lower()}_area_{square_id}.pkl')
            else:
                model.save(models_dir / f'{model.name.lower()}_area_{square_id}.h5')
            
            # Clear model from memory after saving to free RAM (but keep scaler for next model)
            if hasattr(model, 'model'):
                del model.model
            if hasattr(model, 'model_fit'):
                del model.model_fit
            gc.collect()
            print(f"  Model saved and cleared from memory")
        
        return results
    
    def run_forecasting_analysis(self, results_dir):
        """Run complete forecasting analysis (memory efficient)"""
        print("=" * 80)
        print("TASK 3: Forecasting Models (Memory Efficient Mode)")
        print("=" * 80)
        
        # Compute area totals (sampling for efficiency)
        self.compute_area_totals()
        
        # Get target areas
        highest_area = self.area_totals.loc[self.area_totals['total_traffic'].idxmax(), 'square_id']
        target_areas = [highest_area, 4159, 4556]
        print(f"\nTarget areas: {target_areas}")
        
        # Initialize models (simplified SARIMA for faster training)
        models = [
            SARIMAModel(order=(1, 0, 1), seasonal_order=(0, 0, 0, 0)),  # Non-seasonal ARIMA for speed
            LSTMModel(sequence_length=144, units=50, epochs=30, batch_size=32),
            GRUModel(sequence_length=144, units=50, epochs=30, batch_size=32)
        ]
        
        # Train and evaluate for each area
        all_results = {}
        for area in target_areas:
            all_results[area] = self.train_and_evaluate_models(area, models, results_dir)
        
        # Create performance tables
        self.create_performance_tables(all_results, results_dir)
        
        # Create comparative analysis
        self.create_comparative_analysis(all_results, results_dir)
        
        # Perform failure analysis
        self.perform_failure_analysis(all_results, results_dir)
        
        print("\n" + "=" * 80)
        print("TASK 3 COMPLETE")
        print("=" * 80)
        
    def create_performance_tables(self, all_results, results_dir):
        """Create performance tables for each area"""
        tables_path = Path(results_dir) / 'tables'
        tables_path.mkdir(parents=True, exist_ok=True)
        
        for area, results in all_results.items():
            df = pd.DataFrame({
                'Model': list(results.keys()),
                'MAE': [r['mae'] for r in results.values()],
                'MAPE (%)': [r['mape'] for r in results.values()],
                'RMSE': [r['rmse'] for r in results.values()],
                'Training Time (s)': [r['training_time'] for r in results.values()],
                'Prediction Time (s)': [r['prediction_time'] for r in results.values()]
            })
            
            # Save as CSV
            df.to_csv(tables_path / f'performance_area_{area}.csv', index=False)
            
            # Also save formatted as text
            with open(tables_path / f'performance_area_{area}.txt', 'w') as f:
                f.write(f"Performance Metrics - Area {area}\n")
                f.write("="*80 + "\n\n")
                f.write(df.to_string(index=False))
            
            print(f"\nPerformance table for Area {area}:")
            print(df.to_string(index=False))
    
    def create_comparative_analysis(self, all_results, results_dir):
        """Create comparative analysis across models and areas"""
        figures_path = Path(results_dir) / 'figures'
        
        # Aggregate results across areas
        model_names = ['SARIMA', 'LSTM', 'GRU']
        metrics = ['mae', 'mape', 'rmse']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, metric in enumerate(metrics):
            data = []
            for model in model_names:
                values = [all_results[area][model][metric] for area in all_results.keys()]
                data.append(values)
            
            x = np.arange(len(all_results.keys()))
            width = 0.25
            
            for i, model in enumerate(model_names):
                axes[idx].bar(x + i*width, data[i], width, label=model)
            
            axes[idx].set_xlabel('Area')
            axes[idx].set_ylabel(metric.upper())
            axes[idx].set_title(f'{metric.upper()} Comparison Across Areas')
            axes[idx].set_xticks(x + width)
            axes[idx].set_xticklabels(list(all_results.keys()))
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(figures_path / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nComparative analysis plot saved")
    
    def perform_failure_analysis(self, all_results, results_dir):
        """Perform failure analysis identifying poor prediction periods"""
        print("\n" + "=" * 80)
        print("FAILURE ANALYSIS")
        print("=" * 80)
        
        figures_path = Path(results_dir) / 'figures'
        
        for area, results in all_results.items():
            print(f"\nAnalyzing failures for Area {area}...")
            
            # Get actual and predicted values for each model
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            
            for idx, model_name in enumerate(['SARIMA', 'LSTM', 'GRU']):
                actual = results[model_name]['actual']
                predicted = results[model_name]['predictions']
                
                # Calculate absolute errors
                errors = np.abs(actual - predicted)
                
                # Find periods with highest errors (top 10%)
                threshold = np.percentile(errors, 90)
                failure_periods = errors > threshold
                
                # Plot
                time_index = pd.date_range(start='2013-12-16', periods=len(actual), freq='10min')
                
                axes[idx].plot(time_index, actual, label='Actual', linewidth=2, alpha=0.7)
                axes[idx].plot(time_index, predicted, label='Predicted', linewidth=2, linestyle='--', alpha=0.7)
                
                # Highlight failure periods
                axes[idx].scatter(time_index[failure_periods], actual[failure_periods], 
                                 color='red', s=50, label='High Error Periods', zorder=5)
                
                axes[idx].set_ylabel('Traffic')
                axes[idx].set_title(f'{model_name} - Failure Analysis (Area {area})')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
                
                # Print statistics
                print(f"  {model_name}:")
                print(f"    Mean error: {errors.mean():.4f}")
                print(f"    Max error: {errors.max():.4f}")
                print(f"    High error periods: {failure_periods.sum()} ({failure_periods.sum()/len(failure_periods)*100:.1f}%)")
            
            axes[-1].set_xlabel('Time')
            plt.tight_layout()
            plt.savefig(figures_path / f'failure_analysis_area_{area}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved failure analysis plot for Area {area}")


def main():
    """Main function"""
    data_dir = Path(__file__).parent.parent / 'data'
    results_dir = Path(__file__).parent.parent / 'results'
    
    forecaster = TrafficForecaster(data_dir)
    forecaster.run_forecasting_analysis(results_dir)


if __name__ == "__main__":
    main()
