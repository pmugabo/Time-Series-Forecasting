"""
Task 2: Exploratory Data Analysis / Data Characterization
This script performs comprehensive EDA on the mobile network traffic dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MobileTrafficEDA:
    """Class for performing EDA on mobile traffic data"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.df = None
        self.area_totals = None
        self.ts_matrix = None
        
    def load_data(self, sample_files=None):
        """Load and preprocess data"""
        print("Loading data...")
        txt_files = sorted(list(self.data_dir.glob('sms-call-internet-mi-*.txt')))
        
        if sample_files:
            txt_files = txt_files[:sample_files]
        
        dfs = []
        for file in txt_files:
            print(f"  Loading: {file.name}")
            df = pd.read_csv(file, sep='\t', header=None)
            df.columns = ['square_id', 'time_interval', 'internet_traffic'] + list(df.columns[3:])
            df = df[['square_id', 'time_interval', 'internet_traffic']]
            
            # Optimize dtypes
            df['square_id'] = pd.to_numeric(df['square_id'], downcast='integer')
            df['square_id'] = df['square_id'].astype('int16')
            df['internet_traffic'] = pd.to_numeric(df['internet_traffic'], errors='coerce')
            df['internet_traffic'] = pd.to_numeric(df['internet_traffic'], downcast='float')
            
            # Convert timestamp from milliseconds to datetime
            df['time_interval'] = pd.to_datetime(df['time_interval'], unit='ms')
            
            dfs.append(df)
        
        self.df = pd.concat(dfs, ignore_index=True)
        
        print(f"Loaded {len(self.df)} records from {len(txt_files)} files")
        print(f"Date range: {self.df['time_interval'].min()} to {self.df['time_interval'].max()}")
        
    def compute_area_totals(self):
        """Compute total traffic per geographical area"""
        print("\nComputing total traffic per area...")
        self.area_totals = self.df.groupby('square_id')['internet_traffic'].sum().reset_index()
        self.area_totals.columns = ['square_id', 'total_traffic']
        print(f"Computed totals for {len(self.area_totals)} areas")
        
    def plot_pdf_traffic(self, save_path):
        """
        I. Plot probability density function of traffic across 10,000 areas
        """
        print("\nI. Computing PDF of total traffic across areas...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        axes[0].hist(self.area_totals['total_traffic'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Total Traffic (2 months)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Total Traffic Across Areas')
        axes[0].grid(True, alpha=0.3)
        
        # KDE plot
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(self.area_totals['total_traffic'])
        x_range = np.linspace(self.area_totals['total_traffic'].min(), 
                              self.area_totals['total_traffic'].max(), 1000)
        axes[1].plot(x_range, kde(x_range), linewidth=2)
        axes[1].fill_between(x_range, kde(x_range), alpha=0.3)
        axes[1].set_xlabel('Total Traffic (2 months)')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Probability Density Function of Traffic')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved PDF plot to {save_path}")
        plt.close()
        
        # Print statistics
        print(f"  Mean total traffic: {self.area_totals['total_traffic'].mean():.2f}")
        print(f"  Median total traffic: {self.area_totals['total_traffic'].median():.2f}")
        print(f"  Std total traffic: {self.area_totals['total_traffic'].std():.2f}")
        print(f"  Skewness: {stats.skew(self.area_totals['total_traffic']):.2f}")
        print(f"  Kurtosis: {stats.kurtosis(self.area_totals['total_traffic']):.2f}")
        
    def get_time_series_for_area(self, square_id, start_date=None, end_date=None):
        """Get time series for a specific area"""
        area_df = self.df[self.df['square_id'] == square_id].copy()
        area_df = area_df.sort_values('time_interval')
        
        if start_date:
            area_df = area_df[area_df['time_interval'] >= start_date]
        if end_date:
            area_df = area_df[area_df['time_interval'] <= end_date]
            
        return area_df.set_index('time_interval')['internet_traffic']
    
    def plot_time_series_comparison(self, save_path):
        """
        II. Plot time series for (i) highest traffic area, (ii) Square 4159, (iii) Square 4556
        """
        print("\nII. Plotting time series for selected areas (first 2 weeks)...")
        
        # Find highest traffic area
        highest_area = self.area_totals.loc[self.area_totals['total_traffic'].idxmax(), 'square_id']
        print(f"  Highest traffic area: {highest_area}")
        
        # Get first two weeks of data
        start_date = self.df['time_interval'].min()
        end_date = start_date + pd.Timedelta(days=14)
        
        areas = [highest_area, 4159, 4556]
        area_names = [f'Highest Traffic ({highest_area})', 'Square 4159', 'Square 4556']
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        
        for idx, (area, name) in enumerate(zip(areas, area_names)):
            ts = self.get_time_series_for_area(area, start_date, end_date)
            axes[idx].plot(ts.index, ts.values, linewidth=1.5)
            axes[idx].set_ylabel('Traffic')
            axes[idx].set_title(name)
            axes[idx].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time Interval')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved time series comparison to {save_path}")
        plt.close()
        
    def analyze_stationarity(self, square_id, save_path):
        """
        III. Stationarity analysis with rolling statistics and ADF test
        """
        print(f"\nIII. Analyzing stationarity for area {square_id}...")
        
        ts = self.get_time_series_for_area(square_id)
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        
        # Rolling statistics
        rolling_mean = ts.rolling(window=144).mean()  # 1 day = 144 intervals (10-min)
        rolling_std = ts.rolling(window=144).std()
        
        axes[0].plot(ts.index, ts.values, label='Original', alpha=0.7)
        axes[0].plot(rolling_mean.index, rolling_mean.values, label='Rolling Mean (24h)', color='red')
        axes[0].plot(rolling_std.index, rolling_std.values, label='Rolling Std (24h)', color='green')
        axes[0].set_ylabel('Traffic')
        axes[0].set_title(f'Rolling Statistics - Area {square_id}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # ADF Test
        result = adfuller(ts.dropna())
        adf_statistic = result[0]
        p_value = result[1]
        critical_values = result[4]
        
        axes[1].bar(['ADF Statistic', '1%', '5%', '10%'], 
                   [adf_statistic, critical_values['1%'], critical_values['5%'], critical_values['10%']])
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_ylabel('Value')
        axes[1].set_title('Augmented Dickey-Fuller Test Results')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved stationarity analysis to {save_path}")
        plt.close()
        
        # Print ADF results
        print(f"  ADF Statistic: {adf_statistic:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Critical Values:")
        for key, value in critical_values.items():
            print(f"    {key}: {value:.4f}")
        
        is_stationary = p_value < 0.05
        print(f"  Series is {'stationary' if is_stationary else 'non-stationary'} (p < 0.05)")
        
        return is_stationary
        
    def decompose_time_series(self, square_id, save_path):
        """
        IV. Decompose time series into trend, seasonal, and residual components
        """
        print(f"\nIV. Decomposing time series for area {square_id}...")
        
        ts = self.get_time_series_for_area(square_id)
        
        # Resample to hourly for clearer decomposition
        ts_hourly = ts.resample('H').sum()
        
        # Perform decomposition
        # Period = 24 for daily seasonality
        decomposition = seasonal_decompose(ts_hourly.dropna(), model='additive', period=24)
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        axes[0].plot(decomposition.observed.index, decomposition.observed.values)
        axes[0].set_ylabel('Traffic')
        axes[0].set_title('Original Time Series')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(decomposition.trend.index, decomposition.trend.values)
        axes[1].set_ylabel('Traffic')
        axes[1].set_title('Trend Component')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values)
        axes[2].set_ylabel('Traffic')
        axes[2].set_title('Seasonal Component')
        axes[2].grid(True, alpha=0.3)
        
        axes[3].plot(decomposition.resid.index, decomposition.resid.values)
        axes[3].set_ylabel('Traffic')
        axes[3].set_title('Residual Component')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved decomposition to {save_path}")
        plt.close()
        
    def plot_acf_pacf(self, square_id, save_path, lags=50):
        """
        V. Plot ACF and PACF
        """
        print(f"\nV. Plotting ACF and PACF for area {square_id}...")
        
        ts = self.get_time_series_for_area(square_id)
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        
        plot_acf(ts.dropna(), lags=lags, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)')
        axes[0].grid(True, alpha=0.3)
        
        plot_pacf(ts.dropna(), lags=lags, ax=axes[1], method='ywm')
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved ACF/PACF plots to {save_path}")
        plt.close()
        
    def create_spatial_heatmap(self, save_path):
        """
        VI. Create heatmap of traffic intensity across the grid
        """
        print("\nVI. Creating spatial heatmap...")
        
        # Create 100x100 grid
        grid = np.zeros((100, 100))
        
        for _, row in self.area_totals.iterrows():
            square_id = int(row['square_id'])
            # Convert square_id to grid coordinates
            # Assuming square_id 1 is at (0,0) and 10000 at (99,99)
            row_idx = (square_id - 1) // 100
            col_idx = (square_id - 1) % 100
            grid[row_idx, col_idx] = row['total_traffic']
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(grid, cmap='YlOrRd', aspect='equal')
        ax.set_xlabel('Grid Column')
        ax.set_ylabel('Grid Row')
        ax.set_title('Spatial Distribution of Total Traffic (2 months)')
        plt.colorbar(im, ax=ax, label='Total Traffic')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved spatial heatmap to {save_path}")
        plt.close()
        
    def analyze_anomalies(self, square_id, save_path):
        """
        VII. Analyze anomalies and outliers
        """
        print(f"\nVII. Analyzing anomalies for area {square_id}...")
        
        ts = self.get_time_series_for_area(square_id)
        
        # Calculate z-scores
        z_scores = np.abs(stats.zscore(ts.dropna()))
        threshold = 3
        
        anomalies = z_scores > threshold
        anomaly_count = anomalies.sum()
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Time series with anomalies highlighted
        axes[0].plot(ts.index, ts.values, label='Traffic', alpha=0.7)
        anomaly_times = ts.dropna().index[anomalies]
        anomaly_values = ts.dropna().values[anomalies]
        axes[0].scatter(anomaly_times, anomaly_values, color='red', s=50, label=f'Anomalies (z>{threshold})')
        axes[0].set_ylabel('Traffic')
        axes[0].set_title(f'Time Series with Anomalies - Area {square_id}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(ts.dropna().values, vert=False)
        axes[1].set_xlabel('Traffic')
        axes[1].set_title('Box Plot of Traffic Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved anomaly analysis to {save_path}")
        plt.close()
        
        print(f"  Number of anomalies detected: {anomaly_count}")
        print(f"  Percentage of data points: {(anomaly_count/len(ts)*100):.2f}%")
        print(f"  Anomaly threshold: z-score > {threshold}")
        
    def run_all_analyses(self, results_dir):
        """Run all EDA analyses"""
        print("=" * 80)
        print("TASK 2: Exploratory Data Analysis")
        print("=" * 80)
        
        results_path = Path(results_dir)
        figures_path = results_path / 'figures'
        figures_path.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.load_data(sample_files=10)  # Load sample files for demo
        
        # Compute area totals
        self.compute_area_totals()
        
        # I. PDF plot
        self.plot_pdf_traffic(figures_path / 'pdf_traffic.png')
        
        # II. Time series comparison
        self.plot_time_series_comparison(figures_path / 'timeseries_comparison.png')
        
        # Get highest traffic area for further analyses
        highest_area = self.area_totals.loc[self.area_totals['total_traffic'].idxmax(), 'square_id']
        
        # III. Stationarity analysis
        self.analyze_stationarity(highest_area, figures_path / 'stationarity_analysis.png')
        
        # IV. Decomposition
        self.decompose_time_series(highest_area, figures_path / 'time_series_decomposition.png')
        
        # V. ACF/PACF
        self.plot_acf_pacf(highest_area, figures_path / 'acf_pacf.png')
        
        # VI. Spatial heatmap
        self.create_spatial_heatmap(figures_path / 'spatial_heatmap.png')
        
        # VII. Anomaly analysis
        self.analyze_anomalies(highest_area, figures_path / 'anomaly_analysis.png')
        
        print("\n" + "=" * 80)
        print("TASK 2 COMPLETE")
        print("=" * 80)
        print(f"\nAll figures saved to: {figures_path}")


def main():
    """Main function"""
    data_dir = Path(__file__).parent.parent / 'data'
    results_dir = Path(__file__).parent.parent / 'results'
    
    eda = MobileTrafficEDA(data_dir)
    eda.run_all_analyses(results_dir)


if __name__ == "__main__":
    main()
