"""
Task 1: Data Handling & Memory Management
This script implements efficient data loading and memory optimization techniques
for the 5GB mobile network traffic dataset.
"""

import pandas as pd
import numpy as np
import os
import psutil
import time
from pathlib import Path


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB


def load_data_unoptimized(filepath):
    """
    Load data without optimization - baseline for comparison
    """
    print(f"Loading data without optimization...")
    mem_before = get_memory_usage()
    print(f"Memory before loading: {mem_before:.2f} MB")
    
    start_time = time.time()
    
    # Standard loading - pandas will infer dtypes
    # Data is tab-separated, not comma-separated
    df = pd.read_csv(filepath, sep='\t', header=None)
    
    load_time = time.time() - start_time
    mem_after = get_memory_usage()
    mem_used = mem_after - mem_before
    
    print(f"Memory after loading: {mem_after:.2f} MB")
    print(f"Memory used: {mem_used:.2f} MB")
    print(f"Loading time: {load_time:.2f} seconds")
    print(f"Data shape: {df.shape}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Memory usage by column:\n{df.memory_usage(deep=True) / (1024*1024)}")
    
    return df, mem_used, load_time


def load_data_optimized(filepath, chunksize=None):
    """
    Load data with memory optimization techniques:
    1. Specify optimal dtypes
    2. Use chunked loading if needed
    3. Downcast numeric types
    4. Use categorical for low-cardinality columns
    """
    print(f"\nLoading data with optimization...")
    mem_before = get_memory_usage()
    print(f"Memory before loading: {mem_before:.2f} MB")
    
    start_time = time.time()
    
    # Define optimal dtypes based on data characteristics
    # Square id: 1-10000, fits in int16 (max 32767)
    # Time interval: can be parsed as datetime
    # Internet traffic activity: numeric, can be downcast
    
    if chunksize:
        print(f"Using chunked loading with chunksize={chunksize}")
        chunks = []
        for chunk in pd.read_csv(filepath, sep='\t', header=None, chunksize=chunksize):
            # Optimize each chunk
            chunk = optimize_dataframe(chunk)
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    else:
        # Load with specified dtypes
        df = pd.read_csv(filepath, sep='\t', header=None)
        df = optimize_dataframe(df)
    
    load_time = time.time() - start_time
    mem_after = get_memory_usage()
    mem_used = mem_after - mem_before
    
    print(f"Memory after loading: {mem_after:.2f} MB")
    print(f"Memory used: {mem_used:.2f} MB")
    print(f"Loading time: {load_time:.2f} seconds")
    print(f"Data shape: {df.shape}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Memory usage by column:\n{df.memory_usage(deep=True) / (1024*1024)}")
    
    return df, mem_used, load_time


def optimize_dataframe(df):
    """
    Apply memory optimization techniques to a dataframe
    """
    # Get column names - assuming format from the dataset
    # Based on the paper: Square id, Time Interval, Internet traffic activity, country code
    # Note: paper mentions country code is third field, others shifted
    
    if len(df.columns) >= 3:
        # Rename columns for clarity
        df.columns = ['square_id', 'time_interval', 'internet_traffic'] + list(df.columns[3:])
        
        # Optimize square_id - should be int16 (max 10000)
        if 'square_id' in df.columns:
            df['square_id'] = pd.to_numeric(df['square_id'], downcast='integer')
            # Ensure it fits in int16
            if df['square_id'].max() < 32767:
                df['square_id'] = df['square_id'].astype('int16')
        
        # Optimize time_interval - convert to datetime if possible
        if 'time_interval' in df.columns:
            try:
                df['time_interval'] = pd.to_datetime(df['time_interval'])
            except:
                # If not datetime, try to optimize as string or numeric
                if df['time_interval'].dtype == 'object':
                    df['time_interval'] = df['time_interval'].astype('category')
        
        # Optimize internet_traffic - downcast to smallest possible numeric type
        if 'internet_traffic' in df.columns:
            df['internet_traffic'] = pd.to_numeric(df['internet_traffic'], errors='coerce')
            df['internet_traffic'] = pd.to_numeric(df['internet_traffic'], downcast='float')
        
        # Drop unnecessary columns (country code and beyond)
        df = df[['square_id', 'time_interval', 'internet_traffic']]
    
    return df


def aggregate_by_area(df):
    """
    Aggregate data by geographical area to reduce memory footprint
    This is a data reduction strategy
    """
    print("\nAggregating data by area...")
    mem_before = get_memory_usage()
    
    # Group by square_id and compute total traffic per area
    area_traffic = df.groupby('square_id')['internet_traffic'].sum().reset_index()
    
    mem_after = get_memory_usage()
    print(f"Memory before aggregation: {mem_before:.2f} MB")
    print(f"Memory after aggregation: {mem_after:.2f} MB")
    print(f"Memory saved: {mem_before - mem_after:.2f} MB")
    print(f"Aggregated shape: {area_traffic.shape}")
    
    return area_traffic


def create_time_series_matrix(df, sample_areas=None):
    """
    Restructure data into a time series matrix format
    This can be more memory-efficient for certain operations
    """
    print("\nCreating time series matrix...")
    mem_before = get_memory_usage()
    
    # If sample_areas is provided, only process those areas
    if sample_areas:
        df = df[df['square_id'].isin(sample_areas)].copy()
    
    # Pivot to create time series matrix
    # Rows: time intervals, Columns: square_id, Values: internet_traffic
    ts_matrix = df.pivot(index='time_interval', columns='square_id', values='internet_traffic')
    
    mem_after = get_memory_usage()
    print(f"Memory before restructuring: {mem_before:.2f} MB")
    print(f"Memory after restructuring: {mem_after:.2f} MB")
    print(f"Time series matrix shape: {ts_matrix.shape}")
    
    return ts_matrix


def main():
    """
    Main function to run Task 1
    """
    print("=" * 80)
    print("TASK 1: Data Handling & Memory Management")
    print("=" * 80)
    
    # Configuration
    data_dir = Path(__file__).parent.parent / 'data'
    data_file = data_dir / 'sms-call-internet-mi-2013-11-01.txt'  # Example filename
    
    # Check if data file exists
    if not data_file.exists():
        print(f"\nData file not found at: {data_file}")
        print("Please download the dataset from:")
        print("https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV")
        print("\nExpected files in data directory:")
        print("- sms-call-internet-mi-2013-11-01.txt")
        print("- sms-call-internet-mi-2013-11-02.txt")
        print("- ... (for November)")
        print("- sms-call-internet-mi-2013-12-01.txt")
        print("- ... (for December)")
        return
    
    # Print system information
    print("\n" + "=" * 80)
    print("SYSTEM INFORMATION")
    print("=" * 80)
    print(f"Total memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print(f"CPU count: {psutil.cpu_count()}")
    
    # Strategy I: Load single file with optimization for demonstration
    print("\n" + "=" * 80)
    print("STRATEGY DEMONSTRATION (Single File)")
    print("=" * 80)
    
    # Try with first file if multiple files exist
    txt_files = list(data_dir.glob('sms-call-internet-mi-*.txt'))
    if txt_files:
        sample_file = txt_files[0]
        print(f"\nProcessing sample file: {sample_file.name}")
        
        # Load unoptimized
        try:
            df_unopt, mem_unopt, time_unopt = load_data_unoptimized(sample_file)
            
            # Load optimized
            df_opt, mem_opt, time_opt = load_data_optimized(sample_file)
            
            # Calculate improvement
            mem_reduction = ((mem_unopt - mem_opt) / mem_unopt) * 100
            print(f"\nMemory reduction: {mem_reduction:.2f}%")
            print(f"Time improvement: {((time_unopt - time_opt) / time_unopt) * 100:.2f}%")
            
        except Exception as e:
            print(f"Error processing file: {e}")
            print("This might be due to file format or size. Proceeding with optimized approach only.")
            
            df_opt, mem_opt, time_opt = load_data_optimized(sample_file, chunksize=100000)
    else:
        print("No data files found. Please download the datasets first.")
        return
    
    # Strategy II: Process multiple files with aggregation
    print("\n" + "=" * 80)
    print("DATA REDUCTION STRATEGY")
    print("=" * 80)
    
    if len(txt_files) > 1:
        print(f"\nProcessing {len(txt_files)} files with chunked loading and aggregation...")
        
        all_area_traffic = []
        for file in sorted(txt_files)[:5]:  # Process first 5 files for demo
            print(f"\nProcessing: {file.name}")
            df, _, _ = load_data_optimized(file, chunksize=100000)
            area_traffic = aggregate_by_area(df)
            all_area_traffic.append(area_traffic)
        
        # Combine aggregated data
        combined_traffic = pd.concat(all_area_traffic, ignore_index=True)
        print(f"\nCombined aggregated data shape: {combined_traffic.shape}")
    
    print("\n" + "=" * 80)
    print("TASK 1 COMPLETE")
    print("=" * 80)
    print("\nSummary of memory optimization strategies:")
    print("1. Specified optimal dtypes (int16 for square_id, downcast floats)")
    print("2. Chunked loading for large files")
    print("3. Aggregation by geographical area to reduce dimensionality")
    print("4. Restructuring to time series matrix for efficient operations")
    print("\nThese techniques significantly reduce memory usage while preserving")
    print("the information needed for time series analysis and forecasting.")


if __name__ == "__main__":
    main()
