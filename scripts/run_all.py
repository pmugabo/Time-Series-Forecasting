"""
Main script to run all tasks for the Mobile Network Traffic Analysis project
"""

import sys
from pathlib import Path

# Add parent directory to path to import scripts
sys.path.append(str(Path(__file__).parent))

from task1_data_loading import main as task1_main
from task2_eda import MobileTrafficEDA
from task3_forecasting import TrafficForecaster


def main():
    """Run all tasks sequentially"""
    print("=" * 80)
    print("MOBILE NETWORK TRAFFIC ANALYSIS - COMPLETE PIPELINE")
    print("=" * 80)
    print()
    
    # Task 1: Data Loading and Memory Management
    print("\n" + "=" * 80)
    print("RUNNING TASK 1: Data Handling & Memory Management")
    print("=" * 80)
    task1_main()
    
    # Task 2: Exploratory Data Analysis
    print("\n" + "=" * 80)
    print("RUNNING TASK 2: Exploratory Data Analysis")
    print("=" * 80)
    data_dir = Path(__file__).parent.parent / 'data'
    results_dir = Path(__file__).parent.parent / 'results'
    eda = MobileTrafficEDA(data_dir)
    eda.run_all_analyses(results_dir)
    
    # Task 3: Forecasting Models
    print("\n" + "=" * 80)
    print("RUNNING TASK 3: Forecasting Models")
    print("=" * 80)
    forecaster = TrafficForecaster(data_dir)
    forecaster.run_forecasting_analysis(results_dir)
    
    print("\n" + "=" * 80)
    print("ALL TASKS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nResults saved to: {results_dir}")
    print("\nNext steps:")
    print("1. Review the generated figures in results/figures/")
    print("2. Review the performance tables in results/tables/")
    print("3. Compile the PDF report with your analysis and interpretations")
    print("4. Create a video demonstration as required")


if __name__ == "__main__":
    main()
