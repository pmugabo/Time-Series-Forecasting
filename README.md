# Mobile Network Traffic Time Series Analysis and Forecasting

This project implements a comprehensive time series analysis and forecasting system for mobile network traffic data from Milan, Italy (November-December 2013).

## Video Demonstration

[Video Link Placeholder]

## Project Overview

- **Dataset**: 10,000 geographical areas (100×100 grid), 2 months of data at 10-minute intervals
- **Models**: SARIMA, LSTM, GRU for time series forecasting
- **Target Areas**: Highest traffic area (6383), Square 4159, Square 4556
- **Hardware**: Optimized for 8GB RAM

## Project Structure

```
mobile_traffic_analysis/
├── data/                  # Raw datasets (17 GB)
├── scripts/               # Analysis scripts
│   ├── task1_data_loading.py
│   ├── task2_eda.py
│   └── task3_forecasting.py
├── results/               # Generated outputs
│   ├── figures/          # 20 plots (EDA + forecasting)
│   └── tables/           # Performance tables
├── models/               # 9 saved models (3 × 3 areas)
└── requirements.txt
```

## Dataset

The dataset is from Telecom Italia Mobile (TIM) and contains mobile network traffic information for Milan, organized in a 100×100 grid (10,000 areas) over two months.

**Note:** The raw data files (17GB total) are not included in this repository due to size limitations. Please download them from the source.

**Download links:**
- Telecommunications activity dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV
- Grid dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QJWLFU

Place the downloaded files in the `data/` directory. Only `milano-grid.geojson` is included in the repository.

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Task 1: Data Loading and Memory Management
```bash
python scripts/task1_data_loading.py
```

### Task 2: Exploratory Data Analysis
```bash
python scripts/task2_eda.py
```

### Task 3: Forecasting Models
```bash
python scripts/task3_forecasting.py
```

Or run the complete pipeline:
```bash
python scripts/run_all.py
```

## Results

**Task 1**: Memory optimization using chunked loading, dtype downcasting, area-specific processing

**Task 2**: EDA including:
- PDF of traffic distribution
- Time series plots
- Stationarity analysis (ADF test)
- Seasonal decomposition
- ACF/PACF analysis
- Spatial heatmap
- Anomaly detection

**Task 3**: Forecasting results for 3 areas:
- SARIMA: Best accuracy, fastest training (~5-20 seconds)
- LSTM: Comparable results, longer training (~300-700 seconds)
- GRU: Similar to LSTM (~500-650 seconds)

**Metrics**: MAE, MAPE, RMSE, training/prediction times

## Requirements

- Python 3.8+
- See requirements.txt for full list of dependencies

## Reference

Barlacchi et al., "A multi-source dataset of urban life in the city of Milan and the Province of Trentino." Sci Data 2, 150055 (2015). https://doi.org/10.1038/sdata.2015.55
