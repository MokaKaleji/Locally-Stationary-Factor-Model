# Locally Stationary Factor Model (LSFM)

**Author:** Moka Kaleji  
**Based on:** Motta, Hafner & von Sachs (2011)

This repo implements both estimation and forecasting for the LSFM on macroeconomic series.  

### Files

- **lsfm_estimation.m**  
  - Interactive selection of monthly vs. quarterly data  
  - Specify training sample size `T_train`  
  - Choose number of factors `q` and bandwidth `h`  
  - Produces:
    - `lsfm_estimation_results.mat` (factors `Fhat`, loadings `Lhat`, metadata)  
    - Diagnostic plot of variance explained  

- **lsfm_forecasting.m**  
  - Loads `lsfm_estimation_results.mat`  
  - Specify VAR lag order `p`  
  - Forecasts H-step ahead for GDP, Unemployment, Inflation, Interest Rate  
  - Computes MSFE/RMSE and plots actual vs. forecast  

### Usage

1. Open MATLAB (R2020b+ recommended) with Econometrics & ML toolboxes.
2. Place processed data files in:
3. Run estimation:
```matlab
lsfm_estimation
lsfm_forecasting
