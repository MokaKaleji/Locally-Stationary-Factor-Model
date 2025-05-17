%% lsfm_forecasting.m
% Author: Moka Kaleji • Contact: mohammadkaleji1998@gmail.com
% Affiliation: Master Thesis in Econometrics: 
% Advancing High-Dimensional Factor Models: Integrating Time-Varying 
% Loadings and Transition Matrix with Dynamic Factors.
% University of Bologna
% Description:
%   Executes forecasting for selected macroeconomic indicators (GDP, Unemployment,
%   Inflation, Interest Rate) using the previously estimated Locally Stationary
%   Factor Model (LSFM). Supports user-defined VAR lag order and computes
%   forecasting performance metrics with visualization.

clear; close all; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Dataset Selection, Load Data and Define Key Variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Allow user to select a dataset (monthly or quarterly) and load 
% corresponding data.
% Explanation: Presents a dialog for selecting between monthly (MD1959.xlsx)
% or quarterly (QD1959.xlsx) datasets, loads the chosen data, and defines 
% key variables (GDP, Unemployment, Inflation, 1-Year Treasury - 3-Month Treasury)
% for forecasting evaluation. The datasets are high-dimensional time series
% used in macroeconomic forecasting, with specific indices for key variables.
% References:
%   - McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for 
%     macroeconomic research. Journal of Business & Economic Statistics, 
%     34(4), 574-589. (And FRED-QD).
options = {'Monthly (MD1959.xlsx)', 'Quarterly (QD1959.xlsx)'};
[choiceIndex, ok] = listdlg( ...
    'PromptString', 'Select the dataset to load:', ...
    'SelectionMode', 'single', ...
    'ListString', options, ...
    'Name', 'Dataset Selection', ...
    'ListSize', [400 200]);
if ~ok
    error('No dataset selected. Exiting...');
end
% Load data based on user selection
% Explanation: Loads the dataset from a specified filepath, sets the time 
% series length T, and extracts key variables’ indices. The monthly dataset
% has T=790 observations, and the quarterly has T=264, reflecting typical 
% macroeconomic data frequencies.
switch choiceIndex
    case 1
        filepath = ['/Users/moka/Research/Thesis/Live Project/' ...
            'Processed_Data/MD1959.xlsx'];
        T = 790;
        tableData = readtable(filepath);
        x = table2array(tableData);
        key_vars = [1, 24, 105, 77];                                       % Indices for key variables
    case 2
        filepath = ['/Users/moka/Research/Thesis/Live Project/' ...
            'Processed_Data/QD1959.xlsx'];
        T = 264;
        tableData = readtable(filepath);
        x = table2array(tableData(:,2:end));                               % Exclude ate column
        key_vars = [1, 58, 116, 147];                                      % Indices for key variables
    otherwise
        error('Unexpected selection index.');
end
var_names = {'GDP', 'Unemployment', 'Inflation', '1_Y Treasury - 3_M Treasury'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load LSFM Estimation Outputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('lsfm_estimation_results.mat', 'Fhat_train', 'Lhat_train', 'T_train', ...
    'mean_train', 'std_train', 'N');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Forecast Settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s = T - T_train;
H = input(sprintf('Enter forecast horizon H from 1 to T = %d: ', s));      % Forecast horizon
if ~(isscalar(H) && isnumeric(H) && H == floor(H) && H >= 1 && H <= s)
    error('H must be an **integer** between 1 and T = %d.', s);            
end

p_opt = input('Enter the VAR lag order (p, 1 to 12): ');                   % Var lag order
assert(p_opt >= 1 && p_opt <= 12, 'p must be integer between 1 and 12');
disp(['Using forecast horizon H = ', num2str(H)]);
disp(['Using VAR lag order p = ', num2str(p_opt)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Prepare Test Sample 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Extract and normalize the test data for forecasting evaluation.
% Explanation: Selects the test data x_test for the specified horizon H 
% from the full dataset and normalizes it using training data statistics 
% (mean_train, std_train) to match the scale of the estimation inputs,
% ensuring consistency.
x_test = x(T_train+1:T_train+H, :);    % Observations for evaluation
x_test_norm = (x_test - mean_train) ./ std_train;  % Normalize as in training

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Running Forecast Via LSFM + VAR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Forecast standardized observations using factor forecasts
yhat_norm = LSFM_Forecast(Fhat_train, Lhat_train, T_train, N, H, p_opt);
% Rescale forecasts to original units
yhat = yhat_norm .* std_train + mean_train;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MSFE/Ratio, RMSE, Diebold–Mariano Tests, Ljung–Box Tests, Encompassing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Evaluate forecast accuracy and compare against benchmarks.
% Explanation: Computes Mean Squared Forecast Error (MSFE), Root Mean 
% Squared Error (RMSE), MSFE ratios against random walk (RW) and AR(1) 
% benchmarks, and conducts statistical tests to assess forecast performance
% and residual properties.
% References:
%   - Diebold & Mariano (1995) for comparing predictive accuracy.
%   - Harvey & Todd (1983) for forecast encompassing tests.
%   - Ljung & Box (1978) for residual autocorrelation tests.

% --- MSFE and RMSE ---
% Purpose: Calculate MSFE and RMSE for key variables.
% Explanation: MSFE is the average squared forecast error, computed per 
% horizon and variable. RMSE is the square root of MSFE, providing a 
% scale-interpretable metric.
squared_errors = (x_test(:, key_vars) - yhat(:, key_vars)).^2;
MSFE_horizon = mean(squared_errors, 2);
MSFE_all = mean(squared_errors, 1);
MSFE_overall = mean(MSFE_all);

fprintf('Overall MSFE (original): %.6f\n', MSFE_overall);
for k = 1:length(key_vars)
    fprintf('MSFE (original) %s: %.6f\n', var_names{k}, MSFE_all(k));
end

% Root MSFE in original scale
RMSE = sqrt(MSFE_all);
for k = 1:length(key_vars)
    fprintf('RMSE %s: %.4f\n', var_names{k}, RMSE(k));
end

% --- Random Walk Benchmark ---
% Purpose: Compute MSFE for a naive random walk forecast.
% Explanation: The random walk forecast assumes y_{t+h} = y_t, using the 
% last training observation. MSFE ratios (DFTLTA vs. RW) indicate relative 
% performance.yhat_rw = repmat(x(T_train, :), H, 1);
squared_errors_rw = (x_test(:, key_vars) - yhat_rw(:, key_vars)).^2;
MSFE_rw_all = mean(squared_errors_rw, 1);
MSFE_ratio_rw = MSFE_all ./ MSFE_rw_all;

% --- AR(1) Benchmark ---
% Purpose: Compute MSFE for an AR(1) forecast.
% Explanation: Fits an AR(1) model to each key variable’s training data and
% forecasts H steps ahead. MSFE ratios (DFTLTA vs. AR(1)) assess DFTLTA’s 
% performance against a simple time series model.
clear forecast
ar1_errors = zeros(H, length(key_vars));
for k = 1:length(key_vars)
    y_train = x(1:T_train, key_vars(k));
    ar_model = arima(1,0,0);
    est_model = estimate(ar_model, y_train, 'Display', 'off');
    y_forecast = forecast(est_model, H, 'Y0', y_train);
    ar1_errors(:,k) = (x_test(:, key_vars(k)) - y_forecast).^2;
end
MSFE_ar1 = mean(ar1_errors, 1);
MSFE_ratio_ar1 = MSFE_all ./ MSFE_ar1;


% --- Diebold-Mariano Tests ---
% Purpose: Test whether TL forecasts are significantly more accurate than
% benchmarks.
% Explanation: The Diebold-Mariano (DM) test compares squared forecast 
% errors: DM = mean(d_t) / (std(d_t) / sqrt(n)), where d_t is the 
% difference in squared errors. A two-sided p-value is computed using a 
% t-distribution.
n = H;  % number of out‐of‐sample forecasts

for k = 1:length(key_vars)
    % 1) DFTL vs Naive RW
    d_rw = squared_errors(:,k) - squared_errors_rw(:,k);
    dbar = mean(d_rw);
    sd = std(d_rw,1);           % use 1/N normalization for variance
    DM_rw = dbar / (sd/sqrt(n));
    p_rw = 2*(1 - tcdf(abs(DM_rw), n-1));
    
    % 2) DFTL vs AR(1)
    d_ar1 = squared_errors(:,k) - ar1_errors(:,k);
    dbar1 = mean(d_ar1);
    sd1 = std(d_ar1,1);
    DM_ar1 = dbar1 / (sd1/sqrt(n));
    p_ar1 = 2*(1 - tcdf(abs(DM_ar1), n-1));
    
    fprintf('DM test (%s vs RW):  DM=%.3f, p=%.3f\n', var_names{k}, ...
        DM_rw, p_rw);
    fprintf('DM test (%s vs AR1): DM=%.3f, p=%.3f\n\n', var_names{k}, ...
        DM_ar1, p_ar1);
end

% --- Forecast-Encompassing Tests ---
% Purpose: Test whether TL forecasts encompass RW or AR(1) forecasts.
% Explanation: Regresses the forecast error (y - f_TL) on the difference
% (f_benchmark - f_TL). A non-significant coefficient suggests TL 
% encompasses the benchmark, containing all relevant information. 
nObs = H;                      % number of forecasts
dfree = nObs - 2;              % df for t‐tests

for k = 1:length(key_vars)
    y   = x_test(:, key_vars(k));       % actuals
    f0  = yhat(:,   key_vars(k));       % DFTL forecast
    f_rw= yhat_rw(:,key_vars(k));       % RW forecast
    f_a1= y_forecast;      % AR(1) forecast

    % --- Test 1: Does DFTL encompass RW? regress (y - f0) on (f_rw - f0)
    d_rw = y - f0;
    X_rw = [ones(nObs,1), (f_rw - f0)];
    beta_rw = (X_rw'*X_rw)\(X_rw'*d_rw);
    res_rw  = d_rw - X_rw*beta_rw;
    sigma2_rw = (res_rw'*res_rw)/dfree;
    Vbeta_rw  = sigma2_rw * inv(X_rw'*X_rw);
    t_rw      = beta_rw(2)/sqrt(Vbeta_rw(2,2));
    p_rw      = 2*(1 - tcdf(abs(t_rw), dfree));
    
    % --- Test 2: Does DFTL encompass AR(1)? regress (y - f0) on (f_a1 - f0)
    d_a1 = y - f0;
    X_a1 = [ones(nObs,1), (f_a1 - f0)];
    beta_a1 = (X_a1'*X_a1)\(X_a1'*d_a1);
    res_a1  = d_a1 - X_a1*beta_a1;
    sigma2_a1 = (res_a1'*res_a1)/dfree;
    Vbeta_a1  = sigma2_a1 * inv(X_a1'*X_a1);
    t_a1      = beta_a1(2)/sqrt(Vbeta_a1(2,2));
    p_a1      = 2*(1 - tcdf(abs(t_a1), dfree));

    fprintf('\nEncompassing tests for %s:\n', var_names{k});
    fprintf(' DFTL vs RW:  \tβ̂=%.3f, t=%.2f, p=%.3f\n', beta_rw(2), ...
        t_rw,   p_rw);
    fprintf(' DFTL vs AR1: \tβ̂=%.3f, t=%.2f, p=%.3f\n', beta_a1(2), ...
        t_a1, p_a1);
end
% --- Ljung-Box Test ---
% Purpose: Test for autocorrelation in forecast residuals.
% Explanation: The Ljung-Box test computes a Q-statistic for residual 
% autocorrelations up to maxLags, testing the null hypothesis of no 
% autocorrelation against the alternative of serial correlation.
if H > 1
maxLags = ceil(H/3);      % how many lags to test
alpha   = 0.05;    % significance level

% Compute residuals matrix (H×4)
residuals = x_test(:, key_vars) - yhat(:, key_vars);

for k = 1:length(key_vars)
    res_k = residuals(:,k);
    n     = length(res_k);

    % compute autocorrelations up to maxLags
    acfAll = autocorr(res_k, 'NumLags', maxLags);
    % autocorr returns [lag0, lag1, …, lagMaxLags]
    rho = acfAll(2:end);   % drop lag-0

    % Ljung–Box Q statistic
    Q = n*(n+2) * sum( rho.^2 ./ (n - (1:maxLags))' );
    pValue = 1 - chi2cdf(Q, maxLags);

    % decision
    if pValue < alpha
        verdict = 'Reject H_0 → autocorrelation';
    else
        verdict = 'Fail to reject H_0';
    end

    fprintf('%s: Q(%d)=%.2f, p=%.3f → %s\n', ...
        var_names{k}, maxLags, Q, pValue, verdict);
end
else
end
% Helper for printing
function s = ternary(cond, a, b)
    if cond, s = a; else s = b; end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Visualization 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Visualize forecasting performance and model diagnostics.
% Explanation: Creates a multi-panel figure showing MSFE, actual vs. 
% forecasted values, factor time series, cross-correlations, and MSFE 
% ratios, aiding interpretation of DFTLTA performance and factor dynamics.

% --- Cross-Correlation of Factors ---
% Purpose: Identify the most correlated pair of factors.
% Explanation: Computes the correlation matrix of estimated factors and 
% finds the pair with the highest absolute correlation, useful for 
% understanding factor relationships.
corr_matrix = corr(Fhat_train);
lower_tri = tril(corr_matrix, -1);
[~, max_idx] = max(abs(lower_tri(:)));
[i, j] = ind2sub(size(lower_tri), max_idx);
disp(['Most correlated factors: ', num2str(i), ' & ', num2str(j)]);

max_lags = 10;
[cross_corr, lags] = xcorr(Fhat_train(:,i), Fhat_train(:,j), max_lags, ...
    'coeff');

figure('Position', [100, 100, 1200, 800]);

% MSFE over forecast horizon
subplot(3,4,1);
plot(1:H, MSFE_horizon, '-o', 'LineWidth', 1.5);
title('MSFE by Horizon'); xlabel('Horizon'); ylabel('MSFE'); grid on;

% Actual vs Forecast for key variables
for idx = 1:4
    subplot(3,4,1+idx);
    plot(1:T_train, x(1:T_train, key_vars(idx)), 'b-', 'DisplayName', ...
        'Train Actual'); hold on;
    plot(T_train+1:T_train+H, x_test(:, key_vars(idx)), 'k-', ...
        'DisplayName', 'Test Actual');
    plot(T_train+1:T_train+H, yhat(:, key_vars(idx)), 'r--', ...
        'DisplayName', 'Forecast');
    hold off; title(var_names{idx});
    xlabel('Time'); ylabel('Value'); grid on;
    if idx==1, legend('Location', 'Best'); end
end

% Histogram of MSFE across variables
subplot(3,4,6);
histogram(MSFE_all, 10);
title('MSFE Distribution'); xlabel('MSFE'); ylabel('Frequency'); grid on;

% Time series of factors
subplot(3,4,7);
plot(Fhat_train, 'LineWidth', 1.5);
title('Estimated Factors over Time'); xlabel('Time'); ylabel('Value');
legend(arrayfun(@(k)['Factor ' num2str(k)], 1:size(Fhat_train,2), ...
    'UniformOutput', false), 'Location', 'Best'); grid on;

% Cross-correlogram of most correlated pair
subplot(3,4,8);
stem(lags, cross_corr, 'LineWidth', 1.5);
title(['Cross-Correlogram: Factor ' num2str(i) ' vs ' num2str(j)]);
xlabel('Lag'); ylabel('Corr'); grid on;

sgtitle(sprintf('LSFM Forecasting: p=%d', p_opt));

% MSFE Ratio Bar Chart
subplot(3,4,9);
bar([MSFE_ratio_rw; MSFE_ratio_ar1]');
title('MSFE Ratios: DFTL vs RW and AR(1)');
xticklabels(var_names); ylabel('Ratio');
legend('DFTL / RW', 'DFTL / AR(1)', 'Location', 'Best'); grid on;

sgtitle(sprintf('LSFM Forecasting Performance: p=%d', p_opt));

% MSFE per Horizon for Key Variables (Individual)
subplot(3,4,10);
plot(1:H, squared_errors(:,1), '-o', 'LineWidth', 1.3); hold on;
plot(1:H, squared_errors(:,2), '-s', 'LineWidth', 1.3);
plot(1:H, squared_errors(:,3), '-d', 'LineWidth', 1.3);
plot(1:H, squared_errors(:,4), '-^', 'LineWidth', 1.3);
hold off;
title('MSFE per Horizon for Key Variables');
xlabel('Horizon'); ylabel('Squared Error');
legend(var_names, 'Location', 'Best'); grid on;

sgtitle(sprintf('LSFM Forecasting: p=%d', p_opt));

disp('Forecasting and visualization complete.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Main Forecast Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Forecasts H-step-ahead observations via VAR on latent factors
function [yhat] = LSFM_Forecast(Fhat, Lhat, T, N, H, p)
    q = size(Fhat, 2);
    if p >= T, p = floor(T/2); end
    mdl = varm(q, p);
    EstMdl = estimate(mdl, Fhat);
    Ff = forecast(EstMdl, H, Fhat);
    Lambda_T = squeeze(Lhat(:,:,T));
    yhat = zeros(H, N);
    for h = 1:H
        yhat(h,:) = (Lambda_T * Ff(h,:)')';
    end
end
