%% lsfm_forecasting.m
% Author: Moka Kaleji
% Affiliation: Master Thesis in Econometrics, University of Bologna
% Description:
%   Executes forecasting for selected macroeconomic indicators (GDP, Unemployment,
%   Inflation, Interest Rate) using the previously estimated Locally Stationary
%   Factor Model (LSFM). Supports user-defined VAR lag order and computes
%   forecasting performance metrics (MSFE, RMSE) with visualization.

clear; close all; clc;

%% Dataset Selection
% Present dataset frequency options to the user via dialog
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

%% Load Data and Define Key Variables
switch choiceIndex
    case 1  % Monthly data
        filepath = '/Users/moka/Research/Thesis/Live Project/Processed_Data/MD1959.xlsx';
        T = 790;
        tableData = readtable(filepath);
        x = table2array(tableData);
        key_vars = [1, 24, 105, 77];  % Indices for GDP, Unemp, Infl, IntRate
    case 2  % Quarterly data
        filepath = '/Users/moka/Research/Thesis/Live Project/Processed_Data/QD1959.xlsx';
        T = 264;
        tableData = readtable(filepath);
        x = table2array(tableData(:,2:end));  % Drop date column
        key_vars = [1, 58, 116, 136];
    otherwise
        error('Unexpected selection index.');
end
var_names = {'GDP', 'Unemployment', 'Inflation', 'Interest Rate'};

%% Load LSFM Estimation Outputs
% Load precomputed factors, loadings, and scaling parameters
load('lsfm_estimation_results.mat', 'Fhat_train', 'Lhat_train', 'T_train', 'mean_train', 'std_train', 'N');

%% Forecast Settings
H = 8;  % Forecast horizon
p_opt = input('Enter the VAR lag order (p, 1 to 12): ');
assert(p_opt >= 1 && p_opt <= 12, 'p must be integer between 1 and 12');

disp(['Using VAR lag order p = ', num2str(p_opt)]);

%% Prepare Test Sample
x_test = x(T_train+1:T_train+H, :);    % Observations for evaluation
x_test_norm = (x_test - mean_train) ./ std_train;  % Normalize as in training

%% Step: Forecasting via LSFM + VAR
% Forecast standardized observations using factor forecasts
yhat_norm = forecast_LSFM(Fhat_train, Lhat_train, T_train, N, H, p_opt);
% Rescale forecasts to original units
yhat = yhat_norm .* std_train + mean_train;

%% Performance Metrics: MSFE and RMSE
squared_errors = (x_test_norm(:, key_vars) - yhat_norm(:, key_vars)).^2;
MSFE_horizon = mean(squared_errors, 2);  % Across key variables per horizon
MSFE_all = mean(squared_errors, 1);      % Across horizons per variable
MSFE_overall = mean(MSFE_all);

disp(['Overall MSFE (key vars): ', num2str(MSFE_overall)]);
for idx = 1:length(key_vars)
    disp(['MSFE for ', var_names{idx}, ': ', num2str(MSFE_all(idx))]);
end

% Compute root MSFE in original units
RMSE_original = sqrt(MSFE_all) .* std_train(key_vars);
for idx = 1:length(key_vars)
    disp(['RMSE for ', var_names{idx}, ' (original): ', num2str(RMSE_original(idx))]);
end

%% Cross-Correlation Analysis of Latent Factors
corr_matrix = corr(Fhat_train);
lower_tri = tril(corr_matrix, -1);
[~, max_idx] = max(abs(lower_tri(:)));
[i, j] = ind2sub(size(lower_tri), max_idx);
disp(['Most correlated factors: ', num2str(i), ' & ', num2str(j)]);

max_lags = 10;
[cross_corr, lags] = xcorr(Fhat_train(:,i), Fhat_train(:,j), max_lags, 'coeff');

%% Visualization of Forecasting Results
figure('Position', [100, 100, 1200, 800]);

% MSFE over forecast horizon
subplot(3,3,1);
plot(1:H, MSFE_horizon, '-o', 'LineWidth', 1.5);
title('MSFE by Horizon'); xlabel('Horizon'); ylabel('MSFE'); grid on;

% Actual vs Forecast for key variables
for idx = 1:4
    subplot(3,3,1+idx);
    plot(1:T_train, x(1:T_train, key_vars(idx)), 'b-', 'DisplayName', 'Train Actual'); hold on;
    plot(T_train+1:T_train+H, x_test(:, key_vars(idx)), 'k-', 'DisplayName', 'Test Actual');
    plot(T_train+1:T_train+H, yhat(:, key_vars(idx)), 'r--', 'DisplayName', 'Forecast');
    hold off; title(var_names{idx});
    xlabel('Time'); ylabel('Value'); grid on;
    if idx==1, legend('Location', 'Best'); end
end

% Histogram of MSFE across variables
subplot(3,3,6);
histogram(MSFE_all, 10);
title('MSFE Distribution'); xlabel('MSFE'); ylabel('Frequency'); grid on;

% Time series of factors
subplot(3,3,7);
plot(Fhat_train, 'LineWidth', 1.5);
title('Estimated Factors over Time'); xlabel('Time'); ylabel('Value');
legend(arrayfun(@(k)['Factor ' num2str(k)], 1:size(Fhat_train,2), 'UniformOutput', false), 'Location', 'Best'); grid on;

% Cross-correlogram of most correlated pair
subplot(3,3,8);
stem(lags, cross_corr, 'LineWidth', 1.5);
title(['Cross-Correlogram: Factor ' num2str(i) ' vs ' num2str(j)]);
xlabel('Lag'); ylabel('Corr'); grid on;

sgtitle(sprintf('LSFM Forecasting: p=%d', p_opt));

disp('Forecasting and visualization complete.');

%% forecast_LSFM.m (Helper Function)
% Forecasts H-step-ahead observations via VAR on latent factors
function [yhat] = forecast_LSFM(Fhat, Lhat, T, N, H, p)
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
