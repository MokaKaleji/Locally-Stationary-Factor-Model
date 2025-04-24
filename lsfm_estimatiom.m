%% lsfm_estimation.m
% Author: Moka Kaleji
% Affiliation: Master Thesis in Econometrics, University of Bologna
% Description:
%   Implements Locally Stationary Factor Model (LSFM) estimation following
%   Motta, Hafner & von Sachs (2011). This script provides an interactive
%   interface to select dataset frequency (monthly or quarterly), specify
%   training horizon, and determine model hyperparameters (number of factors q
%   and bandwidth h). It computes diagnostic measures, performs estimation,
%   and visualizes variance explained by the chosen factors.

clear; close all; clc;

%% Dataset Selection
% Define available dataset options for periodicity
options = {'Monthly (MD1959.xlsx)', 'Quarterly (QD1959.xlsx)'};

% Create modal dialog to allow user to choose dataset
[choiceIndex, ok] = listdlg(...
    'PromptString', 'Select the dataset to load:',...
    'SelectionMode', 'single',...
    'ListString', options,...
    'Name', 'Dataset Selection',...
    'ListSize', [400 200]);  % Dialog dimensions

% Exit script if user cancels selection
if ~ok
    error('No dataset selected. Exiting...');
end

%% Load Data and Define Dimensions
switch choiceIndex
    case 1  % Monthly frequency data
        filepath = '/Users/moka/Research/Thesis/Live Project/Processed_Data/MD1959.xlsx';
        T = 790;  % Total observations for monthly data
        rawTable = readtable(filepath);
        x = table2array(rawTable);  % Convert table to numeric matrix

    case 2  % Quarterly frequency data
        filepath = '/Users/moka/Research/Thesis/Live Project/Processed_Data/QD1959.xlsx';
        T = 264;  % Total observations for quarterly data
        rawTable = readtable(filepath);
        x = table2array(rawTable(:,2:end));  % Discard date column

    otherwise
        error('Unexpected selection index.');
end

%% Specify Training Sample
% Prompt user to define training horizon T_train
prompt = sprintf('Dataset has %d observations. Enter the number of training periods (T_train):', T);
userInput = inputdlg(prompt, 'Input T_train', [3 100], {'640'});
if isempty(userInput)
    error('Training horizon not specified. Exiting...');
end
T_train = str2double(userInput{1});
if isnan(T_train) || T_train <= 0 || T_train >= T
    error('Invalid T_train: must be integer between 1 and %d.', T-1);
end

%% Hyperparameter Input
% Prompt for factor dimension (q) and kernel bandwidth (h)
q_max = 12;  % Maximum allowable factors
q_opt = input(sprintf('Enter the number of factors (q, 1 to %d): ', q_max));
h_opt = input('Enter the bandwidth (h, 0.05 to 0.25): ');
% Validate user inputs
assert(q_opt >= 1 && q_opt <= q_max, 'q must be integer between 1 and %d', q_max);
assert(h_opt >= 0.05 && h_opt <= 0.25, 'h must lie in [0.05, 0.25]');

%% Split and Standardize Training Data
x_train = x(1:T_train, :);
mean_train = mean(x_train);
std_train  = std(x_train);
x_train_norm = (x_train - mean_train) ./ std_train;

%% Diagnostic: Variance Explained by Factors
% Use temporary bandwidth for variance-explained diagnostics
h_temp = 0.10;
[CChat_temp, ~, ~, Sigmahat] = lsfm(x_train_norm, q_max, h_temp);
% Compute eigenvalues of weighted covariance matrices over time
[Tt, N] = size(x_train_norm);
eigvals = NaN(N, T_train);
for t = 1:T_train
    Sigma_t = squeeze(Sigmahat(:,:,t));
    eigvals(:,t) = sort(real(eig(Sigma_t)), 'descend');
end
% Proportion of variance explained by each eigenvalue
cum_var_explained = cumsum(eigvals, 1) ./ sum(eigvals,1);
mean_var_explained = mean(cum_var_explained, 2);
% Display diagnostic for chosen q_opt
var_explained_opt = mean_var_explained(q_opt);
disp(['Average variance explained by q = ', num2str(q_opt), ': ', num2str(var_explained_opt*100), '%']);

%% Main Estimation: LSFM with User-Specified q and h
[CChat_train, Fhat_train, Lhat_train] = lsfm(x_train_norm, q_opt, h_opt);

%% Visualization of Diagnostics
figure('Position', [100, 100, 1000, 800]);
subplot(3,2,1);
plot(1:q_max, mean_var_explained(1:q_max), 'b-o', 'LineWidth', 1.5); hold on;
plot([1,q_max], [0.6,0.6], 'r--','DisplayName','60% Reference');
title('Variance Explained by Number of Factors');
xlabel('Number of Factors (q)');
ylabel('Average Proportion Explained');
legend('Average','60% Threshold'); grid on;
sgtitle(sprintf('LSFM Estimation Results: q=%d, h=%.2f', q_opt, h_opt));

%% Save Results for Forecasting and Further Analysis
save('lsfm_estimation_results.mat', 'q_opt', 'h_opt', 'Fhat_train', 'Lhat_train', ...
     'x_train', 'x_train_norm', 'mean_train', 'std_train', 'T_train', 'N');
disp('LSFM estimation and diagnostics complete. Results saved.');


%% lsfm.m (Helper Function)
% Estimates local covariance, factors, and loadings for LSFM model
function [CChat, Fhat, Lhat, Sigmahat] = lsfm(X, R, h)
    [T, N] = size(X);
    u = (1:T)' / T;  % Scaled time index in [0,1]
    % Preallocate outputs for efficiency
    CChat   = zeros(T, N);
    Fhat    = zeros(T, R);
    Lhat    = zeros(N, R, T);
    Sigmahat= zeros(N, N, T);
    
    % Compute time-varying covariance estimates
    for t = 1:T
        u_t   = u(t);
        z     = (u - u_t) / h;    % Scaled distance from current time
        weights = (1/sqrt(2*pi)) * exp(-0.5 * z.^2);
        weights = weights / sum(weights);  % Normalize kernel weights
        % Compute weighted covariance entries
        for i = 1:N
            for j = 1:i
                cross_prod = X(:,i) .* X(:,j);
                Sigmahat(i,j,t) = sum(weights .* cross_prod);
                Sigmahat(j,i,t)= Sigmahat(i,j,t);
            end
        end
    end
    
    % Eigen-decomposition and factor extraction at each time point
    opts.disp = 0;  % Suppress eigs display
    for t = 1:T
        Sigma_t = squeeze(Sigmahat(:,:,t));
        Sigma_t = (Sigma_t + Sigma_t')/2;  % Symmetrize
        [V,D] = eig(Sigma_t);
        D = max(real(diag(D)),0);  % Ensure non-negative eigenvalues
        Sigma_t = V*diag(D)*V';
        % Extract R largest eigenvectors as loadings
        [A,~] = eigs(Sigma_t, R, 'largestabs', opts);
        Lhat(:,:,t) = A;
        % Compute factors via generalized least squares
        Fhat(t,:)   = (A' * A) \ (A' * X(t,:)');
        % Compute common components
        CChat(t,:)  = (A * Fhat(t,:)')';
    end
end
