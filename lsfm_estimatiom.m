%% lsfm_estim.m
% Author: Moka Kaleji • Contact: mohammadkaleji1998@gmail.com
% Affiliation: Master Thesis in Econometrics: 
% Advancing High-Dimensional Factor Models: Integrating Time-Varying 
% Loadings and Transition Matrix with Dynamic Factors.
% University of Bologna
% Description:
%   Implements Locally Stationary Factor Model (LSFM) estimation following
%   Motta, Hafner & von Sachs (2011). This script provides an interactive
%   interface to select dataset frequency (monthly or quarterly), specify
%   training horizon, and determine model hyperparameters (number of factors q
%   and bandwidth h). It computes diagnostic measures, performs estimation,
%   and visualizes variance explained by the chosen factors.

clear; close all; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Dataset Selection, Frequency, Training Sample Size, and Standardization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose:
% Allow the user to select the dataset periodicity (monthly or quarterly), 
% load the corresponding data, specify the training sample size, and 
% standardize the data for numerical stability in model estimation.
% Explanation:
% The dynamic factor model requires a multivariate time series dataset. 
% This section provides a user-friendly interface to choose between 
% pre-processed monthly or quarterly datasets, ensuring flexibility in 
% periodicity. The training sample size (T_train) is specified to focus on
% a subset of the data, which is useful for in-sample estimation and 
% out-of-sample forecasting. Standardization (zero mean, unit variance) is
% applied to prevent numerical issues and ensure consistent scaling across 
% variables, a common practice in high-dimensional time series modeling.

% --- Present Available Periodicity Options and Capture User Choice ---
% Purpose: Display a dialog for the user to select dataset periodicity.
% Explanation: The listdlg function provides a graphical interface to choose
% between monthly ('MD1959.xlsx') and quarterly ('QD1959.xlsx') datasets. 
% The selection is validated to ensure a choice is made, halting execution 
% if cancelled to prevent undefined behavior.
options = {'Monthly (MD1959.xlsx)', 'Quarterly (QD1959.xlsx)'};
[choiceIndex, ok] = listdlg('PromptString','Select dataset:',...
                             'SelectionMode','single',...
                             'ListString',options,...
                             'Name','Dataset Selection',...
                             'ListSize',[400 200]);
if ~ok
    error('Dataset selection cancelled. Exiting script.');
end
% --- Load Data Based on Frequency ---
% Purpose: Load the selected dataset from an Excel file and extract the time
% series data.
% Explanation: The filepath is constructed based on the user's choice, 
% pointing to pre-processed datasets stored in a specific directory. The data
% is read into a table using readtable, then converted to a numeric array. 
% For quarterly data, the first column (date index) is excluded, as it is not
% part of the time series. The dimensions T (time points) and N (variables)
% are extracted for subsequent processing.
switch choiceIndex
    case 1                                                                 % Monthly frequency
        filepath = ['/Users/moka/Research/Thesis/Live Project/' ...
            'Processed_Data/MD1959.xlsx'];
        raw = readtable(filepath);
        x = table2array(raw);                                              % Include all series
        T = size(x,1);
    case 2                                                                 % Quarterly frequency
        filepath = ['/Users/moka/Research/Thesis/Live Project/' ...
            'Processed_Data/QD1959.xlsx'];
        raw = readtable(filepath);
        x = table2array(raw(:,2:end));                                     % Drop date index
        T = size(x,1);
    otherwise
        error('Unexpected selection index.');
end
[N_obs, N] = size(x);
% --- Define Training Sample Size ---
% Purpose: Prompt the user to specify the number of observations for the 
% training sample.
% Explanation: The training sample size (T_train) determines the subset of 
% data used for model estimation, allowing the remaining observations for 
% out-of-sample validation or forecasting. A default value of 225 is suggested,
% but the user can input any integer between 1 and T-1. The input is validated
% to ensure it is positive and less than the total number of observations, 
% preventing invalid training periods.
defaultTrain = '225';
prompt = sprintf(['Dataset has %d observations. Enter training size ' ...
    '(T_train):'], T);
userInput = inputdlg(prompt, 'Training Horizon', [3 100], {defaultTrain});
if isempty(userInput)
    error('No training size provided. Exiting.');
end
T_train = str2double(userInput{1});
assert(T_train>0 && T_train<T, 'T_train must be integer in (0, %d)', T);
% --- Standardization ---
% Purpose: Standardize the training data to zero mean and unit variance.
% Explanation: Standardization is critical for numerical stability in 
% high-dimensional factor models, as variables with different scales can lead
% to ill-conditioned matrices or biased factor estimates. The training data
% (first T_train observations) is centered by subtracting the mean and scaled
% by dividing by the standard deviation, computed across the training sample.
% This ensures all variables contribute equally to the factor structure and
% prevents numerical overflow in the EM algorithm.
x_train = x(1:T_train, :);
mean_train = mean(x_train);
std_train  = std(x_train);
x_train_norm = (x_train - mean_train) ./ std_train;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Running LSFM estimation 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
R = 6;
h = 0.15;

[CChat, Fhat_train, Lhat_train] = lsfm(x_train_norm, R, h);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Save Results for Forecasting and Further Analysis 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save('lsfm_estimation_results.mat', 'R', 'h', 'Fhat_train', ...
    'Lhat_train', 'x_train', 'x_train_norm', 'mean_train', 'std_train', ...
    'T_train', 'N');
disp('LSFM estimation complete. Results saved.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Variance Explained by Factors 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Use temporary bandwidth for variance-explained diagnostics
q_max = 15;  % Maximum allowable factors
h_temp = 0.1;
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
var_explained_opt = mean_var_explained(R);
disp(['Average variance explained by q = ', num2str(R), ': ', ...
    num2str(var_explained_opt*100), '%']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Main Function 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Locally Stationary Factor Model (LSFM)
% Purpose:
% Estimates a static factor model at each time point using 
% kernel-weighted PCA to provide initial estimates of time-varying loadings
% (Lambda_t), factors (f_t), and idiosyncratic covariance (Sigma_t).
% Reference:
%   - Dahlhaus, R. (1996)
%   - Motta G, Hafner CM, von Sachs R. (2011)

function [CChat, Fhat, Lhat, Sigmahat] = lsfm(X, R, h)
% Mathematical Formulation:
% For each t, compute a local covariance matrix:
% Sigma_t = sum(w_{t,s} * X_s * X_s'),  w_{t,s} = K((u_s - u_t)/h) / sum(K)
% where K is a Gaussian kernel, u_t = t/T, and h is the bandwidth. 
% Loadings Lambda_t are the top R eigenvectors of Sigma_t, and factors 
% are f_t = (Lambda_t' * Lambda_t)^{-1} * Lambda_t' * X_t.
% Inputs:
%   X: T x N data matrix
%   R: Number of factors
%   h: Bandwidth for kernel smoothing
% Outputs:
%   CChat: T x N common components
%   Fhat: T x R factors
%   Lhat: N x R x T loadings
%   Sigmahat: N x N x T covariance matrices

    % --- Dimensions ---
    [T, N] = size(X);


    % --- Initialize Outputs ---
    CChat = zeros(T, N);
    Fhat = zeros(T, R);
    Lhat = zeros(N, R, T);
    Sigmahat = zeros(N, N, T);

    % --- Compute Kernel Weights and Local Covariance ---
    % Explanation: For each t, compute weights based on temporal proximity 
    % using a Gaussian kernel. The local covariance Sigma_t is a weighted 
    % sum of outer products X_s * X_s', capturing local data structure.
    u = (1:T)' / T;                                                        % Normalized time index
    for t = 1:T                                                            
        u_t = u(t);
        z = (u - u_t) / h;
        weights = (1/sqrt(2*pi)) * exp(-0.5 * z.^2);                       % Gaussian kernel
        weights = weights / sum(weights);                                  % Normalize
        for i = 1:N
            for j = 1:i
                cross_prod = X(:,i) .* X(:,j);
                Sigmahat(i,j,t) = sum(weights .* cross_prod);
                Sigmahat(j,i,t) = Sigmahat(i,j,t);                         % Ensure symmetry
            end
        end
    end

    % --- Eigenvalue Decomposition and Factor Estimation ---
    % Explanation: For each t, compute the top R eigenvectors of Sigma_t to
    % obtain Lambda_t. Factors are estimated by projecting X_t onto Lambda_t,
    % and common components are C_t = Lambda_t * f_t. The sign of loadings 
    % is adjusted for consistency.
    opts.disp = 0;                                                         % Suppress eigs output
    for t = 1:T
        Sigma_t = squeeze(Sigmahat(:,:,t));
        Sigma_t = (Sigma_t + Sigma_t')/2;                                  % Ensure symmetry
        [V,D] = eig(Sigma_t);
        D = max(real(diag(D)),0);                                          % Ensure non-negative eigenvalues
        Sigma_t = V * diag(D) * V';                                        % Reconstruct positive semi-definite matrix
        % Compute eigenvectors and eigenvalues
        [A, D] = eigs(Sigma_t, R, 'largestabs', opts);                     % Top R eigenvectors and eigenvalues
        eigenvalues = diag(D);                                             % Extract eigenvalues as a vector
        sqrt_eigenvalues = sqrt(eigenvalues);                              % Square root of eigenvalues
        
        % Adjust sign of eigenvectors for consistency
        sign_adjust = diag(sign(A(1,:)));
        A_adjusted = A * sign_adjust;
        
        % Initialize loadings and factors per user's request
        Lhat(:,:,t) = A_adjusted .* sqrt_eigenvalues';                     % Lhat = A * sqrt(D), scaling each column
        A_scaled = A_adjusted ./ sqrt_eigenvalues';                        % For Fhat = X * A / sqrt(D)
        Fhat(t,:) = X(t,:) * A_scaled;                                     % Factors scaled inversely
        CChat(t,:) = Fhat(t,:) * Lhat(:,:,t)';                             % Common component
    end
end

%[[[[[  % Extract R largest eigenvectors as loadings
%       [A,~] = eigs(Sigma_t, R, 'largestabs', opts);
%       Lhat(:,:,t) = A * diag(sign(A(1,:)));
        % Compute factors via generalized least squares
%       Fhat(t,:)   = X(t,:) * A;
        % Compute common components
%       CChat(t,:)  = (A * Fhat(t,:)')';
        %C = Fhat' * Fhat / T;           % Factor covariance
        %[V, D] = eig(C);                % Eigen-decomposition
        %D_inv_sqrt = diag(1 ./ sqrt(diag(D)));
        %Fhat = Fhat * V * D_inv_sqrt; % Rotated factors
        %Lhat(:,:,t) = Lhat(:,:,t) * V * inv(D_inv_sqrt);]]]]]
