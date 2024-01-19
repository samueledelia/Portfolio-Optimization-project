%% Compare the performance of the computed portfolio against equally weighted with rebalancing
%
% CAUTION: Execute this secondary script ONLY AFTER completing the execution of Run1_group10
% This script requires a considerable amount of time to execute
%
% We assess the effectiveness of portfolios generated in steps 1-7 by 
% comparing them to an equally weighted portfolio, WITH rebalancing 
% occurring approximately every month.


G_Financials = strcmp(sectors.Sector,'Financials')';

% Rebalance approximately every 1 month (252 / 12 = 21).
rebalFreq = 21;

strat1 = backtestStrategy('Equal Weighted', @(w,TT) equalWeightFcn(w,subsample), ...
    'RebalanceFrequency', rebalFreq, ...
    'LookbackWindow', 0, ...
    'TransactionCosts', 0, ...
    'InitialWeights', w_Port_EW);

% Min Variance Portfolio Point 1
strat2 = backtestStrategy('Min Variance Portfolio', @(w,TT) minVarianceFcn(w,subsample, false), ...
    'RebalanceFrequency', rebalFreq, ...
    'LookbackWindow', 0, ...
    'TransactionCosts', 0, ...
    'InitialWeights', w_Port_A);

% Max Sharpe Ratio Portfolio Point 1
strat3 = backtestStrategy('Max Sharpe Ratio Portfolio', @(w,TT) maxSharpeRatioFcn(w,subsample, false), ...
    'RebalanceFrequency', rebalFreq, ...
    'LookbackWindow', 0, ...
    'TransactionCosts', 0, ...
    'InitialWeights', w_Port_B);
% Min Variance Portfolio Point 2
strat4 = backtestStrategy('Min Variance Portfolio additional constr', @(w,TT) minVarianceFcn(w,subsample, true), ...
    'RebalanceFrequency', rebalFreq, ...
    'LookbackWindow', 0, ...
    'TransactionCosts', 0, ...
    'InitialWeights', w_Port_C);

% Max Sharpe Ratio Portfolio Point 2
strat5 = backtestStrategy('Max Sharpe Ratio Portfolio additional constr', @(w,TT) maxSharpeRatioFcn(w,subsample, true), ...
    'RebalanceFrequency', rebalFreq, ...
    'LookbackWindow', 0, ...
    'TransactionCosts', 0, ...
    'InitialWeights', w_Port_D);
% Min Variance Portfolio Point 4

strat6 = backtestStrategy('Min Variance Portfolio BL', @(w,TT) BLMVP(w,subsample, sectors,wMkt), ...
    'RebalanceFrequency', rebalFreq, ...
    'LookbackWindow', 0, ...
    'TransactionCosts', 0, ...
    'InitialWeights', w_Port_I);
% Max Sharpe Ratio Portfolio Point 4
strat7 = backtestStrategy('Max Sharpe Ratio Portfolio BL', @(w,TT) BLSharpe(w,subsample, sectors,wMkt), ...
    'RebalanceFrequency', rebalFreq, ...
    'LookbackWindow', 0, ...
    'TransactionCosts', 0, ...
    'InitialWeights', w_Port_L);
% Most Diversified Portfolio Point 5
strat8 = backtestStrategy('Most Diversified Portfolio', @(w,TT) MDR(w,subsample, G_Financials), ...
    'RebalanceFrequency', rebalFreq, ...
    'LookbackWindow', 0, ...
    'TransactionCosts', 0, ...
    'InitialWeights', w_Port_M);
% Max Entropy Portfolio Point 5
strat9 = backtestStrategy('Max Entropy Portfolio', @(w,TT) MENTR(w,subsample, G_Financials), ...
    'RebalanceFrequency', rebalFreq, ...
    'LookbackWindow', 0, ...
    'TransactionCosts', 0, ...
    'InitialWeights', w_Port_N);
% PCA Portfolio Point 6
strat10 = backtestStrategy('PCA Portfolio', @(w,TT) PCABack(w,subsample, k), ...
    'RebalanceFrequency', rebalFreq, ...
    'LookbackWindow', 0, ...
    'TransactionCosts', 0, ...
    'InitialWeights', w_Port_P);
% ES Portfolio Point 7
strat11 = backtestStrategy('ES', @(w,TT) ESBack(w,subsample), ...
    'RebalanceFrequency', rebalFreq, ...
    'LookbackWindow', 0, ...
    'TransactionCosts', 0, ...
    'InitialWeights', w_Port_Q);

%% Backtesting from 11/05/2021 to 24/10/2023

% Aggregate the strategy objects into an array.
strategies = [strat1, strat2, strat3, strat4, strat5, strat6, strat7, strat8, strat9, strat10, strat11]; 
% Define a backtest engine
backtester = backtestEngine(strategies);
% Run the backtest
backtester = runBacktest(backtester,subsample); 

% Backtesting result
equityCurve(backtester)
summary(backtester)


%% Backtesting from 12/05/2022-12/05/2023

backtester_1 = runBacktest(backtester,subsample_new); 

% Backtesting result
equityCurve(backtester_1)
summary(backtester_1)
