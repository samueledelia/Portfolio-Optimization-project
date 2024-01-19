%% We propose a portfolio based on iShares S&P 100 ETF (OEF) 
%
% This code proposes a portfolio based on the iShares S&P 100 ETF (OEF) and 
% evaluates its performance over two distinct periods (11/05/2021-11/05/2022 
% and 12/05/2022-12/05/2023).
clear all 
close all
warning off
clc
rng(42)                                  % Fix the seed

%% Evaluate the performance of the portfolio in the period 11/05/2021-11/05/2022
% Read Prices
filename = 'OEF.csv';
table_prices = readtable(filename);

% Transform prices from table to timetable
dt = table_prices(:, 1).Variables;
values = table_prices(:, 2).Variables;
nm = table_prices.Properties.VariableNames{2}; % Corrected variable name extraction

myPrice_dt = array2timetable(values, 'RowTimes', dt, 'VariableNames', {nm});

% Define the start and end dates for the subset
start_dt = datetime('11/05/2021', 'InputFormat', 'dd/MM/yyyy');
end_dt = datetime('11/05/2022', 'InputFormat', 'dd/MM/yyyy');

% Create a timerange object with closed boundaries using start and end dates
trng = timerange(start_dt, end_dt, 'Closed'); % Renamed variable to avoid conflict
% Extract a subset of the original timetable (myPrice_dt) based on the timerange
subsample = myPrice_dt(trng, :);

% Extract price values and dates from the subset
price_val = subsample.Variables;
dates = subsample.Time;

% Create a portfolio based on the ETF
p = Portfolio('AssetList', {nm}); % Corrected variable type

ret = price_val(2:end, :) ./ price_val(1:end-1, :);
equity_matrix = zeros(length(ret), 1);
perf_table = table('Size', [1 5], 'VariableTypes', {'double', 'double', 'double', 'double', 'double'}, 'VariableNames', {'AnnRet', 'AnnVol', 'Sharpe', 'MaxDD', 'Calmar'}, 'RowNames', {'ETF'});

equity_temp = cumprod(ret*1);
equity_matrix(:) = 100.*equity_temp/equity_temp(1);
[annRet, annVol, Sharpe, MaxDD, Calmar] = getPerformanceMetrics(equity_temp);

perf_table(1, :) = table(annRet, annVol, Sharpe, MaxDD, Calmar);

%% Plot Equity Line 
% We plot the portfolio performance graph
blue  = [21/255, 110/255, 235/255];
figure;
plot(dates(2:end,1),equity_matrix,'Color',blue,'LineWidth',2)
grid on
disp('Result on the fist period');
disp(perf_table)
title('S&P100 ETF Equity Line');



%% Evaluate the performance of the portfolio in the period 12/05/2022-12/05/2023

% Define the start and end dates for the subset
start_dt_new = datetime('12/05/2022', 'InputFormat', 'dd/MM/yyyy');
end_dt_new = datetime('12/05/2023', 'InputFormat', 'dd/MM/yyyy');

% Create a timerange object with closed boundaries using start and end dates
rng_new = timerange(start_dt_new, end_dt_new, 'Closed');
% Extract a subset of the original time table (myPrice_dt) based on the timerange
subsample_new = myPrice_dt(rng_new,:);                                  

% Extract price values and dates from the subset
price_val_new = subsample_new.Variables;
dates_new = subsample_new.Time;
% Names of Portfolios
ret_new = price_val_new(2:end,:)./price_val_new(1:end-1,:);
% Initialize Equity Matrix and Performance Table
equity_matrix_new = zeros(length(ret_new), 1);
perf_table_new = table('Size',[1 5],'VariableTypes', {'double','double','double','double','double'}, 'VariableNames',{'AnnRet','AnnVol','Sharpe','MaxDD','Calmar'},'RowNames',{'ETF'});


equity_temp = cumprod(ret_new*1);
equity_matrix_new(:) = 100.*equity_temp/equity_temp(1);
[annRet, annVol, Sharpe, MaxDD, Calmar] = getPerformanceMetrics(equity_temp);
perf_table_new(1,:) = table(annRet, annVol, Sharpe, MaxDD, Calmar);

%% Plot Equity Line
% We plot the portfolio performance graph
f2 = figure();
plot(dates_new(2:end,1),equity_matrix_new,'Color',blue,'LineWidth',2)
grid on
disp('Result on the second period');
disp(perf_table_new)
title('S&P 100 ETF Equity Line')

