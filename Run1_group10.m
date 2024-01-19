%% Group 10
% Portfolio Optimization Project Computational Finance - A.Y. 2023-2024
%
% D'Elia Samuele (samuele.delia@mail.polimi.it)
% Lefosse Lorenzo Saverio (saveriolorenzo.lefosse@mail.polimi.it)
% Segarini Simone (simonedamiano.segarini@mail.polimi.it)
% Monti Filippo (filippo2.monti@mail.polimi.it)
% Pizzo Luca(luca.pizzo@mail.polimi.it)

clear all 
close all
warning off
clc
rng(42)                                  % we fix the seed

%% PART A
% Read Prices
filename = 'prices_fin.xlsx';
table_prices = readtable(filename);
% Transform prices from table to timetable

dt = table_prices(:,1).Variables;
values = table_prices(:, 2:end).Variables;
nm = table_prices.Properties.VariableNames(2:end);

myPrice_dt = array2timetable(values, 'RowTimes', dt, 'VariableNames', nm);

% Define the start and end dates for the subset
start_dt = datetime('11/05/2021', 'InputFormat', 'dd/MM/yyyy');
end_dt = datetime('11/05/2022', 'InputFormat', 'dd/MM/yyyy');

% Create a timerange object with closed boundaries using start and end dates
rng = timerange(start_dt, end_dt, 'Closed');
% Extract a subset of the original time table (myPrice_dt) based on the timerange
subsample = myPrice_dt(rng,:);                                  

% Extract price values and dates from the subset
price_val = subsample.Variables;
dates = subsample.Time;

% Computing Returns and Variance-Covariance matrix
LogRet = tick2ret(price_val,"Method","continuous"); % Daily logreturn
ExpRet = mean(LogRet);                              % Expected return for asset

V = cov(LogRet);                                    % Covariance matrix
var = var(LogRet);                                  % Total variance for asset

%% POINT 1: Solve the Minimun Variance Portfolio and the Maximum Sharpe Ratio Portfolio
% Create Portfolio object with the given asset list
p = Portfolio('AssetList',nm);

% Set default portfolio constraints for a long-only, fully invested portfolio.
p = setDefaultConstraints(p);

% Find MVP and MaxSharpe portfolios
[w_Port_A, w_Port_B, P, min_vola, mvp_ret] = MVPandMaxSharpe(p, LogRet);

% Calculate Sharpe Ratio portfolio return and volatility
pf_SharpeRet = w_Port_B'*ExpRet';
pf_SharpeVol = sqrt(w_Port_B'*V*w_Port_B);

%% Result Point 1: plot and summary table
figure;
plotFrontier(P); hold on;
plot(min_vola, mvp_ret,'r*');
hold on 
plot(pf_SharpeVol, pf_SharpeRet,'g*');
legend('','Minimum Variance Portfolio', 'Maximum Sharpe Ratio','Location', 'southeast');

% Create a table to store the results
result_1 = table(nm', w_Port_A, w_Port_B);
% Rename the columns for clarity
result_1.Properties.VariableNames = {'Asset', 'Weight_MVP_Port_A', 'Weight_MaxSharpe_Port_B'};
% Display the table
disp("Result Point 1:");
disp(result_1);
%% POINT 2: Solve the MV Port. and the MaxSR Port. with additional constraint on the weights
sectors = readtable('sectors_fin.xlsx');

% Create logical matrix
global G_Consumer;
global G_Industrial;
global G_Null;

G_Consumer = strcmp(sectors.Sector,'Consumer Discretionary')';
G_Industrial = strcmp(sectors.Sector,'Industrials')';

% Count the number of companies in each sector
sectorCounts = varfun(@length, sectors, 'GroupingVariables', 'Sector', 'OutputFormat', 'table');
sectorCounts.Properties.VariableNames{'GroupCount'} = 'CompanyCount';

% Create a logical matrix for sectors with less than 5 companies
G_Null = ismember(sectors.Sector, sectorCounts(sectorCounts.CompanyCount < 5, :).Sector)';

% Create Portfolio with additional constraint
p_2 = Portfolio('AssetList',nm);
p_2 = setDefaultConstraints(p_2);                               % Set default portfolio constraints
p_2 = addGroups(p_2, G_Consumer,0.15);                          % 'Consumer Discretionary' weight limit of 0.15
p_2 = addGroups(p_2, G_Industrial,[],0.05);                     % 'Industrials' weight limit of 0.05
p_2 = addGroups(p_2, G_Null,0,0);                               % sectors with less than 5 companies are not count  

[w_Port_C, w_Port_D, P_2, min_vola2, mvp_ret2] = MVPandMaxSharpe(p_2, LogRet);

% MV Portfolio (PORTFOLIO C)
pf_SharpeRet_2 = w_Port_D'*ExpRet';
pf_SharpeVol_2 = sqrt(w_Port_D'*V*w_Port_D);

%% Result Point 2: plot and summary table
figure;
plotFrontier(P_2); hold on;
plot(min_vola2, mvp_ret2,'r*');
hold on 
plot(pf_SharpeVol_2, pf_SharpeRet_2,'g*');
legend('','Minimum Variance Portfolio', 'Maximum Sharpe Ratio','Location', 'southeast')

result_2 =  table(p_2.AssetList',w_Port_C,w_Port_D, sectors.Sector);
result_2.Properties.VariableNames = {'Asset', 'Weight_MVP_Port_C', 'Weight_MaxSharpe_Port_D', 'Sector'};
disp("Result Point 2:");
disp(result_2);

%% Display the sector-wise composition of the portfolios from Point 2
% Result for MVP (PORTFOLIO C)
sector = sectors.Sector;
data_MVP_2 = table(w_Port_C, sector);

% Convert sector to categorical for grouping
data_MVP_2.sector = categorical(data_MVP_2.sector);

% Use groupsummary to calculate the sum for each sector
result_MVP_2_sector = groupsummary(data_MVP_2, 'sector', 'sum', 'w_Port_C');

% Display the results
disp("Minimun Variance Portfolio by sector:");
disp(result_MVP_2_sector);

% Result for MaxSharpeRatio (PORTFOLIO D)
sector = sectors.Sector;
data_MaxSR_2 = table(w_Port_D, sector);
data_MaxSR_2.sector = categorical(data_MaxSR_2.sector);
result_MaxSR_2_sector = groupsummary(data_MaxSR_2, 'sector', 'sum', 'w_Port_D');
disp("Maximum Sharpe Ratio by sector:");
disp(result_MaxSR_2_sector);

%% POINT 3: Solve MVP and MaxSR Port. of Point 1,2 using the resampling method

% Number of simulations
N = 50;

% Create Portfolio object with the given asset list with default constraints
p_3 = Portfolio('AssetList',nm);
p_3 = setDefaultConstraints(p_3);

constraint = false;

[MVP_SimA, MaxSharpe_SimA, pfSimA] = Robust_MVPandMaxSharpe(p_3, ExpRet, V, LogRet, N, constraint);

w_Port_E = MVP_SimA.w;                          % MV Portfolio without constraints
w_Port_G = MaxSharpe_SimA.w;                    % Max Sharpe Ratio Portfolio without constraints

%% Additional constraint 
N = 10;
p_3B = Portfolio('AssetList',nm);
p_3B = setDefaultConstraints(p_3B);
p_3B = addGroups(p_3B, G_Consumer,0.15);       % 'Consumer Discretionary' weight limit of 0.15
p_3B = addGroups(p_3B, G_Industrial,0,0.05);   % 'Industrials' weight limit of 0.05
p_3B = addGroups(p_3B, G_Null,0,0);            % sectors with less than 5 companies are not count                         

constraint = true;

[MVP_SimB, MaxSharpe_SimB, pfSimB] = Robust_MVPandMaxSharpe(p_3B, ExpRet, V, LogRet, N, constraint);     

w_Port_F = MVP_SimB.w;                         % MV Portfolio with constraints
w_Port_H = MaxSharpe_SimB.w;                   % Max Sharpe Ratio Portfolio with constraints

%% Result Point 3: plot and summary table
figure;
plot(mean(pfSimA.risk,2), mean(pfSimA.ret,2)); hold on; 
plot(MVP_SimA.risk, MVP_SimA.ret,'y*')
hold on
plot(MaxSharpe_SimA.risk, MaxSharpe_SimA.ret,'c*')
legend('Robust Frontier','Minimum Variance Portfolio', 'Maximum Sharpe Ratio','Location', 'southeast')
title('Robust w/Resampling')

result_3A =  table(p_3.AssetList',w_Port_E,w_Port_G, sectors.Sector);
result_3A.Properties.VariableNames = {'Asset', 'Weight_MVP_Port_E', 'Weight_MaxSharpe_Port_G', 'Sector'};

figure;
plot(mean(pfSimB.risk,2), mean(pfSimB.ret,2)); hold on; 
plot(MVP_SimB.risk, MVP_SimB.ret,'y*')
hold on
plot(MaxSharpe_SimB.risk, MaxSharpe_SimB.ret,'c*')
legend('Robust Frontier','Minimum Variance Portfolio', 'Maximum Sharpe Ratio','Location', 'southeast')
title('Robust w/Resampling & additional constraints')

result_3B =  table(p_3B.AssetList',w_Port_F,w_Port_H, sectors.Sector);
result_3B.Properties.VariableNames = {'Asset', 'Weight_MVP_Port_F', 'Weight_MaxSharpe_Port_H', 'Sector'};
%% POINT 4: Solve MVP and MaxSR Portf. using Black-Litterman model

% Get the number of assets and initialize variables
numAssets = size(LogRet, 2);
tau = 1 / length(LogRet);
v = 3; 
P = zeros(v, numAssets);
q = zeros(v, 1);
Omega = zeros(v);

% View 1: Companies in the "Consumer Staples" sector will have a 7% annual return
P(1, strcmp(sectors.Sector, 'Consumer Staples')) = 1;
q(1) = 0.07;

% View 2: Companies in the "Healthcare" sector will have a 3% annual return
P(2, strcmp(sectors.Sector, 'Health Care')) = 1;
q(2) = 0.03;

% View 3: "Communication Services" will outperform "Utilities" by 4%
P(3, strcmp(sectors.Sector, 'Communication Services')) = 1;
P(3, strcmp(sectors.Sector, 'Utilities')) = -1;
q(3) = 0.04;

% Compute Omega matrix
Omega(1, 1) = tau * P(1, :) * V * P(1, :)';
Omega(2, 2) = tau * P(2, :) * V * P(2, :)';
Omega(3, 3) = tau * P(3, :) * V * P(3, :)';

% Adjust views and Omega for daily returns
bizyear2bizday = 1 / 252;
q = q * bizyear2bizday;
Omega = Omega * bizyear2bizday;

% Views distribution
X_views = mvnrnd(q, Omega, 200);
%% Market implied return
cap = readtable('market_cap_fin.xlsx');
cap = sortrows(cap, 'Ticker');
wMkt = cap.MarketCap / sum(cap.MarketCap);
lambda = 1.2;                                   % Scaling factor
mu_mkt = lambda * V * wMkt;
C = tau * V;

% prior distribution
X = mvnrnd(mu_mkt, C, 200);

%% Black Litterman Model

muBL = inv(inv(C) + P' * inv(Omega) * P) * (P' * inv(Omega) * q + inv(C) * mu_mkt);
covBL = inv(P' * inv(Omega) * P + inv(C));

% Display the results in a table
table(nm', mu_mkt * 252, muBL * 252, 'VariableNames', ["Asset Names", "Prior Belief of Exp Ret", "BL ExpRet"])

% Black Litterman return Distribution
XBL = mvnrnd(muBL, covBL, 200);

%% Expected return distribution Plot
figure;
histogram(XBL);
legend('Views Distribution','Prior Distribution','BL Distribution');
title('Expected Return Distribution');
xlabel('Expected Returns');
ylabel('Frequency');

%% Solve the two portfolios using Black-Litterman model
pBL = Portfolio('AssetList', nm);
pBL = setAssetMoments(pBL, muBL, V + covBL);
pBL = setDefaultConstraints(pBL);
pwgt_BL = estimateFrontier(pBL, 1000);
[pf_Risk_BL, pf_Ret_BL] = estimatePortMoments(pBL, pwgt_BL);

% MV Portfolio (PORTFOLIO I)
[min_vola, idx_BL] = min(pf_Risk_BL);
w_Port_I = pwgt_BL(:, idx_BL);

% MaxSR Portfolio (PORTFOLIO L)
w_Port_L = estimateMaxSharpeRatio(pBL, 'Method', 'iterative');
pf_SharpeRet_BL = w_Port_L' * muBL;
pf_SharpeVol_BL = sqrt(w_Port_L' * (V + covBL) * w_Port_L);

%% Result Point 4: plot and summary table
figure;
plotFrontier(pBL); hold on;
plot(min_vola, pf_Ret_BL(idx_BL),'r*');
hold on 
plot(pf_SharpeVol_BL, pf_SharpeRet_BL,'g*');
legend('','Minimum Variance Portfolio', 'Maximum Sharpe Ratio','Location', 'southeast');


result_4 =  table(pBL.AssetList',w_Port_I,w_Port_L, sectors.Sector);
result_4.Properties.VariableNames = {'Asset', 'Weight_MVP_Port_I', 'Weight_MaxSharpe_Port_L', 'Sector'};
disp("Result Point 4:");
disp(result_4);

%% POINT 5: Solve the Maximum Diversified Portfolio and the Maximum Entropy Portfolio under constraint

% Define risk measures as functions
DR = @(x) (x' * sqrt(var)') / sqrt(x' * V * x);
ENTR = @(x) -x' * log(x);
ENTR_INASS = @(x) -sum(((x.^2).*(std(LogRet)'.^2))./sum((x.^2).*(std(LogRet)'.^2)).*log(((x.^2).*(std(LogRet)'.^2))./sum((x.^2).*(std(LogRet)'.^2))));


% Create Portfolio object
p_5 = Portfolio('AssetList', nm);
p_5 = setDefaultConstraints(p_5);

% Estimate asset moments for the portfolio
p_E_n = estimateAssetMoments(p_5, LogRet, 'MissingData', false);

% Estimate the efficient frontier
pwgt_E = estimateFrontier(p_E_n, 1000);

% Create portfolios with custom objectives
P_ENTR = estimateCustomObjectivePortfolio(p_5, ENTR, ObjectiveSense = "maximize");

% Maximum Diversified Portfolio
w_Port_N = estimateCustomObjectivePortfolio(p_5, ENTR_INASS, ObjectiveSense = "maximize");
% Maximum Entropy (in asset volatility) Portfolio
w_Port_M = estimateCustomObjectivePortfolio(p_5, DR, ObjectiveSense = "maximize");

%% Result Point 5: plot and summary table
figure;
plotFrontier(p_E_n)
hold on;
plot(sqrt(w_Port_M'*V*w_Port_M), w_Port_M'*ExpRet','r*')
hold on 
plot(sqrt(P_ENTR'*V*P_ENTR), P_ENTR'*ExpRet','g*')
hold on 
plot(sqrt(w_Port_N'*V*w_Port_N), w_Port_N'*ExpRet','b*')
legend('','Maximum Diversified Portfolio', 'Maximum Entropy Portfolio', 'Maximum Entropy (in volatility) Portfolio','Location', 'southeast');

% summary table
result_5 =  table(p_5.AssetList',w_Port_M,w_Port_N, sectors.Sector);
result_5.Properties.VariableNames = {'Asset', 'Weight_MaxDiv_Port_M', 'Weight_MaxEntropy_Port_N', 'Sector'};
disp("Result Point 5:");
disp(result_5);
%% POINT 6: Solve the Maximum expected return portfolio usign the Principal Component Analysis under constraint
% Set the number of principal components
k = 10;

% Perform Principal Component Analysis (PCA)
[factorLoading, factorRetn, latent] = pca(LogRet, 'NumComponents', k);

% Calculate Total Variance
ToTVar = sum(latent);

% Calculate Explained Variance for each principal component
ExplainedVar = latent(1:k) / ToTVar;

% List of components for cumulative explained variance calculation
n_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

% Initialize an array to store Cumulative Explained Variance
CumExplainedVar = zeros(1, size(n_list, 2));

% Calculate Cumulative Explained Variance for different numbers of components
for i = 1:size(n_list, 2)
    n = n_list(i);
    CumExplainedVar(1, i) = getCumulativeExplainedVar(latent, n);
end
%% Plot of the PCA results

% Create a bar chart to visualize the percentage of explained variances for each principal component
h = figure();
bar(n_list,ExplainedVar)
title('Percentage of Explained Variances for each principal component')
xlabel('Principal Components')
ylabel('Percentage of Explained Variance')

% Create a line plot and scatter plot to show the cumulative percentage of explained variances
f = figure();
plot(n_list,CumExplainedVar,'m')
title('Total Percentage of Explained Variances for the first n-components')
hold on
scatter(n_list,CumExplainedVar,'m','filled')
grid on 
xlabel('Total number of Principal Components')
ylabel('Percentage of Explained Variances')
hold off
%% Solve the Maximum expected return portfolio
% Reconstruct asset returns
covarFactor = cov(factorRetn);
reconReturn = factorRetn*factorLoading' + ExpRet;
unexplainedRetn = LogRet - reconReturn;

unexplainedCovar = diag(cov(unexplainedRetn));
D = diag(unexplainedCovar);
covarAsset = factorLoading*covarFactor*factorLoading' + D;

% Portfolio Optimization 
targetRisk = 0.007;  
tRisk = targetRisk*targetRisk;  
meanStockRetn = ExpRet;

optimProb = optimproblem('Description','Portfolio with factor covariance matrix','ObjectiveSense','max');
wgtAsset = optimvar('asset_weight', numAssets, 1, 'Type', 'continuous', 'LowerBound', 0, 'UpperBound', 1);
wgtFactor = optimvar('factor_weight', k, 1, 'Type', 'continuous');

optimProb.Objective = sum(meanStockRetn'.*wgtAsset);

optimProb.Constraints.asset_factor_weight = factorLoading'*wgtAsset - wgtFactor == 0;
optimProb.Constraints.risk = wgtFactor'*covarFactor*wgtFactor + wgtAsset'*D*wgtAsset <= tRisk;
optimProb.Constraints.budget = sum(wgtAsset) == 1;

x0.asset_weight = ones(numAssets, 1)/numAssets;
x0.factor_weight = zeros(k, 1);
opt = optimoptions("fmincon", "Algorithm","sqp", "Display", "off", ...
    'ConstraintTolerance', 1.0e-8, 'OptimalityTolerance', 1.0e-8, 'StepTolerance', 1.0e-8);
x = solve(optimProb,x0, "Options",opt);

% Max ExpRet Portfolio (PORTFOLIO P)
w_Port_P = x.asset_weight;

%% Result Point 6: plot and summary table
result_6 =  table(p_5.AssetList', w_Port_P, sectors.Sector);
result_6.Properties.VariableNames = {'Asset', 'Weight_MaxExpRet_Port_P', 'Sector'};
disp("Result Point 6:");
disp(result_6);

%% Point 7: Solve Maximum Expected Shortfall-modified Sharpe Ratio under standard constraint
alpha = 0.05;
Mod_ES = @(x) (ExpRet*x)/(mean(LogRet*x)+std(LogRet*x)*(pdf('normal',norminv(1-alpha),0,1))/alpha);
p_7 = Portfolio('AssetList', nm);

p_7 = setDefaultConstraints(p_7);

% the Portfolio that maximizes the Expected Shortfall-modified Sharpe Ratio
w_Port_Q = estimateCustomObjectivePortfolio(p_7, Mod_ES, ObjectiveSense = "maximize");
p_7 = estimateAssetMoments(p_7, LogRet, 'MissingData', false);

%% Result Point 7: Plot and Summary Table
% Plot the efficient frontier
figure;
plotFrontier(p_7);
hold on 
plot(sqrt(w_Port_Q'*V*w_Port_Q), w_Port_Q'*ExpRet','g*')
legend('','Maximized ES-Modified Sharpe Ratio Portfolio','Location', 'southeast');

% summary table
result_7 =  table(p_7.AssetList', w_Port_Q, sectors.Sector);
result_7.Properties.VariableNames = {'Asset', 'Weight_MaxExpRet_Port_Q', 'Sector'};
disp("Result Point 7:");
disp(result_7);

%% Point 8: Compare the performance of the computed portfolio against equally weighted portfolio using backtesting
% Solve Equally Weighted Portfolio
% Equally weighted portfolio
num_assets = numel(nm);
w_Port_EW = repmat(1/num_assets, num_assets, 1);
% Calculate portfolio returns and volatility
pf_equal_Ret = w_Port_EW' * ExpRet';
pf_equal_Vol = sqrt(w_Port_EW' * V * w_Port_EW);

ret = price_val(2:end,:)./price_val(1:end-1,:);
% Names of Portfolios
names = {'A','B','C','D','E','F','G','H','I','L','M','N','P','Q','EW'};

% Initialize Equity Matrix and Performance Table
equity_matrix = zeros(length(ret), 15);
perf_table = table('Size',[15 5],'VariableTypes', {'double','double','double','double','double'}, 'VariableNames',{'AnnRet','AnnVol','Sharpe','MaxDD','Calmar'},'RowNames',names);


for i = 1:length(names)
    equity_temp = cumprod(ret*eval(['w_Port_', cell2mat(names(i))]));
    equity_matrix(:,i) = 100.*equity_temp/equity_temp(1);
    [annRet, annVol, Sharpe, MaxDD, Calmar] = getPerformanceMetrics(equity_temp);
    perf_table(i,:) = table(annRet, annVol, Sharpe, MaxDD, Calmar);
end

%% Plot Equity Line 
% We plot the portfolio performance graph
figure;
plot(dates(2:end,1),equity_matrix)
grid on
legend(names,'Location', 'northwest')
disp(perf_table)
title('Equity Line')


%% Part B: Evaluate the performance of the portfolios in the period 12/05/2022-12/05/2023

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
equity_matrix_new = zeros(length(ret_new), 15);
perf_table_new = table('Size',[15 5],'VariableTypes', {'double','double','double','double','double'}, 'VariableNames',{'AnnRet','AnnVol','Sharpe','MaxDD','Calmar'},'RowNames',names);

for i = 1:length(names)
    equity_temp = cumprod(ret_new*eval(['w_Port_', cell2mat(names(i))]));
    equity_matrix_new(:,i) = 100.*equity_temp/equity_temp(1);
    [annRet, annVol, Sharpe, MaxDD, Calmar] = getPerformanceMetrics(equity_temp);
    perf_table_new(i,:) = table(annRet, annVol, Sharpe, MaxDD, Calmar);
end

%% Plot Equity Line
% We plot the portfolio performance graph
f2 = figure();
plot(dates_new(2:end,1),equity_matrix_new)
grid on
legend(names,'Location', 'northwest')
disp(perf_table_new)
title('Equity Line')



