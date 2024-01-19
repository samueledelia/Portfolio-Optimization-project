function [w_MVP, w_MaxSharpe, P, min_vola, mvp_ret] = MVPandMaxSharpe(p, LogRet)

% function used to compute weights for MVP and Max Sharpe portfolio
%
% INPUTS: 
% p -> portfolio object with constraint
% LogRet -> Log Returns
%
% OUTPUTS:
% w_MVP -> MVP portfolio weights
% w_MaxSharpe -> weights of the portfolio with the highest Sharpe Ratio
% P -> portfollio object with moments 
% min_vola -> lowest volatility value
% mvp_return -> return of the MVP portfolio

P = estimateAssetMoments(p, LogRet, 'MissingData',false);

% Estimate efficient frontier weights
pwgt = estimateFrontier(P,1000);

% Estimate portfolio moments for the efficient frontier
[pf_Risk, pf_Ret] = estimatePortMoments(P,pwgt);

% Find minimum variance portfolio
[min_vola, idx] = min(pf_Risk);
mvp_ret = pf_Ret(idx);

w_MVP = pwgt(:,idx);                                     % Weights MVP

% Estimate Maximum Sharpe Ratio Portfolio
w_MaxSharpe = estimateMaxSharpeRatio(P,Method='iterative');    % Weights Max Sharpe

end