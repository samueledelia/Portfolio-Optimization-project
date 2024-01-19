function [MVP, MaxSharpe, pf] = Robust_MVPandMaxSharpe(p, ExpRet,V,  LogRet, N, constraint)

% function used to compute weights for MVP and Max Sharpe portfolio with
% resampling
%
% INPUTS: 
% p -> portfolio object with constraint
% ExpRet -> Expected Returns
% V -> Covariance Matrix
% LogRet -> Log Returns
% N -> Number of simulations
% constraint -> Logical variable that is true when our portfolio has
%               additional constraints

%
% OUTPUTS:
% MVP -> Struct containing weights, volatility and returns of the MVP
% MaxSharpe -> Struct containing weights, volatility and returns of 
%              the portfolio with the highest Sharpe Ratio
% pf -> Struct containing portfolio moments of all simulations

% Initializing the Structs
Port.w = zeros(101,100,N);
pf.ret = zeros(100,N);
pf.risk = zeros(100,N);

rng(2)
if constraint             % case with constraints
    for i = 1 : N
   
    flag = true;
        while flag
            try
                % Generate random returns based on the current mean and covariance
                R = mvnrnd(ExpRet,V,length(LogRet));
        
                % Update expected return and covariance
                NewExpRet = mean(R);
                NewCov = cov(R);
                % Set asset moments for the portfolio using the simulated data
                P = setAssetMoments(p,NewExpRet,NewCov);
                % Estimate the efficient frontier and estimate moments
                w = estimateFrontier(P,100);
       
        
                [pf.risk(:,i),pf.ret(:,i)] = estimatePortMoments(P,w);
         
                Port.w(:,:,i) = w;
               
            

                flag = false;
            end
        end
    end

else                % case without constraints
    for i = 1 : N
    % Generate random returns based on the current mean and covariance
    R = mvnrnd(ExpRet,V,length(LogRet));

    % Update expected return and covariance
    NewExpRet = mean(R);
    NewCov = cov(R);

    % Set asset moments for the portfolio using the simulated data
    P = setAssetMoments(p,NewExpRet,NewCov);

    % Estimate the efficient frontier and estimate moments
    w = estimateFrontier(P,100);
    [pf.risk(:,i),pf.ret(:,i)] = estimatePortMoments(P,w);
    
    Port.w(:,:,i) = w;
 
    end
end
w_sim_mean = mean(Port.w,3);
ExpRetSim = mean(pf.ret,2);
ExpRiskSim = mean(pf.risk,2);
[~, indexMVP] = min(mean(pf.risk,2));
MVP.w = w_sim_mean(:,indexMVP);
MVP.ret = ExpRetSim(indexMVP);
MVP.risk = ExpRiskSim(indexMVP);
[~, indexSHARPE] = max(mean(pf.ret,2)./mean(pf.risk,2));
MaxSharpe.w = w_sim_mean(:,indexSHARPE);
MaxSharpe.ret = ExpRetSim(indexSHARPE);
MaxSharpe.risk = ExpRiskSim(indexSHARPE);
end



