function new_weights = BLSharpe(~,subsample,sectors, wMkt)
    numAssets = size(subsample, 2);
    tau = 1 / size(subsample, 2);
    v = 3; 
    P = zeros(v, numAssets);
    q = zeros(v, 1);
    Omega = zeros(v);
    
    P(1, strcmp(sectors.Sector, 'Consumer Staples')) = 1;
    q(1) = 0.07;
    
    P(2, strcmp(sectors.Sector, 'Health Care')) = 1;
    q(2) = 0.03;
    
    P(3, strcmp(sectors.Sector, 'Communication Services')) = 1;
    P(3, strcmp(sectors.Sector, 'Utilities')) = -1;
    q(3) = 0.04;
    assetReturns = tick2ret(subsample, 'Method', 'continuous');

    V = cov(assetReturns.Variables); 
    
    Omega(1, 1) = tau * P(1, :) * V * P(1, :)';
    Omega(2, 2) = tau * P(2, :) * V * P(2, :)';
    Omega(3, 3) = tau * P(3, :) * V * P(3, :)';
    
    bizyear2bizday = 1 / 252;
    q = q * bizyear2bizday;
    Omega = Omega * bizyear2bizday;
    
  
    lambda = 1.2;                                   
    mu_mkt = lambda * V * wMkt;
    C = tau * V;
    
    
    muBL = inv(inv(C) + P' * inv(Omega) * P) * (P' * inv(Omega) * q + inv(C) * mu_mkt);
    covBL = inv(P' * inv(Omega) * P + inv(C));
    
    pBL = Portfolio('AssetList', subsample.Properties.VariableNames);
    pBL = setAssetMoments(pBL, muBL, V + covBL);
    pBL = setDefaultConstraints(pBL);

    new_weights = estimateMaxSharpeRatio(pBL, 'Method', 'iterative');
end