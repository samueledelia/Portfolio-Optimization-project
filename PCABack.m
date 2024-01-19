function new_weights = PCABack(~,subsample, k)
    assetReturns = tick2ret(subsample, 'Method', 'continuous');
    [factorLoading, factorRetn, ~] = pca(assetReturns.Variables, 'NumComponents', k);
    numAssets = size(subsample, 2);
    
    ExpRet = mean(assetReturns.Variables);
    covarFactor = cov(factorRetn);
    reconReturn = factorRetn*factorLoading' + ExpRet;
    unexplainedRetn = assetReturns.Variables - reconReturn;
    
    unexplainedCovar = diag(cov(unexplainedRetn));
    D = diag(unexplainedCovar);
    
    
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
    new_weights = x.asset_weight;
end