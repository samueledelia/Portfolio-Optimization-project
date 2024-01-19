function new_weights = ESBack(~,subsample)
    assetReturns = tick2ret(subsample, 'Method', 'continuous');
    LogRet = assetReturns.Variables;
    ExpRet = mean(LogRet);

    alpha = 0.05;
    Mod_ES = @(x) (ExpRet*x)/(mean(LogRet*x)+std(LogRet*x)*(pdf('normal',norminv(1-alpha),0,1))/alpha);
    p_7 = Portfolio('AssetList', subsample.Properties.VariableNames);

    p_7 = setDefaultConstraints(p_7);


    new_weights = estimateCustomObjectivePortfolio(p_7, Mod_ES, ObjectiveSense = "maximize");


end