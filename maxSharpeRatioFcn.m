function new_weights = maxSharpeRatioFcn(~, subsample, flag)
    global G_Consumer;
    global G_Industrial;
    global G_Null;


    % Mean-variance portfolio allocation
    assetReturns = tick2ret(subsample, 'Method', 'continuous');

    % Create Portfolio object
    p = Portfolio('AssetList', subsample.Properties.VariableNames);
    p = setDefaultConstraints(p);
    if flag
        p = addGroups(p, G_Consumer,0.15);                          
        p = addGroups(p, G_Industrial,[],0.05);                    
        p = addGroups(p, G_Null,0,0);
    end
    
    % Estimate asset moments
    P = estimateAssetMoments(p, assetReturns, 'MissingData', false);
    new_weights = estimateMaxSharpeRatio(P,Method='iterative');

end