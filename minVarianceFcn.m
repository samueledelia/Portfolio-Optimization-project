function new_weights = minVarianceFcn(~,subsample,flag)
    global G_Consumer;
    global G_Industrial;
    global G_Null;

    assetReturns = tick2ret(subsample, 'Method', 'continuous');

    % Create Portfolio object
    p = Portfolio('AssetList', subsample.Properties.VariableNames);
    p = setDefaultConstraints(p);
    if flag
        p = addGroups(p, G_Consumer,0.15);                          
        p = addGroups(p, G_Industrial,[],0.05);                    
        p = addGroups(p, G_Null,0,0);
    end
    P = estimateAssetMoments(p, assetReturns, 'MissingData', false);
    
    pwgt = estimateFrontier(P, 1000);
    
    [pf_Risk, ~] = estimatePortMoments(P, pwgt);
    
    [~, idx] = min(pf_Risk);
    new_weights = pwgt(:, idx)';
end