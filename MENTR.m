function new_weights = MENTR(~,subsample, G_Financials)
    global G_Industrial

    original_vector = 0.01 * G_Industrial + 0.02 * G_Financials;
    new_vector = original_vector;
    new_vector(new_vector == 0) = 1;
    assetReturns = tick2ret(subsample, 'Method', 'continuous');
    
   
    ENTR_INASS = @(x) -sum(((x.^2).*(std(assetReturns.Variables)'.^2))./sum((x.^2).*(std(assetReturns.Variables)'.^2)).*log(((x.^2).*(std(assetReturns.Variables)'.^2))./sum((x.^2).*(std(assetReturns.Variables)'.^2))));
   

    p = Portfolio('AssetList', subsample.Properties.VariableNames);
    p = setDefaultConstraints(p);
    p = setBounds(p, 0.005 * G_Industrial + 0.001 * G_Financials, new_vector);
    
    
    new_weights = estimateCustomObjectivePortfolio(p, ENTR_INASS, ObjectiveSense = "maximize");
end