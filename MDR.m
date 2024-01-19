function new_weights = MDR(~,subsample, G_Financials)
    global G_Industrial

    original_vector = 0.01 * G_Industrial + 0.02 * G_Financials;
    new_vector = original_vector;
    new_vector(new_vector == 0) = 1;
    assetReturns = tick2ret(subsample, 'Method', 'continuous');
    
    V = cov(assetReturns.Variables);     
    va = var(assetReturns.Variables);
    DR = @(x) (x' * sqrt(va)') / sqrt(x' * V * x);
   

    p = Portfolio('AssetList', subsample.Properties.VariableNames);
    p = setDefaultConstraints(p);
    p = setBounds(p, 0.005 * G_Industrial + 0.001 * G_Financials, new_vector);
    
    
    new_weights = estimateCustomObjectivePortfolio(p, DR, ObjectiveSense = "maximize");
end