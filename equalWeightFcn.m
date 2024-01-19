function new_weights = equalWeightFcn(~,subsample)
% Equal-weighted portfolio allocation
    
    nAssets = size(subsample, 2);
    new_weights = ones(1,nAssets);
    new_weights = new_weights / sum(new_weights);

end