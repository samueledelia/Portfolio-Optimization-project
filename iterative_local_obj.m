function sratio = iterative_local_obj(ret, obj)
% Set the objective function to the negative of the reward-to-risk ratio.

risk = estimatePortRisk(obj,estimateFrontierByReturn(obj,ret));

if ~isempty(obj.RiskFreeRate)
    sratio = -(ret - obj.RiskFreeRate)/risk;
else
    sratio = -ret/risk;
end

end