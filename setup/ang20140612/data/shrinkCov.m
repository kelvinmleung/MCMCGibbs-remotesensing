clear; clc;
load('surface_uncorrelated.mat');


covsNew = zeros(size(covs));
figure; hold on;
for i = 1:8
%     covsNew(i,:,:) = squeeze(covs(i,1,1)) * eye(432) * 1e-3;
    covsNew(i,:,:) = eye(432) * 5e-5;
%     semilogy(diag(squeeze(covsNew(i,:,:))));
end
% legend;

figure; hold on;
for i = 1:8
c = chol(squeeze(covsNew(i,:,:)));
samp = means(i,:)' + c * randn(432,1);
plot(abs(samp));

end

covs = covsNew;
    

save surface.mat covs means normalize refwl wl attribute_covs attribute_means attributes



