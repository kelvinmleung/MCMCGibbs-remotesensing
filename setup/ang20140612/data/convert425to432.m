clear; clc;
load('convert425to432.mat');

% get the map from 432 to 425
% figure(1); hold on;
% for i = 1:8
%     plot(wl_432,means_432(i,:), 'Linewidth',2);
%     legend('1','2','3','4','5','6','7','8')
% end

[x, y] = meshgrid(wl, wl);
[x_432,y_432] = meshgrid(wl_432, wl_432);

covs_432 = zeros(8,432,432);

ind425 = [2,6,4,8,7,1,5,3];

for i = 1:8
    covi = squeeze(covs(i,:,:));
    covs_432(i,:,:) = interp2(x,y,covi, x_432,y_432, 'linear',0);
    inv(covi);
    disp(i)
    % make the extrapolated values 0, except diagonal to 2e-3
    % this ensures the invertibility of the matrix
    for j = 1:432
        if covs_432(i,j,j) == 0
            covs_432(i,j,j) = 2e-3;
        end
    end
    inv(squeeze(covs_432(i,:,:))); % check invertibility
%     plot(diag(squeeze(covs_432(i,:,:)))); hold on;
    
end

refwl = refwl_432;
wl = wl_432;
means = means_432;
covs = covs_432;

figure; hold on;
for i = 1:8
    e = eig(squeeze(covs(i,:,:)));
    semilogy(sort(e));
end

% save surface.mat covs means normalize refwl wl attribute_covs attribute_means attributes



