clear;clc;
load surface.mat

ind = 3;

cov = squeeze(covs(ind,:,:));
cholcov = chol(cov);

figure();
plot(wl,means(ind,:), 'Linewidth', 2); hold on;

i = 0;
while i < 1000
    samp = cholcov * normrnd(0,1,[432,1]) + means(ind,:)';
    if all(samp > 0)
        i = i + 1;
        plot(wl,samp);
    end
end
