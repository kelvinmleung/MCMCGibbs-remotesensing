import numpy as np
import scipy as s
from scipy.stats import probplot, kstest, norm, truncnorm
import matplotlib.pyplot as plt

from resultAnalysis import ResultAnalysis

def plotbands(a, ax, y, linestyle, linewidth=2, label='', axis='normal'):
    wl = a.wavelengths
    r1, r2, r3 = a.plotIndices
    if axis == 'normal':
        ax.plot(wl[r1[0]:r1[1]], y[r1[0]:r1[1]], linestyle, linewidth=linewidth, label=label)
        ax.plot(wl[r2[0]:r2[1]], y[r2[0]:r2[1]], linestyle, linewidth=linewidth)
        ax.plot(wl[r3[0]:r3[1]], y[r3[0]:r3[1]], linestyle, linewidth=linewidth)
    elif axis == 'semilogy':
        ax.semilogy(wl[r1[0]:r1[1]], y[r1[0]:r1[1]], linestyle, linewidth=linewidth, label=label)
        ax.semilogy(wl[r2[0]:r2[1]], y[r2[0]:r2[1]], linestyle, linewidth=linewidth)
        ax.semilogy(wl[r3[0]:r3[1]], y[r3[0]:r3[1]], linestyle, linewidth=linewidth)

setupDir = 'setup/ang20140612/'
n = 432
nplot = 180000

essSpec = np.zeros([2,434])

radiance = np.zeros([4,432])

isofitMean = np.zeros([4,432])
isofitVar = np.zeros([4,432])

posMean = np.zeros([4,432])
posVar = np.zeros([4,432])
posMeanError = np.zeros([4,432])


xplot10 = np.zeros([4,434,int(nplot/10)])
KSref = np.zeros([4,432])
plotName = ['Building 177', 'Building 306', 'Mars Yard', 'Parking Lot']
cases = ['177','306','mars','dark']

for i in range(4):
    resultsDir = '../resultsGibbs/' + cases[i] + '_SNR50_RandWalkIsofitCovEps0_11_2M/'

    a = ResultAnalysis(setupDir, resultsDir)

    radiance[i,:] = a.yobs

    isofitMean[i,:] = a.isofitMuPos[:n]
    isofitVar[i,:] = np.diag(a.isofitGammaPos)[:n]
    
    posMean[i,:] = a.MCMCmean[:n]
    posVar[i,:] = np.diag(a.MCMCcov)[:n]
    posMeanError[i,:] = abs(isofitMean[i,:] - posMean[i,:]) / posMean[i,:] 

    
    xplot10[i,:,:] = a.x_plot[:,::10]

    wavelengths = a.wavelengths
    bands = a.bands
    print(len(bands))

essSpec[0,:] = np.load('../resultsGibbs/177_SNR50_RandWalkEps0_14_2M/' + 'ESSspectrum.npy')
essSpec[1,:] = np.load('../resultsGibbs/177_SNR50_RandWalkIsofitCovEps0_11_2M/' + 'ESSspectrum.npy')
xplotAtm = np.zeros([4,200000])
xplotAtm[:2,:] = np.load('../resultsGibbs/177_SNR50_RandWalkEps0_14_2M/' + 'mcmcchain.npy')[-2:,:]
xplotAtm[2:,:] = np.load('../resultsGibbs/177_SNR50_RandWalkIsofitCovEps0_11_2M/' + 'mcmcchain.npy')[-2:,:]

# Radiance Plot
fig, ax = plt.subplots(1)
for i in range(4):
    ax.plot(wavelengths, radiance[i,:], linewidth=0.5, label=plotName[i])
ax.set_title('Radiance Measurements')
plt.ylabel(r'$\mu \,W \,nm^{-1}\, cm^{-2}\, sr^{-1}$')
plt.xlabel('Wavelength [nm]')
plt.legend()
fig.set_size_inches(8,5)


# ESS
fig, ax = plt.subplots(2)
for i in range(2):
    ax[i].plot(wavelengths[bands], essSpec[i,bands], 'b.')
ax[0].set_title('Proposal from Linear Inversion Theory')
ax[1].set_title('Proposal from Laplace Approximation')
# ax[0].legend()
plt.xlabel('Wavelength [nm]')
fig.supylabel('Effective Sample Size')
fig.tight_layout(pad=1, h_pad=0.5, w_pad=0.5)
fig.set_size_inches(5, 5)

# Trace
fig, ax = plt.subplots(4)
traceTitle = ['AOD','H2O','AOD','H2O']
for i in range(4):
    ax[i].plot(list(range(200000)), xplotAtm[i,:])#, 'b')
    ax[i].set_ylabel(traceTitle[i])
ax[0].set_title('Proposal from Linear Inversion Theory')
ax[2].set_title('Proposal from Laplace Approximation')
# ax[0].legend()
plt.xlabel('Sample')
# fig.supylabel('Effective Sample Size')
fig.tight_layout(pad=1, h_pad=0.5, w_pad=0.5)
fig.set_size_inches(5, 5)

# Posterior mean
# uncer = np.sqrt(posVar)
fig, ax = plt.subplots(4)
for i in range(4):
    # ax[i].fill_between(wavelengths[bands], posMean[i,bands] + uncer[i,bands], posMean[i,bands] - uncer[i,bands])
    plotbands(a, ax[i], isofitMean[i,:], 'r', linewidth=1, label='OE', axis='normal')
    plotbands(a, ax[i], posMean[i,:], 'b', linewidth=1, label='MCMC', axis='normal')
    ax[i].set_title(plotName[i])# + ' Reflectance')
ax[0].legend()
plt.xlabel('Wavelength [nm]')
fig.supylabel('Reflectance')
fig.tight_layout(pad=1, h_pad=0, w_pad=0)
fig.set_size_inches(5, 8)

# Posterior var
fig, ax = plt.subplots(4)
for i in range(4):
    plotbands(a, ax[i], isofitVar[i,:], 'r', linewidth=1, label='OE', axis='normal')
    plotbands(a, ax[i], posVar[i,:], 'b', linewidth=1, label='MCMC', axis='normal')
    ax[i].set_title(plotName[i])# + ' Reflectance - Variance')
ax[0].legend()
plt.xlabel('Wavelength [nm]')
fig.supylabel('Variance')
fig.tight_layout(pad=1, h_pad=0, w_pad=0)
fig.set_size_inches(5, 8)

# Posterior mean error
fig, ax = plt.subplots(4)
for i in range(4):
    plotbands(a, ax[i], posMeanError[i,:], 'k', linewidth=1, axis='normal')
    ax[i].set_title(plotName[i])# + ' Reflectance - Error')
plt.xlabel('Wavelength [nm]')
fig.supylabel('Relative Difference')
fig.tight_layout(pad=1, h_pad=0, w_pad=0)
fig.set_size_inches(5, 8)

# QQ Atm
fig, ax = plt.subplots(4,2)
for i in range(4):
    # ax[i,0].plot(*probplot(xplot10[432,:])[0], 'b.')
    # ax[i,0].plot(*probplot(xplot10[432,:])[1], 'r')

    
    lowAOD = -np.mean(xplot10[i,432,:]) / np.std(xplot10[i,432,:])
    tnormAOD = truncnorm(lowAOD, np.inf)
    probplot(xplot10[i,432,:], dist=tnormAOD, plot=ax[i,0])
    # ax[i,0].set_title(plotName[i] + ' - AOD')

    ax[i,0].set_xlabel('')
    ax[i,0].set_title('')
    ax[i,0].set_ylabel(plotName[i])
    # ax[i,1].plot(*probplot(xplot10[433,:])[0], 'b.')

    lowH2O = -np.mean(xplot10[i,433,:]) / np.std(xplot10[i,433,:])
    tnormH2O = truncnorm(lowH2O, np.inf)
    probplot(xplot10[i,433,:], dist=tnormH2O, plot=ax[i,1])
    # ax[i,1].set_title(plotName[i] + ' - H2O')
    ax[i,1].set_xlabel('')
    ax[i,1].set_ylabel('')
    ax[i,1].set_title('')
ax[0,0].set_title('AOD')
ax[0,1].set_title('H2O')
fig.supxlabel('Theoretical Quantiles')
fig.supylabel('Ordered Values')
fig.set_size_inches(5, 8)
fig.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)

# fig.suptitle('QQ Plots - AOD Parameter')

# KS Test Refl
for i in range(4):
    # ks_stat = np.zeros(n)
    # p_val = np.zeros(n)
    for j in range(n):
        normDist = norm(loc=posMean[i,j], scale=np.sqrt(posVar[i,j]))
        # KSref[i,j], p_val[j] = kstest(xplot10[i,j,:], normDist.cdf)
        KSref[i,j] = kstest(xplot10[i,j,:], normDist.cdf)[1]
lineX = [wavelengths[bands[0]], wavelengths[bands[-1]]]
lineY = [0.05, 0.05]


fig, ax = plt.subplots(4,1)
for i in range(4):
    ax[i].plot(wavelengths[bands], KSref[i,bands], 'b.')
    ax[i].plot(lineX, lineY, 'r-', label='p = 0.05')
    # ax[i].set_title(plotName[i] + ' Reflectance - Variance')
    ax[i].set_ylabel(plotName[i])
ax[0].legend()
plt.xlabel('Wavelength [nm]')
# fig.supxlabel('Wavelength [nm]')
fig.supylabel('p-Value')
fig.suptitle('Kolmogorov-Smirnov Test for Gaussianity')
fig.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
fig.set_size_inches(6, 8)

plt.show()
    





