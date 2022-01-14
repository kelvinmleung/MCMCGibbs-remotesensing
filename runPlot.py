import sys, os, json
import numpy as np
import scipy as s
import matplotlib.pyplot as plt

from plot import PlotFromFile


# resultsDir = '../resultsGibbs/177_SNR50_DetRefl_1M/'
resultsDir = '../resultsGibbs/Test1/'
setupDir = 'setup/ang20140612/' #'setup/ang20170228/' #
p = PlotFromFile(setupDir, resultsDir)
# p.plotRadiance()
p.plotPosterior()
# p.plotError()
p.plot2Dmarginal()
# p.plot2Dcontour()
p.kdcontouratm(indX=432, indY=433)
p.diagnostics(indSet=[20,50,150,160,250,260,400,410])
# p.quantDiagnostic()

fig, ax = plt.subplots()
p.twoDimVisual(432,433,ax)
# p.twoDimVisual(425,426,ax)
plt.title('2D Marginal - Atmospheric Parameters')
plt.savefig(resultsDir + '2DmarginalATM.png', dpi=300)


# plot radiance
# from isofitSetup import Setup
# np_load_old = np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# with open(setupDir + 'config/config_inversion.json', 'r') as f:
#     config = json.load(f)
# setup = Setup(wv=p.wavelengths, ref=p.truth[:-2], radiance=0, config=config, resultsDir=resultsDir, setupDir=setupDir)

# plt.figure()
# plt.plot(p.wavelengths, p.yobs, 'r', label='Observed Radiance')
# plt.plot(p.wavelengths, p.posPredictive, 'k', label='MAP Posterior Predictive')
# plt.plot(p.wavelengths, setup.fm.calc_rdn(p.MCMCmean, setup.geom), 'b', label='MCMC Posterior Predictive')
# plt.xlabel('Wavelength')
# plt.ylabel('Radiance')
# plt.title('Observed and Predicted Radiance')
# plt.grid()
# plt.legend()
# plt.savefig(p.resultsDir + 'radiances.png', dpi=300)


plt.show()
