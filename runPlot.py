import sys, os, json
import numpy as np
import scipy as s
import matplotlib.pyplot as plt

from resultAnalysis import ResultAnalysis
from resultPlot import ResultPlot


resultsDir = '../resultsGibbs/177_SNR50_RandWalkIsofitCovEps0_11_2M/'
# resultsDir = '../resultsGibbs/177_SNR50_RandWalkIsofitCovEps0_11_1M_noThin/'
setupDir = 'setup/ang20140612/' #'setup/ang20170228/' #
a = ResultAnalysis(setupDir, resultsDir)
p = ResultPlot(a.exportParam())

# # Results Visualization
# p.plotRadiance()
# p.plotPosterior()
# p.plotError()
# p.plot2Dcontour(vis='samples')
# p.contourRef(vis='contour')
# p.contourAtm()


# # Results Analysis
# a.diagnostics(indSet=[20,50,150,160,250,260,400,410])
# a.traceRef()
# a.traceAtm()
# a.plotacceptance()
# a.plotlogpos()

# a.plotac(132)

a.genESSspectrum()
# a.ESSanalysis()
# a.MCMCIsofitEig()
# a.comparePosCov()
# a.comparePosCorr()
# a.quantDiagnostic()
# a.qqRef()
# a.qqAtm()
# a.KStestRef()



plt.show()
