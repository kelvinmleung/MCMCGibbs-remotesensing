import sys, os, json
import numpy as np
import scipy as s
import matplotlib.pyplot as plt

from resultAnalysis import ResultAnalysis
from resultPlot import ResultPlot


resultsDir = '../resultsGibbs/dark_SNR50_RandWalkEps0_14_2M/'
# resultsDir = '../resultsGibbs/177_SNR50_RandWalkIsofitCovEps0_11_1M_noThin/'
setupDir = 'setup/ang20140612/' #'setup/ang20170228/' #
a = ResultAnalysis(setupDir, resultsDir)
p = ResultPlot(a.exportParam())

# # Results Visualization
p.plotRadiance()
p.plotPosterior()
p.plotError()
p.contourRef(vis='samples')
p.contourRef(vis='contour')
p.contourAtm()


# Results Analysis
a.traceRef()
a.traceAtm()
a.plotacceptance()
a.plotlogpos()

# a.genESSspectrum()
a.ESSanalysis()
a.MCMCIsofitEig()
a.comparePosCov()
a.comparePosCorr()
a.quantDiagnostic()
a.qqRef()
a.qqAtm()
a.KStestRef()



plt.show()
