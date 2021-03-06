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

# CHANGE COLOURS

# ## For Paper and Conference ##
# cases = ['177']#,'306','mars','dark']
# for case in cases:
#     resultsDir = '../resultsGibbs/' + case + '_SNR50_RandWalkIsofitCovEps0_11_2M/'
#     # resultsDir = '../resultsGibbs/177_SNR50_RandWalkIsofitCovEps0_11_1M_noThin/'
#     setupDir = 'setup/ang20140612/' #'setup/ang20170228/' #
#     a = ResultAnalysis(setupDir, resultsDir)
#     p = ResultPlot(a.exportParam()) 
#     # p.plotRadiance()
#     # p.plotPosterior()
#     # p.plotError()
#     # p.contourRef(vis='contour')
#     # p.contourAtm()
#     # # p.corrRef(indset1=[20,40,60,80,100], indset2=[30,45,52,58,62,66,72,75,79])
#     # p.corrRef(indset1=[20,40,60], indset2=[30,45,52,58,62,80])

#     # a.traceRef()
#     # a.traceAtm()
#     # a.plotacceptance()
#     # a.plotlogpos()

#     # a.genESSspectrum()
#     # a.ESSanalysis()
#     # # a.MCMCIsofitEig()
#     a.comparePosCov()
#     # a.qqRef()
#     # a.qqAtm()
#     # a.KStestRef()


# # Results Visualization
# p.plotRadiance()
# p.plotPosterior()
# p.plotError()
# p.contourRef(vis='samples')
# p.contourRef(vis='contour')
# p.contourAtm()
# p.corrRef(indset1=[20,80,140,230,280,380], indset2=[50,110,170,250,350,410])
# p.corrRef(indset1=[20,40,60,80,100,120], indset2=[30,45,52,58,62,66])
# p.corrRef(indset1=[20,40,60,80,100], indset2=[30,45,52,58,62,66,72,75,79])

# Results Analysis
# a.traceRef()
# a.traceAtm()
# a.plotacceptance()
# a.plotlogpos()

# a.genESSspectrum()
# a.ESSanalysis()
# a.MCMCIsofitEig()
# a.comparePosCov()
# a.comparePosCorr()
# a.quantDiagnostic()
# a.qqRef()
# a.qqAtm()
# a.KStestRef()

# a.MCMCIsofitEigShrink()
a.isofitVarInMCMCdir()
a.MCMCVarInIsofitdir()

plt.show()
