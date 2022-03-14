import sys, os, json
import numpy as np
import scipy as s
import scipy.stats as st
from scipy.stats import multivariate_normal, gaussian_kde, probplot, kstest, norm
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

class ResultAnalysis:

    def __init__(self, setupDir, resultsDir):

        self.setupDir = setupDir
        self.resultsDir = resultsDir

        self.loadFromFile()

        try:
            self.loadMCMC()
        except:
            print('MCMC files not found.')

        self.NAC = 10000

        self.checkConfig()
        self.plotIndices = self.windowInd()
        self.wavelengths = self.wavelengths.astype(int)
        self.bands = self.getBands()

        
    def loadFromFile(self):

        # self.wavelengths = np.load(self.resultsDir + 'wavelength.npy')
        self.yobs = np.load(self.resultsDir + 'radiance.npy')
        self.truth = np.load(self.resultsDir + 'truth.npy')
        self.bands = np.load(self.resultsDir + 'bands.npy')

        self.mu_x = np.load(self.resultsDir + 'mu_x.npy')
        self.gamma_x = np.load(self.resultsDir + 'gamma_x.npy')
        self.isofitMuPos = np.load(self.resultsDir + 'isofitMuPos.npy')
        self.isofitGammaPos = np.load(self.resultsDir + 'isofitGammaPos.npy')
        self.nx = self.mu_x.shape[0]
        self.posPredictive = np.load(self.resultsDir + 'posPredictive.npy')
        
        try:
            self.Nsamp = np.load(self.resultsDir + 'Nsamp.npy')
            self.burn = np.load(self.resultsDir + 'burn.npy')
            self.thinning = np.load(self.resultsDir + 'thinning.npy')

            self.Nthin = int(self.Nsamp / self.thinning)
        except:
            print('MCMC files not found.')
    
    def loadMCMC(self):
        self.x_vals = np.load(self.resultsDir + 'mcmcchain.npy', mmap_mode='r')

        self.x_plot = self.x_vals[:,self.burn:]
        self.MCMCmean = np.mean(self.x_plot, axis=1)
        self.MCMCcov = np.cov(self.x_plot)

        self.logpos = np.load(self.resultsDir + 'logpos.npy')
        self.acceptAtm = np.load(self.resultsDir + 'acceptAtm.npy')
        self.acceptRef = np.load(self.resultsDir + 'acceptRef.npy')

        # self.x_vals_ac = x_vals[:,:self.NAC]
    def exportParam(self):
        params= {
            "wavelengths": self.wavelengths,
            "yobs": self.yobs,
            "truth": self.truth,
            "bands": self.bands,
            "mu_x": self.mu_x,
            "gamma_x": self.gamma_x,
            "isofitMuPos": self.isofitMuPos,
            "isofitGammaPos": self.isofitGammaPos,
            "nx": self.nx,
            "posPredictive": self.posPredictive,
            "Nsamp": self.Nsamp,
            "burn": self.burn,
            "Nthin": self.Nthin,
            "x_vals": self.x_vals,
            "x_plot": self.x_plot,
            "MCMCmean": self.MCMCmean,
            "MCMCcov": self.MCMCcov,
            "logpos": self.logpos,
            "acceptAtm": self.acceptAtm,
            "acceptRef": self.acceptRef,
            "NAC": self.NAC,
            "plotIndices": self.plotIndices,
            "resultsDir": self.resultsDir
            }
        return params

    def checkConfig(self):

        configFile = self.setupDir + 'config/config_inversion.json'
        wvFile = self.setupDir + 'data/wavelengths.txt'
        
        fileLoad = np.loadtxt(wvFile).T
        if fileLoad.shape[0] == 2:
            wv, fwhm = fileLoad
        elif fileLoad.shape[0] == 3:
            ind, wv, fwhm = fileLoad
            wv = wv * 1000
        self.wavelengths = wv

        with open(configFile, 'r') as f:
            self.config = json.load(f)

    def windowInd(self):
        wl = self.wavelengths
        w = self.config['implementation']['inversion']['windows']
        range1, range2, range3 = [], [], []
        for i in range(wl.size):
            if wl[i] > w[0][0] and wl[i] < w[0][1]:
                range1 = range1 + [i]
            elif wl[i] > w[1][0] and wl[i] < w[1][1]:
                range2 = range2 + [i]
            elif wl[i] > w[2][0] and wl[i] < w[2][1]:
                range3 = range3 + [i]
        r1 = [min(range1), max(range1)]
        r2 = [min(range2), max(range2)]
        r3 = [min(range3), max(range3)]  

        return [r1, r2, r3]

    def getBands(self):
        # get indices that are in the window (i.e. take out deep water spectra)
        wl = self.wavelengths
        w = self.config['implementation']['inversion']['windows']
        bands = []
        for i in range(wl.size):
            # if (wl[i] > 380 and wl[i] < 1300) or (wl[i] > 1450 and wl[i] < 1780) or (wl[i] > 1950 and wl[i] < 2450):
            if (wl[i] > w[0][0] and wl[i] < w[0][1]) or (wl[i] > w[1][0] and wl[i] < w[1][1]) or (wl[i] > w[2][0] and wl[i] < w[2][1]):
                bands = bands + [i]
        return bands
        
    
    def quantDiagnostic(self):
        ## Error for reflectance parameters

        # Error in reflectance
        isofitErrorVec = self.isofitMuPos[:self.nx-2] - self.truth[:self.nx-2]
        mcmcErrorVec = self.MCMCmean[:self.nx-2] - self.truth[:self.nx-2]

        isofitError = np.linalg.norm(self.isofitMuPos[self.bands] - self.truth[self.bands]) / np.linalg.norm(self.truth[self.bands])
        mcmcError = np.linalg.norm(self.MCMCmean[self.bands] - self.truth[self.bands]) / np.linalg.norm(self.truth[self.bands])

        # Inverse variance weighted error
        # ivweIsofit, isofitVarDenom = 0, 0
        # ivweMCMC, mcmcVarDenom = 0, 0
        # isofitVar = np.diag(self.isofitGammaPos)
        # mcmcVar = np.diag(self.MCMCcov)

        isofitWeightCov = np.linalg.inv(self.isofitGammaPos[:,:self.nx-2][:self.nx-2,:])
        mcmcWeightCov = np.linalg.inv(self.MCMCcov[:,:self.nx-2][:self.nx-2,:])

        weightErrIsofit = isofitErrorVec.T @ isofitWeightCov @ isofitErrorVec
        weightErrMCMC = mcmcErrorVec.T @ mcmcWeightCov @ mcmcErrorVec

        # for i in self.bands:
        #     isofitVarDenom = isofitVarDenom + isofitVar[i]
        #     mcmcVarDenom = mcmcVarDenom + mcmcVar[i]
        #     ivweIsofit = ivweIsofit + isofitErrorVec[i] / isofitVar[i]
        #     ivweMCMC = ivweMCMC + mcmcErrorVec[i] / mcmcVar[i]

        print('Relative Error in Retrieved Reflectance')
        print('\tIsofit:', isofitError)
        print('\tMCMC:', mcmcError)
        print('\nInverse Posterior Covariance Weighted Error')
        print('\tIsofit:', weightErrIsofit)
        print('\tMCMC:', weightErrMCMC)

    def autocorr(self, x_elem):
        x = x_elem
        Nsamp = len(x)
        laglen = 2000
        # varX = np.var(x_elem)
        meanX = np.mean(x)
        ac = np.zeros(laglen)            
        denom = np.sum((x - np.ones(Nsamp)*meanX)**2)

        for k in range(laglen):
            one = np.ones(Nsamp - k)
            num = np.sum((x[k:] - one*meanX) * (x[:Nsamp-k] - one*meanX))
            ac[k] = num / denom
        #     cov = np.cov(x_elem[:Nsamp-k], x_elem[k:Nsamp]) 
        #     ac[k] = cov[1,0] * (Nsamp-k) / (varX * Nsamp)
        return ac

    def plotac(self, ind):
        
        ac = self.autocorr(self.x_plot[ind,:])
        print(np.sum(ac))
        laglen=len(ac)
        plt.figure()
        plt.plot(list(range(laglen)), ac[:laglen], '.')

    def ESS(self, ac):
        # denom = 0
        # for i in range(len(ac)):
        #     denom = denom + ac[i]
        denom = np.sum(ac)
        # print('IACT:', denom)
        return self.Nthin / (1 + 2 * denom)

    def genESSspectrum(self):
        essSpec = np.zeros(self.nx)
        for i in range(self.nx):
            print('Calculating ESS, index =', i)
            ac = self.autocorr(self.x_plot[i,:])
            essSpec[i] = self.ESS(ac)
            print('\t', essSpec[i])
        np.save(self.resultsDir + 'ESSspectrum.npy', essSpec)
        
    def ESSanalysis(self):
        essSpec = np.load(self.resultsDir + 'ESSspectrum.npy')
        essRef = essSpec[self.bands]
        fig = plt.figure()
        plt.semilogy(self.wavelengths, essSpec[:self.nx-2], '.')
        plt.title('Effective Sample Size - Reflectances')
        plt.xlabel('Wavelength')
        plt.ylabel('ESS')
        fig.savefig(self.resultsDir + 'ess.png', dpi=300)  
        print('Reflectances:')
        print('\tMin ESS:', np.min(essRef))
        print('\tMed ESS:', np.median(essRef))
        print('\tMax ESS:', np.max(essRef))
        print('AOD ESS:', essSpec[-2])
        print('H2O ESS:', essSpec[-1])


    def MCMCIsofitEig(self):

        covIsofit = self.isofitGammaPos[:,self.bands][self.bands,:] 
        covMCMC = self.MCMCcov[:,self.bands][self.bands,:] 

        eigs, eigvec = s.linalg.eigh(covIsofit, covMCMC, eigvals_only=False)
        eigs = np.flip(eigs, axis=0)
        eigvec = np.flip(eigvec, axis=0)
        fig = plt.figure()
        plt.semilogy(eigs)
        plt.title('Eigenspectrum of Isofit vs MCMC Covariances')
        plt.xlabel('Large eig signifies larger Isofit variance')
        fig.savefig(self.resultsDir + 'eigval.png', dpi=300)  


        fig = plt.figure()
        for i in range(3):
            plt.plot(self.bands, eigvec[:,i], '-', label='nu='+str(round(eigs[i],2)))
        plt.xlabel('Wavelength')
        plt.title('Eigenvectors of Isofit vs MCMC Covariances')
        plt.legend()
        fig.savefig(self.resultsDir + 'eigvec.png', dpi=300)  

        # plt.figure()
        # plt.plot()
        # print(eigs)
        # for j in range(eigs.size):
        #     forstner = forstner + (np.log(eigs[j])) ** 2
        # forstner = np.sqrt(forstner)


    # def diagnostics(self, indSet=[20,50,80,110,140,170,230,250,280,250,380,410]):
    #     # assume there are 12 elements in indSet
    #     # default: indSet = [10,20,50,100,150,160,250,260,425,426]
    #     if self.nx-2 not in indSet:
    #         indSet.extend([self.nx-2, self.nx-1]) 

    #     N = self.x_vals.shape[1]
    #     numPairs = int(len(indSet) / 2) 


    #     # subplot setup
    #     fig1, axs1 = plt.subplots(numPairs, 2)
    #     # fig2, axs2 = plt.subplots(numPairs, 2)
    #     xPlot = np.zeros(numPairs * 2, dtype=int)
    #     yPlot = np.zeros(numPairs * 2, dtype=int)
    #     xPlot[::2] = range(numPairs)
    #     xPlot[1::2] = range(numPairs)
    #     yPlot[1::2] = 1

    #     for i in range(len(indSet)):
    #         # print('Diagnostics:',indSet[i])
    #         x_elem = self.x_vals[indSet[i],:]
    #         xp = xPlot[i]
    #         yp = yPlot[i]

    #         # plot trace
    #         axs1[xp,yp].plot(range(N) * self.thinning, x_elem)
    #         axs1[xp,yp].set_title('Trace - Index ' + str(indSet[i]))

    #         # plot autocorrelation
    #         # ac = self.autocorr(self.x_plot[indSet[i]])
    #         # ac = ac[:int(len(ac)/2)]
    #         # axs2[xp,yp].plot(range(1,len(ac)+1) * self.thinning, ac)
    #         # axs2[xp,yp].set_title('Autocorrelation - Index ' + str(indSet[i]))

    #     fig1.set_size_inches(5, 7)
    #     fig1.tight_layout()
    #     fig1.savefig(self.resultsDir + 'trace.png', dpi=300)
    #     # fig2.set_size_inches(5, 7)
    #     # fig2.tight_layout()
    #     # fig2.savefig(self.resultsDir + 'autocorr.png', dpi=300)

    def traceRef(self, indset=[20,50,80,110,140,170,230,250,280,250,380,410]):
        n = 4
        m = 3

        # n = int(len(indset)/4)
        # m = int(len(indset)/n)

        fig, ax = plt.subplots(n, m)
        for i in range(n):
            for j in range(m):
                ind = indset[i*m + j]

                ax[i,j].plot(range(self.Nthin) * self.thinning, self.x_vals[ind,:])
                ax[i,j].set_title(r'$\lambda = $' + str(self.wavelengths[ind]) + ' nm')

        fig.suptitle('Trace - Reflectances')


        # handles, labels = ax[0,0].get_legend_handles_labels()
        # fig.legend(handles, labels, loc='center right')
        # fig.subplots_adjust(right=0.83)
        fig.set_size_inches(15, 9)
        fig.tight_layout()
        fig.savefig(self.resultsDir + 'traceRef.png', dpi=300)  

    def traceAtm(self):
        fig, ax = plt.subplots(2,1)

        ax[0].plot(range(self.Nthin) * self.thinning, self.x_vals[self.nx-2,:])
        ax[0].set_title('AOD550')

        ax[1].plot(range(self.Nthin) * self.thinning, self.x_vals[self.nx-1,:])
        ax[1].set_title('H20STR')

        fig.set_size_inches(6, 4)
        fig.tight_layout()
        
        fig.suptitle('Trace - Atm Parameters')
        fig.savefig(self.resultsDir + 'traceAtm.png', dpi=300)    
        
        
    def plotacceptance(self):
        # acceptance rate
        acceptRateAtm= np.mean(self.acceptAtm[self.burn:])
        acceptRateRef= np.mean(self.acceptRef[self.burn:])
        binWidth = 1000
        numBin = int(self.Nsamp / binWidth)
        xPlotAccept = np.arange(binWidth, self.Nsamp+1, binWidth) * self.thinning
        acceptPlotAtm = np.zeros(numBin)
        acceptPlotRef = np.zeros(numBin)
        for i in range(numBin):
            acceptPlotAtm[i] = np.mean(self.acceptAtm[binWidth*i : binWidth*(i+1)])
            acceptPlotRef[i] = np.mean(self.acceptRef[binWidth*i : binWidth*(i+1)])
        plt.figure()
        plt.plot(xPlotAccept, acceptPlotAtm, label='Atm Param')
        plt.plot(xPlotAccept, acceptPlotRef, label='Reflectance')
        plt.xlabel('Number of Samples')
        plt.ylabel('Acceptance Rate')
        plt.title('Acceptance Rate')# = ' + str(round(acceptRate,2)))
        plt.legend()
        plt.ylim([0, 1])
        plt.savefig(self.resultsDir + 'acceptance.png', dpi=300)

    def plotlogpos(self):
        # plot logpos
        plt.figure()
        plt.plot(range(self.Nthin) * self.thinning, self.logpos)
        plt.xlabel('Number of Samples')
        plt.ylabel('Log Posterior')
        plt.savefig(self.resultsDir + 'logpos.png', dpi=300)

    def KStestAll(self, indset=[20,50,80,110,140,170,230,250,280,250,380,410]):
        
        indset = list(range(self.nx-2))
        n = len(indset)
        for i in range(n):
            ind = indset[i]
            normDist = norm(loc=self.MCMCmean[ind], scale=np.sqrt(self.MCMCcov[ind,ind]))
            ks_statistic, p_value = kstest(self.x_plot[ind,::10], normDist.cdf)
            print(str(self.wavelengths[ind]) + 'nm\t', round(ks_statistic,3), '\t', round(p_value,3))

        ind = self.nx-2
        normDist = norm(loc=self.MCMCmean[ind], scale=np.sqrt(self.MCMCcov[ind,ind]))
        ks_statistic, p_value = kstest(self.x_plot[ind,::10], normDist.cdf)
        print('AOD \t', round(ks_statistic,3), '\t', round(p_value,3))

        ind = self.nx-1
        normDist = norm(loc=self.MCMCmean[ind], scale=np.sqrt(self.MCMCcov[ind,ind]))
        ks_statistic, p_value = kstest(self.x_plot[ind,::10], normDist.cdf)
        print('H2O \t', round(ks_statistic,3), '\t', round(p_value,3))


    def KStestRef(self):
        indset = self.bands
        n = len(indset)

        ks_stat = np.zeros(n)
        p_val = np.zeros(n)
        for i in range(n):
            ind = indset[i]
            normDist = norm(loc=self.MCMCmean[ind], scale=np.sqrt(self.MCMCcov[ind,ind]))
            ks_stat[i], p_val[i] = kstest(self.x_plot[ind,::10], normDist.cdf)

        fig = plt.figure()
        plt.plot(self.wavelengths[self.bands], p_val, 'b.')
        lineX = [self.wavelengths[self.bands[0]], self.wavelengths[self.bands[-1]]]
        lineY = [0.05, 0.05]
        plt.plot(lineX, lineY, 'r-', label='p=0.05')
        plt.title('KS Test for Gaussianity')
        plt.xlabel('Wavelength')
        plt.ylabel('p-value')
        plt.legend()
        fig.savefig(self.resultsDir + 'kstest.png', dpi=300)      

    
    def qqRef(self, indset=[20,50,80,110,140,170,230,250,280,250,380,410]):

        n = int(len(indset)/4)
        m = int(len(indset)/n)
        fig, ax = plt.subplots(n, m)
        
        for i in range(n):
            for j in range(m):
                ind = indset[i*m + j]
                probplot(self.x_plot[ind,::10], dist='norm', plot=ax[i,j])
                ax[i,j].set_title(r'$\lambda = $' + str(self.wavelengths[ind]) + ' nm')

        fig.suptitle('QQ Plots - Reflectances')

        # handles, labels = ax[0,0].get_legend_handles_labels()
        # fig.legend(handles, labels, loc='center right')
        # fig.subplots_adjust(right=0.83)
        fig.set_size_inches(15, 9)
        fig.tight_layout()
        fig.savefig(self.resultsDir + 'qqRef.png', dpi=300)      

    def qqAtm(self):

        fig, ax = plt.subplots(1,2)

        probplot(self.x_plot[self.nx-2,::10], dist='norm', plot=ax[0])
        ax[0].set_title('AOD550')

        probplot(self.x_plot[self.nx-1,::10], dist='norm', plot=ax[1])
        ax[1].set_title('H20STR')

        fig.set_size_inches(8, 4)
        fig.tight_layout()
        
        fig.suptitle('QQ Plots - Atm Parameters')
        fig.savefig(self.resultsDir + 'qqAtm.png', dpi=300)    

    def comparePosCov(self):
        covPrior = self.gamma_x[:,self.bands][self.bands,:] 
        covIsofit = self.isofitGammaPos[:,self.bands][self.bands,:] 
        covMCMC = self.MCMCcov[:,self.bands][self.bands,:] 
        print('Compare Covariance Matrices\n')
        self.compareMatrix(covPrior, covIsofit, covMCMC)

    def comparePosCorr(self):

        covPrior = self.gamma_x[:,self.bands][self.bands,:] 
        varPrior = np.diag(1 / np.sqrt(np.diag(covPrior)))
        corrPrior = varPrior @ covPrior @ varPrior

        covIsofit = self.isofitGammaPos[:,self.bands][self.bands,:] 
        varIsofit = np.diag(1 / np.sqrt(np.diag(covIsofit)))
        corrIsofit = varIsofit.T @ covIsofit @ varIsofit

        covMCMC = self.MCMCcov[:,self.bands][self.bands,:] 
        varMCMC = np.diag(1 / np.sqrt(np.diag(covMCMC)))
        corrMCMC = varMCMC.T @ covMCMC @ varMCMC
        print('Compare Correlation Matrices\n')
        self.compareMatrix(corrPrior, corrIsofit, corrMCMC)
    
    def compareMatrix(self, covPrior, covIsofit, covMCMC):        
        
        # trace
        traceIsofit = np.trace(covIsofit)
        traceMCMC = np.trace(covMCMC)
        traceDiff = abs(traceIsofit - traceMCMC) / abs(traceMCMC)

        # log determinant
        sgn1, detIsofit = np.linalg.slogdet(covIsofit)
        sgn2, detMCMC = np.linalg.slogdet(covMCMC)
        detDiff = abs(detIsofit - detMCMC) / abs(detMCMC)

        # Frobenius norm
        normIsofit = np.linalg.norm(covIsofit)
        normMCMC = np.linalg.norm(covMCMC)
        normDiff = abs(normIsofit - normMCMC) / abs(normMCMC)

        # forstner distance
        forstner = 0
        eigs = s.linalg.eigh(covIsofit, covMCMC, eigvals_only=True)
        # print(eigs)
        for j in range(eigs.size):
            forstner = forstner + (np.log(eigs[j])) ** 2
        forstner = np.sqrt(forstner)

        forstPr = 0
        eigs = s.linalg.eigh(covPrior, covMCMC, eigvals_only=True)
        # print(eigs)
        for j in range(eigs.size):
            forstPr = forstPr + (np.log(eigs[j])) ** 2
        forstPr = np.sqrt(forstPr)

        print('\t\t Isofit \t MCMC \t Percent Diff')
        # print('Determinant:   %10.3E  %10.3E  %10.3f' % (detIsofit, detMCMC, detDiff))
        print('Trace:         %10.3E  %10.3E  %10.3f' % (traceIsofit, traceMCMC, traceDiff))
        print('Frob Norm:     %10.3E  %10.3E  %10.3f' % (normIsofit, normMCMC, normDiff))
        print('Log Det:       %10.3E  %10.3E        ' % (detIsofit, detMCMC))
        print('Forstner:              %10.3f  %10.3f' % (forstner, forstner/forstPr))



    # def plot2ac(self, indset=[120,250,410]):

    #     fig, axs = plt.subplots(1, len(indset))

    #     for i in range(len(indset)):
    #         # print('Autocorr:', indset[i])

    #         ac = self.autocorr(self.x_vals_ac[indset[i],:])
    #         ac2 = self.autocorr(self.x_vals_ac_noLIS[indset[i],:])

    #         ac = ac[:self.numPlotAC]
    #         ac2 = ac2[:self.numPlotAC]

    #         print('Index:', indset[i])
    #         print('ESS LIS:', self.ESS(ac))
    #         print('ESS No LIS:', self.ESS(ac2))

    #         # plot autocorrelation
    #         axs[i].plot(range(1,len(ac)+1), ac, 'b', label='LIS r = 100')
    #         axs[i].plot(range(1,len(ac2)+1), ac2, 'r', label='No LIS')
    #         if indset[i] < 425:
    #             axs[i].set_title(r'$\lambda = $' + str(self.wavelengths[indset[i]]) + ' nm')
    #         elif indset[i] == 425:
    #             axs[i].set_title('AOD')
    #         elif indset[i] == 426:
    #             axs[i].set_title('H2O')
        
    #     axs[0].set_xlabel('Lag', fontsize=14)
    #     axs[0].set_ylabel('Autocorrelation', fontsize=14)
        
    #     handles, labels = axs[0].get_legend_handles_labels()
    #     fig.legend(handles, labels, loc='center right', fontsize=14)
    #     # fig2.savefig(self.resultsDir + 'autocorr.png', dpi=300)



    

    










