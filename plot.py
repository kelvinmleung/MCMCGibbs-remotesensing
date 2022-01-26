import sys, os, json
import numpy as np
import scipy as s
import scipy.stats as st
from scipy.stats import multivariate_normal, gaussian_kde
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

class PlotFromFile:

    def __init__(self, setupDir, resultsDir):

        self.setupDir = setupDir
        self.resultsDir = resultsDir

        self.loadFromFile()

        try:
            self.loadMCMC()
        except:
            print('MCMC files not found.')

        self.NsampAC = 10000#int(100000 / self.thinning)

        self.checkConfig()
        self.plotIndices = self.windowInd()
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
            # print(self.burn)
            self.burnthin = self.burn # int(self.burn / self.thinning)
        except:
            print('MCMC files not found.')

        # self.linMuPos = np.load(self.paramDir + 'linMuPos.npy')
        # self.linGammaPos = np.load(self.paramDir + 'linGammaPos.npy')

        # self.phi = np.load(self.regDir + 'phi.npy')
        # self.meanX = np.load(self.paramDir + 'meanX.npy')
        # self.meanY = np.load(self.paramDir + 'meanY.npy')
        # self.varX = np.load(self.paramDir + 'varX.npy')
        # self.varY = np.load(self.paramDir + 'varY.npy')

    
    def loadMCMC(self):
        self.x_vals = np.load(self.resultsDir + 'mcmcchain.npy', mmap_mode='r')

        self.x_plot = self.x_vals[:,self.burnthin:]
        self.MCMCmean = np.mean(self.x_plot, axis=1)
        self.MCMCcov = np.cov(self.x_plot)

        self.logpos = np.load(self.resultsDir + 'logpos.npy')
        self.acceptAtm = np.load(self.resultsDir + 'acceptAtm.npy')
        self.acceptRef = np.load(self.resultsDir + 'acceptRef.npy')

        # self.x_vals_ac = x_vals[:,:self.NsampAC]

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
        # return r1, r2, r3
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
        
    def plotRadiance(self):
        plt.figure()
        plt.plot(self.wavelengths, self.yobs, 'b', label='Observed Radiance')
        plt.plot(self.wavelengths, self.posPredictive, 'r', label='MAP Posterior Predictive')
        plt.xlabel('Wavelength')
        plt.ylabel('Radiance')
        plt.title('Observed and Predicted Radiance')
        plt.grid()
        plt.legend()
        plt.savefig(self.resultsDir + 'radiances.png', dpi=300)

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
        
    def plotbands(self, y, linestyle, linewidth=2, label='', axis='normal'):
        wl = self.wavelengths
        r1, r2, r3 = self.plotIndices
        if axis == 'normal':
            plt.plot(wl[r1[0]:r1[1]], y[r1[0]:r1[1]], linestyle, linewidth=linewidth, label=label)
            plt.plot(wl[r2[0]:r2[1]], y[r2[0]:r2[1]], linestyle, linewidth=linewidth)
            plt.plot(wl[r3[0]:r3[1]], y[r3[0]:r3[1]], linestyle, linewidth=linewidth)
        elif axis == 'semilogy':
            plt.semilogy(wl[r1[0]:r1[1]], y[r1[0]:r1[1]], linestyle, linewidth=linewidth, label=label)
            plt.semilogy(wl[r2[0]:r2[1]], y[r2[0]:r2[1]], linestyle, linewidth=linewidth)
            plt.semilogy(wl[r3[0]:r3[1]], y[r3[0]:r3[1]], linestyle, linewidth=linewidth)

    def plotPosterior(self):

        plt.figure()
        self.plotbands(self.truth[:self.nx-2], 'k.',label='True Reflectance')
        self.plotbands(self.isofitMuPos[:self.nx-2],'r.', label='Isofit Posterior')
        self.plotbands(self.MCMCmean[:self.nx-2], 'b.',label='MCMC Posterior')
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        plt.title('Posterior Mean - Surface Reflectance')
        plt.grid()
        plt.legend()
        plt.savefig(self.resultsDir + 'reflMean.png', dpi=300)

        plt.figure()
        # plt.plot(self.truth[self.nx-2], self.truth[self.nx-1], 'bo',label='True Reflectance')
        plt.plot(self.mu_x[self.nx-2], self.mu_x[self.nx-1], 'k.', markersize=12, label='Prior')
        plt.plot(self.isofitMuPos[self.nx-2],self.isofitMuPos[self.nx-1],'r.', markersize=12, label='Isofit Posterior')
        plt.plot(self.MCMCmean[self.nx-2], self.MCMCmean[self.nx-1], 'bx',markersize=12, label='MCMC Posterior')
        plt.xlabel('AOT550')
        plt.ylabel('H2OSTR')
        plt.title('Posterior Mean - Atmospheric Parameters')
        plt.grid()
        plt.legend()
        plt.savefig(self.resultsDir + 'atmMean.png', dpi=300)

        # bar graph of atm parameter variances
        # isofitErrorAtm = abs(self.isofitMuPos[self.nx-2:] - self.truth[self.nx-2:]) / abs(self.truth[self.nx-2:])
        # mcmcErrorAtm = abs(self.MCMCmean[self.nx-2:] - self.truth[self.nx-2:]) / abs(self.truth[self.nx-2:])
        # labels = ['425 - AOD550', '426 - H2OSTR']
        # x = np.arange(len(labels))  # the label locations
        # width = 0.175
        # fig, ax = plt.subplots()
        # rects2 = ax.bar(x - width, isofitErrorAtm, width, label='Isofit Posterior')
        # rects4 = ax.bar(x + width, mcmcErrorAtm, width, label='MCMC Posterior')
        # ax.set_yscale('log')
        # ax.set_ylabel('Relative Error')
        # ax.set_title('Error in Atm Parameters')
        # ax.set_xticks(x)
        # ax.set_xticklabels(labels)
        # ax.legend()
        # fig.savefig(self.resultsDir + 'atmError.png', dpi=300)

        # variance plot
        priorVar = np.diag(self.gamma_x)
        isofitVar = np.diag(self.isofitGammaPos)
        MCMCVar = np.diag(self.MCMCcov)
        plt.figure()
        self.plotbands(priorVar[:self.nx-2], 'k.',label='Prior', axis='semilogy')
        self.plotbands(isofitVar[:self.nx-2],'r.', label='Isofit Posterior', axis='semilogy')
        self.plotbands(MCMCVar[:self.nx-2], 'b.',label='MCMC Posterior', axis='semilogy')
        plt.xlabel('Wavelength')
        plt.ylabel('Variance')
        plt.title('Posterior Variance - Surface Reflectance')
        plt.grid()
        plt.legend()
        plt.savefig(self.resultsDir + 'reflVar.png', dpi=300)

        # bar graph of atm parameter variances
        labels = ['Aerosol', 'H2OSTR']
        x = np.arange(len(labels))  # the label locations
        width = 0.175
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width, priorVar[self.nx-2:], width, color='k', label='Prior')
        rects2 = ax.bar(x, isofitVar[self.nx-2:], width, color='r', label='Isofit Posterior')
        rects4 = ax.bar(x + width, MCMCVar[self.nx-2:], width, color='b', label='MCMC Posterior')
        ax.set_yscale('log')
        ax.set_ylabel('Variance')
        ax.set_title('Posterior Variance - Atmospheric Parameters')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        fig.savefig(self.resultsDir + 'atmVar.png', dpi=300)
    
    def plotError(self):

        plt.figure()
        isofitError = abs(self.isofitMuPos[:self.nx-2] - self.truth[:self.nx-2]) / abs(self.truth[:self.nx-2])
        mcmcError = abs(self.MCMCmean[:self.nx-2] - self.truth[:self.nx-2]) / abs(self.truth[:self.nx-2])
        self.plotbands(isofitError,'r.', label='Isofit Posterior',axis='semilogy')
        self.plotbands(mcmcError, 'b.',label='MCMC Posterior',axis='semilogy')
        plt.xlabel('Wavelength')
        plt.ylabel('Relative Error')
        plt.title('Error in Posterior Mean')
        plt.grid()
        plt.legend()
        plt.savefig(self.resultsDir + 'reflError.png', dpi=300)

        plt.figure()
        isofitVar = np.diag(self.isofitGammaPos[:,:self.nx-2][:self.nx-2,:])
        mcmcVar = np.diag(self.MCMCcov[:,:self.nx-2][:self.nx-2,:])
        isofitMatOper = s.linalg.sqrtm(np.linalg.inv(np.diag(isofitVar)))
        mcmcMatOper = s.linalg.sqrtm(np.linalg.inv(np.diag(mcmcVar)))
        isofitWeightError = isofitMatOper @ (self.isofitMuPos[:self.nx-2] - self.truth[:self.nx-2])
        mcmcWeightError = mcmcMatOper @ (self.MCMCmean[:self.nx-2] - self.truth[:self.nx-2])
        self.plotbands(abs(isofitWeightError),'r.', label='Isofit Posterior',axis='semilogy')
        self.plotbands(abs(mcmcWeightError), 'b.',label='MCMC Posterior',axis='semilogy')
        plt.xlabel('Wavelength')
        plt.ylabel('Error Weighted by Marginal Variance')
        plt.title('Weighted Error in Posterior Mean')
        plt.grid()
        plt.legend()
        plt.savefig(self.resultsDir + 'reflWeightError.png', dpi=300)


    
    def plot2Dmarginal(self, indset1=[100,250,410], indset2=[30,101,260]):
        
        n = len(indset1)
        m = len(indset2)
        fig, ax = plt.subplots(n, m)
        
        for i in range(n):
            for j in range(m):
                indX = indset1[i]
                indY = indset2[j]

                ax[i,j] = self.twoDimVisual(indY, indX, ax[i,j])
                # ax[i,j].set_title('CHANGE TITLE')
        #         ax[i,j].set_xlabel('Wavelength Channel ' + str(self.wavelengths[indY]))
        #         ax[i,j].set_ylabel('Wavelength Channel ' + str(self.wavelengths[indX]))
        fig.suptitle('2D Marginal Plots')
        # fig.savefig(self.resultsDir + '2Dmarginal.png', dpi=300)

        # ax[0].set_ylabel(r'$\lambda = $' + str(self.wavelengths[indset1[0]]) + ' nm')
        ax[0,0].set_ylabel(r'$\lambda = $' + str(self.wavelengths[indset1[0]]) + ' nm')
        ax[1,0].set_ylabel(r'$\lambda = $' + str(self.wavelengths[indset1[1]]) + ' nm')
        ax[2,0].set_ylabel(r'$\lambda = $' + str(self.wavelengths[indset1[2]]) + ' nm')

        # ax[0].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[0]]) + ' nm')
        # ax[1].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[1]]) + ' nm')
        # ax[2].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[2]]) + ' nm')
        ax[2,0].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[0]]) + ' nm')
        ax[2,1].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[1]]) + ' nm')
        ax[2,2].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[2]]) + ' nm')
        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')
        fig.set_size_inches(12, 8)
        fig.savefig(self.resultsDir + '2Dmarginal.png', dpi=300)

    def plot2Dcontour(self, indset1=[100,250,410], indset2=[30,101,260]):
        
        n = len(indset1)
        m = len(indset2)
        fig, ax = plt.subplots(n, m)
        levs = [0, 0.05, 0.1, 0.2, 0.5, 1]
        # cfset = ax.contourf(xx, yy, f, levels=levs, cmap='Blues') 
        # cset = ax.contour(xx, yy, f, levels=levs, colors='k') 
        
        for i in range(n):
            for j in range(m):
                indX = indset1[i]
                indY = indset2[j]
                print(i,j)
                ax[i,j], cfset = self.twoDimContour(indY, indX, ax[i,j], levs)
        fig.suptitle('2D Contour Plots')

        ax[0,0].set_ylabel(r'$\lambda = $' + str(self.wavelengths[indset1[0]]) + ' nm')
        ax[1,0].set_ylabel(r'$\lambda = $' + str(self.wavelengths[indset1[1]]) + ' nm')
        ax[2,0].set_ylabel(r'$\lambda = $' + str(self.wavelengths[indset1[2]]) + ' nm')
        ax[2,0].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[0]]) + ' nm')
        ax[2,1].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[1]]) + ' nm')
        ax[2,2].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[2]]) + ' nm')
        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')
        fig.subplots_adjust(right=0.83)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.colorbar(cfset, cax = cbar_ax)
        fig.set_size_inches(12, 8)
        fig.savefig(self.resultsDir + '2Dcontour.png', dpi=300)

    def twoDimContour(self, indX, indY, ax, levs):

        x = self.x_plot[indX,:]
        y = self.x_plot[indY,:]

        isofitPosX = self.isofitMuPos[indX]
        isofitPosY = self.isofitMuPos[indY]
        xmin, xmax = min(min(x), isofitPosX), max(max(x), isofitPosX)
        ymin, ymax = min(min(y), isofitPosY), max(max(y), isofitPosY)

        if indX < self.nx-2 and indY < self.nx-2:
            xmin, xmax = min(xmin, self.truth[indX]), max(xmax, self.truth[indX])
            ymin, ymax = min(ymin, self.truth[indY]), max(ymax, self.truth[indY])

        # Peform the kernel density estimate
        xx, yy = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        f = f / np.max(f) # normalize

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        levs = [0, 0.05, 0.1, 0.2, 0.5, 1]

        # Contourf plot
        cfset = ax.contourf(xx, yy, f, levels=levs, cmap='Blues') 
        cset = ax.contour(xx, yy, f, levels=levs, colors='k') 
        ax.clabel(cset, levs, fontsize='smaller')

        # plot truth, isofit, and mcmc 
        meanIsofit = np.array([isofitPosX, isofitPosY])
        meanMCMC = np.array([self.MCMCmean[indX], self.MCMCmean[indY]])
        ax.plot(self.truth[indX], self.truth[indY], 'go', label='True reflectance', markersize=10)  
        ax.plot(meanIsofit[0], meanIsofit[1], 'rx', label='MAP', markersize=12)
        ax.plot(meanMCMC[0], meanMCMC[1], 'kx', label='MCMC', markersize=12)
            
        return ax, cfset

    def kdcontouratm(self, indX, indY):
        x_vals = np.load(self.resultsDir + 'mcmcchain.npy')
        x_vals_plot = x_vals[:,self.burnthin:]

        x = x_vals_plot[indX,:]
        y = x_vals_plot[indY,:]

        isofitPosX = self.isofitMuPos[indX]
        isofitPosY = self.isofitMuPos[indY]
        xmin, xmax = min(min(x), isofitPosX), max(max(x), isofitPosX)
        ymin, ymax = min(min(y), isofitPosY), max(max(y), isofitPosY)

        if indX < self.nx-2 and indY < self.nx-2:
            xmin, xmax = min(xmin, self.truth[indX]), max(xmax, self.truth[indX])
            ymin, ymax = min(ymin, self.truth[indY]), max(ymax, self.truth[indY])

        # Peform the kernel density estimate
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        f = f / np.max(f) # normalize

        fig = plt.figure()
        plt.title('Contour Plot of Posterior Samples')
        ax = fig.gca()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        levs = [0, 0.05, 0.1, 0.2, 0.5, 1]
        # Contourf plot
        cfset = ax.contourf(xx, yy, f, levels=levs, cmap='Blues') 
        cset = ax.contour(xx, yy, f, levels=levs, colors='k') 
        plt.clabel(cset, levs, fontsize='smaller')

        # plot truth, isofit, and mcmc 
        meanIsofit = np.array([isofitPosX, isofitPosY])
        meanMCMC = np.array([self.MCMCmean[indX], self.MCMCmean[indY]])
        ax.plot(meanIsofit[0], meanIsofit[1], 'rx', label='MAP', markersize=12)
        ax.plot(meanMCMC[0], meanMCMC[1], 'kx', label='MCMC', markersize=12)

        if indX < self.nx-2 and indY < self.nx-2:
            # Label plot
            # ax.clabel(cset, inline=1, fontsize=10)
            ax.set_xlabel(r'$\lambda = $' + str(self.wavelengths[indX]) + ' nm')
            ax.set_ylabel(r'$\lambda = $' + str(self.wavelengths[indY]) + ' nm')

            # plot truth
            ax.plot(self.truth[indX], self.truth[indY], 'ro', label='Truth', markersize=8)  
        else:
            if indX == self.nx-2:
                ax.set_xlabel('AOD550')
                ax.set_ylabel('H20STR')
        ax.legend()
        fig.colorbar(cfset)

        fig.savefig(self.resultsDir + 'kdcontour_' + str(indX) + '_' + str(indY) + '.png', dpi=300)
        return fig


    def plotPosSparsity(self, tol):

        deadbands = list(range(185,215)) + list(range(281,315)) + list(range(414,425))
        cov = self.MCMCcov
        for i in deadbands:
            cov[i,:] = np.zeros(cov.shape[0])
            cov[:,i] = np.zeros(cov.shape[0])

        plt.figure()
        plt.spy(cov, color='b', precision=tol, markersize=2)
        plt.title('Sparsity Plot of Posterior Covariance - Tolerance = ' + str(tol))

    def plotPosCovRow(self, indset=[120,250,410]):
        
        for i in indset:
            plt.figure()
            self.plotbands(self.MCMCcov[i,:], 'b.', axis='semilogy')
            plt.title('Posterior Covariance - Row of index ' + str(i) + ', wavelength=' + str(self.wavelengths[i]))
            plt.xlabel('Wavelength')
            plt.ylabel('Value of Covariance Matrix')

    def drawEllipse(self, mean, cov, ax, colour):
        ''' Helper function for twoDimVisual '''
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor='None', edgecolor=colour)
        scale_x = np.sqrt(cov[0, 0]) * 1
        scale_y = np.sqrt(cov[1, 1]) * 1 
        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean[0], mean[1])
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse) 

    def twoDimVisual(self, indX, indY, ax):
        x_vals = self.x_plot

        if indX < self.nx-2 and indY < self.nx-2:
            ax.plot(self.truth[indX], self.truth[indY], 'ro', label='True reflectance', markersize=10)     
        ax.scatter(x_vals[indX,:], x_vals[indY,:], c='cornflowerblue', s=0.5)

        # plot Isofit mean/cov
        meanIsofit = np.array([self.isofitMuPos[indX], self.isofitMuPos[indY]])
        covIsofit = self.isofitGammaPos[np.ix_([indX,indY],[indX,indY])]
        ax.plot(meanIsofit[0], meanIsofit[1], 'kx', label='MAP/Laplace', markersize=12)
        self.drawEllipse(meanIsofit, covIsofit, ax, colour='black')
        
        # plot MCMC mean/cov
        meanMCMC = np.array([self.MCMCmean[indX], self.MCMCmean[indY]])
        covMCMC = self.MCMCcov[np.ix_([indX,indY],[indX,indY])]
        ax.plot(meanMCMC[0], meanMCMC[1], 'bx', label='MCMC', markersize=12)
        self.drawEllipse(meanMCMC, covMCMC, ax, colour='blue')
        
        # ax.set_title('MCMC - Two Component Visual')
        # ax.set_xlabel('Index ' + str(indX))
        # ax.set_ylabel('Index ' + str(indY))
        # ax.legend()
        return ax

    def autocorr(self, x_elem):
        Nsamp = self.NsampAC
        meanX = np.mean(x_elem)
        varX = np.var(x_elem)
        ac = np.zeros(Nsamp-1)

        for k in range(Nsamp-1):
            cov = np.cov(x_elem[:Nsamp-k], x_elem[k:Nsamp])
            ac[k] = cov[1,0] / varX

        return ac

    def ESS(self, ac):
        denom = 0
        for i in range(len(ac)):
            denom = denom + ac[i]
        return self.Nsamp / (1 + 2 * denom)

    def diagnostics(self, indSet=[10,20,50,100,150,160,250,260]):
        # assume there are 10 elements in indSet
        # default: indSet = [10,20,50,100,150,160,250,260,425,426]
        if self.nx-2 not in indSet:
            indSet.extend([self.nx-2, self.nx-1]) 

        N = self.x_vals.shape[1]
        numPairs = int(len(indSet) / 2) 


        # subplot setup
        fig1, axs1 = plt.subplots(numPairs, 2)
        # fig2, axs2 = plt.subplots(numPairs, 2)
        xPlot = np.zeros(numPairs * 2, dtype=int)
        yPlot = np.zeros(numPairs * 2, dtype=int)
        xPlot[::2] = range(numPairs)
        xPlot[1::2] = range(numPairs)
        yPlot[1::2] = 1

        for i in range(len(indSet)):
            # print('Diagnostics:',indSet[i])
            x_elem = self.x_vals[indSet[i],:]
            xp = xPlot[i]
            yp = yPlot[i]

            # plot trace
            axs1[xp,yp].plot(range(N) * self.thinning, x_elem)
            axs1[xp,yp].set_title('Trace - Index ' + str(indSet[i]))

            # plot autocorrelation
            # ac = self.autocorr(self.x_plot[indSet[i]])
            # ac = ac[:int(len(ac)/2)]
            # axs2[xp,yp].plot(range(1,len(ac)+1) * self.thinning, ac)
            # axs2[xp,yp].set_title('Autocorrelation - Index ' + str(indSet[i]))

        fig1.set_size_inches(5, 7)
        fig1.tight_layout()
        fig1.savefig(self.resultsDir + 'trace.png', dpi=300)
        # fig2.set_size_inches(5, 7)
        # fig2.tight_layout()
        # fig2.savefig(self.resultsDir + 'autocorr.png', dpi=300)
        
        # plot logpos
        plt.figure()
        plt.plot(range(N) * self.thinning, self.logpos)
        plt.xlabel('Number of Samples')
        plt.ylabel('Log Posterior')
        plt.savefig(self.resultsDir + 'logpos.png', dpi=300)

        # acceptance rate
        acceptRateAtm= np.mean(self.acceptAtm[self.burnthin:])
        acceptRateRef= np.mean(self.acceptRef[self.burnthin:])
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



    

    










