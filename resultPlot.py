import sys, os, json
import numpy as np
import scipy as s
import scipy.stats as st
from scipy.stats import multivariate_normal, gaussian_kde, probplot
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from resultAnalysis import ResultAnalysis

class ResultPlot:

    def __init__(self, params):

        self.loadParam(params)
        
    def loadParam(self, params):
        self.wavelengths = params["wavelengths"]
        self.yobs = params["yobs"]
        self.truth = params["truth"]
        self.bands = params["bands"]
        self.mu_x = params["mu_x"]
        self.gamma_x = params["gamma_x"]
        self.isofitMuPos = params["isofitMuPos"]
        self.isofitGammaPos = params["isofitGammaPos"]
        self.nx = params["nx"]
        self.posPredictive = params["posPredictive"]
        self.Nsamp = params["Nsamp"]
        self.burn = params["burn"]
        self.Nthin = params["Nthin"]
        self.x_vals = params["x_vals"]
        self.x_plot = params["x_plot"]
        self.MCMCmean = params["MCMCmean"]
        self.MCMCcov = params["MCMCcov"]
        self.logpos = params["logpos"]
        self.acceptAtm = params["acceptAtm"]
        self.acceptRef = params["acceptRef"]
        self.NAC = params["NAC"]
        self.plotIndices = params["plotIndices"]
        self.resultsDir = params["resultsDir"]
        
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

    def contourRef(self, indset1=[20,80,140,230,280,380], indset2=[50,110,170,250,350,410], vis='contour'):
        
        n = len(indset1)
        m = len(indset2)
        fig, ax = plt.subplots(n, m)
        levs = [0.03, 0.14, 0.6, 1]
        
        for i in range(n):
            for j in range(m):
                indX = indset1[i]
                indY = indset2[j]
                if vis == 'contour':
                    ax[i,j], cfset = self.twoDimContour(indY, indX, ax[i,j], levs)
                elif vis == 'samples':
                    ax[i,j] = self.twoDimSamples(indY, indX, ax[i,j])
                # ax[i,j].set_aspect('equal','box')
                if j == 0:
                    ax[i,j].set_ylabel(r'$\lambda = $' + str(self.wavelengths[indset1[i]]) + ' nm')
                if i == n-1:
                    ax[i,j].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[j]]) + ' nm')

        fig.suptitle('2D Contour Plots')

        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')
        fig.subplots_adjust(right=0.83)
        if vis == 'contour':
            cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
            fig.colorbar(cfset, cax = cbar_ax)
        fig.set_size_inches(18, 10)
        fig.savefig(self.resultsDir + '2Dcontour.png', dpi=300)

    def contourAtm(self):

        fig, ax = plt.subplots(1, 1)
        levs = [0, 0.05, 0.1, 0.2, 0.5, 1]
        
        ax, cfset = self.twoDimContour(self.nx-2, self.nx-1, ax, levs)
        ax.set_xlabel('AOD550')
        ax.set_ylabel('H20STR')
        ax.set_xlim([0,0.3])
        fig.suptitle('Contour Plot for Atmospheric Parameters')

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')
        fig.colorbar(cfset)
        fig.savefig(self.resultsDir + 'atmcontour.png', dpi=300)

        return fig

    def corrRef(self, indset1=[20,80,140,230,280,380], indset2=[50,110,170,250,350,410]):
        
        n = len(indset1)
        m = len(indset2)
        fig, ax = plt.subplots(n, m)
        # levs = [0.03, 0.14, 0.6, 1]

        self.isofitCorr = self.covtocorr(self.isofitGammaPos)
        self.MCMCCorr = self.covtocorr(self.MCMCcov)
        
        for i in range(n):
            for j in range(m):
                indX = indset1[i]
                indY = indset2[j]
                
                ax[i,j] = self.twoDimCorr(indY, indX, ax[i,j])
                # ax[i,j].set_aspect('equal','box')
                if j == 0:
                    ax[i,j].set_ylabel(r'$\lambda = $' + str(self.wavelengths[indset1[i]]) + ' nm')
                if i == n-1:
                    ax[i,j].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[j]]) + ' nm')

        fig.suptitle('2D Correlation Plots')

        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')
        # fig.subplots_adjust(right=0.83)
        # fig.tight_layout()
        fig.set_size_inches(18, 10)
        fig.savefig(self.resultsDir + '2Dcorr.png', dpi=300)

    def drawEllipse(self, mean, cov, ax, colour):
        ''' Helper function for twoDimSamples '''
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor='None', edgecolor=colour)
        scale_x = np.sqrt(cov[0, 0]) * 1
        scale_y = np.sqrt(cov[1, 1]) * 1 
        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean[0], mean[1])
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse) 

    def twoDimSamples(self, indX, indY, ax):
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
        
        return ax

    def twoDimContour(self, indX, indY, ax, levs):

        x = self.x_plot[indX,::10]
        y = self.x_plot[indY,::10]

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

        # Contourf plot
        cfset = ax.contourf(xx, yy, f, levels=levs, cmap='Blues') 
        cset = ax.contour(xx, yy, f, levels=levs, colors='k') 
        # ax.clabel(cset, levs, fontsize='smaller')

        # plot truth, isofit, and mcmc 
        meanIsofit = np.array([isofitPosX, isofitPosY])
        meanMCMC = np.array([self.MCMCmean[indX], self.MCMCmean[indY]])
        if indX < self.nx-2 and indY < self.nx-2:
            ax.plot(self.truth[indX], self.truth[indY], 'go', label='Truth', markersize=10)  
        # ax.plot(meanIsofit[0], meanIsofit[1], 'rx', label='MAP', markersize=12)
        ax.plot(meanMCMC[0], meanMCMC[1], 'kx', label='MCMC', markersize=12)

        meanIsofit = np.array([self.isofitMuPos[indX], self.isofitMuPos[indY]])
        covIsofit = self.isofitGammaPos[np.ix_([indX,indY],[indX,indY])]
        ax.plot(meanIsofit[0], meanIsofit[1], 'rx', label='MAP', markersize=12)
        self.drawEllipse(meanIsofit, covIsofit, ax, colour='red')
            
        return ax, cfset

    def covtocorr(self, cov):
        stddev =  np.sqrt(np.diag(cov))
        denom = np.diag(np.ones(self.nx) / stddev)
        return denom @ cov @ denom


    def twoDimCorr(self, indX, indY, ax):
        x_vals = self.x_plot
        meanzero = np.zeros(2)
        corrIsofit = self.isofitCorr[np.ix_([indX,indY],[indX,indY])]
        corrMCMC = self.MCMCCorr[np.ix_([indX,indY],[indX,indY])]

        ax.plot(meanzero[0], meanzero[1], 'kx', label='Laplace', markersize=12)
        self.drawEllipse(meanzero, corrIsofit, ax, colour='black')
        ax.plot(meanzero[0], meanzero[1], 'bx', label='MCMC', markersize=12)
        self.drawEllipse(meanzero, corrMCMC, ax, colour='blue')
        
        return ax

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

    

    

    



    

    










