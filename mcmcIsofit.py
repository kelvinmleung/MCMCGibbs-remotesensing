import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, gaussian_kde
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from mcmcLIS import MCMCLIS

class MCMCIsofit:
    '''
    Wrapper class to perform MCMC on Isofit
    Contains functions to perform MCMC sampling
    Integrating LIS capabilities
    '''

    def __init__(self, setup, analysis, Nsamp, burn, x0, alg='AM', thinning=1):

        self.mcmcDir = setup.mcmcDir
    
        # initialize problem parameters
        self.wavelengths = setup.wavelengths
        self.reflectance = setup.reflectance # true reflectance
        self.truth = setup.truth # true state (ref + atm)
        # self.radiance = setup.radiance # true radiance
        self.yobs = setup.radiance
        self.bands = setup.bands # reflectance indices excluding deep water spectra
        self.bandsX = setup.bandsX # same indices including atm parameters

        # isofit parameters and functions
        self.mu_x = setup.mu_x
        self.gamma_x = setup.gamma_x
        self.mupos_isofit = setup.isofitMuPos
        self.gammapos_isofit = setup.isofitGammaPos
        self.noisecov = setup.noisecov
        self.fm = setup.fm
        self.geom = setup.geom
        
        # linear model
        self.gamma_ygx = analysis.gamma_ygx # error covariance from linear model
        self.linop = analysis.phi # linear operator
        self.linMuPos, self.linGammaPos = analysis.posterior(self.yobs) # linear posterior covariance 

        # MCMC parameters to initialize
        self.Nsamp = Nsamp
        
        self.x0 = x0
        self.alg = alg
        self.thinning = thinning
        self.burn = int(burn / self.thinning)
        
        self.nx = self.gamma_x.shape[0] # parameter dimension
        self.ny = self.noisecov.shape[0] # data dimension

    def initMCMC(self, LIS=False, rank=427, constrain=False):
        
        # create folder
        if not os.path.exists(self.mcmcDir):
            os.makedirs(self.mcmcDir)

        # define upper and lower bounds 
        if constrain == True:
            lowbound = np.concatenate((np.zeros(self.nx-2), [0, 1.3]))
            upbound = np.concatenate((np.ones(self.nx-2), [0.5, 1.6]))
        else:
            lowbound = np.ones(self.nx) * np.NINF
            upbound = np.ones(self.nx) * np.inf

        self.mcmcConfig = {
            "startX": self.x0,
            "Nsamp": self.Nsamp,
            "burn": self.burn,
            "sd": 2.38 ** 2 / rank,
            "propcov": self.gammapos_isofit * (2.38 ** 2) / rank,# self.linGammaPos * (2.38 ** 2) / rank,
            "lowbound": lowbound,
            "upbound": upbound,
            "LIS": LIS,
            "rank": rank,
            "mu_x": self.mu_x,
            "gamma_x": self.gamma_x,
            "noisecov": self.noisecov,
            "yobs": self.yobs,
            "fm": self.fm,
            "geom": self.geom,
            "linop": self.linop,
            "mcmcDir": self.mcmcDir,
            "thinning": self.thinning
            }
        self.mcmc = MCMCLIS(self.mcmcConfig)
        self.saveMCMCConfig()

    def initPriorSampling(self, rank=427):
        self.mcmcConfig = {
            "x0": np.zeros(rank),
            "Nsamp": self.Nsamp,
            "burn": self.burn,
            "sd": 2.38 ** 2 / rank,
            "propcov": np.identity(rank) * (2.38 ** 2) / rank,
            "LIS": False,
            "rank": rank,
            "mu_x": np.zeros(rank),
            "gamma_x": np.identity(rank),
            "noisecov": self.noisecov,
            "yobs": self.yobs,
            "fm": self.fm,
            "geom": self.geom,
            "linop": self.linop,
            "mcmcDir": self.mcmcDir
            }
        self.mcmc = MCMCLIS(self.mcmcConfig)

        self.mu_x = np.zeros(rank)
        self.gamma_x = np.identity(rank)

    def runAM(self):
        self.mcmc.adaptm(self.alg)   

    def saveMCMCConfig(self):
        # np.save(self.mcmcDir + 'wavelength.npy', self.wavelengths)
        np.save(self.mcmcDir + 'radiance.npy', self.yobs)
        np.save(self.mcmcDir + 'truth.npy', self.truth)
        np.save(self.mcmcDir + 'bands.npy', self.bands)
        np.save(self.mcmcDir + 'mu_x.npy', self.mu_x)
        np.save(self.mcmcDir + 'gamma_x.npy', self.gamma_x)
        np.save(self.mcmcDir + 'isofitMuPos.npy', self.mupos_isofit)
        np.save(self.mcmcDir + 'isofitGammaPos.npy', self.gammapos_isofit)
        np.save(self.mcmcDir + 'Nsamp.npy', self.Nsamp)
        np.save(self.mcmcDir + 'burn.npy', self.burn)
        np.save(self.mcmcDir + 'thinning.npy', self.thinning)

    def calcMeanCov(self):
        self.MCMCmean, self.MCMCcov = self.mcmc.calcMeanCov()
        return self.MCMCmean, self.MCMCcov 

    def autocorr(self, ind):
        return self.mcmc.autocorr(ind)
    '''
    def drawEllipse(self, mean, cov, ax, colour):
        # Helper function for twoDimVisual 

        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor='None', edgecolor=colour)
        scale_x = np.sqrt(cov[0, 0]) * 1
        scale_y = np.sqrt(cov[1, 1]) * 1 
        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean[0], mean[1])
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse) 
        
    def twoDimVisual(self, indX, indY):
        x_vals = np.load(self.mcmcDir + 'MCMC_x.npy')

        fig, ax = plt.subplots()
        if indX < self.nx-2 and indY < self.nx-2:
            ax.plot(self.truth[indX], self.truth[indY], 'ro', label='True reflectance', markersize=10)     
        ax.scatter(x_vals[indX,:], x_vals[indY,:], c='cornflowerblue', s=0.5)
        
        # plot Isofit mean/cov
        meanIsofit = np.array([self.mupos_isofit[indX], self.mupos_isofit[indY]])
        covIsofit = self.gammapos_isofit[np.ix_([indX,indY],[indX,indY])]
        ax.plot(meanIsofit[0], meanIsofit[1], 'kx', label='Isofit posterior', markersize=12)
        self.drawEllipse(meanIsofit, covIsofit, ax, colour='red')
        
        # plot MCMC mean/cov
        meanMCMC = np.array([self.MCMCmean[indX], self.MCMCmean[indY]])
        covMCMC = self.MCMCcov[np.ix_([indX,indY],[indX,indY])]
        ax.plot(meanMCMC[0], meanMCMC[1], 'bx', label='MCMC posterior', markersize=12)
        self.drawEllipse(meanMCMC, covMCMC, ax, colour='blue')
        
        ax.set_title('MCMC - Two Component Visual')
        ax.set_xlabel('Index ' + str(indX))
        ax.set_ylabel('Index ' + str(indY))
        ax.legend()

        return fig

    def kdcontour(self, indX, indY):
        x_vals = np.load(self.mcmcDir + 'MCMC_x.npy')
        x_vals_plot = x_vals[:,self.burn:]

        x = x_vals_plot[indX,:]
        y = x_vals_plot[indY,:]

        isofitPosX = self.mupos_isofit[indX]
        isofitPosY = self.mupos_isofit[indY]
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
        cset = ax.contour(xx, yy, f, levels=levs, colors='k') ##############################ADD INLINE
        plt.clabel(cset, levs, fontsize='smaller')

        # plot truth, isofit, and mcmc mean
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
            ax.plot(self.truth[indX], self.truth[indY], 'go', label='Truth', markersize=8)  
        else:
            if indX == self.nx-2:
                ax.set_xlabel('AOD550')
                ax.set_ylabel('H20STR')
        ax.legend()
        fig.colorbar(cfset)
        return fig

    def plot2Dmarginal(self, indset1=[100,250,410], indset2=[30,101,260]):
        
        n = len(indset1)
        m = len(indset2)
        fig, ax = plt.subplots(n, m)
        
        for i in range(n):
            for j in range(m):
                indX = indset1[i]
                indY = indset2[j]

                ax[j] = self.twoDimVisual(indY, indX, ax[j])
                # ax[i,j].set_title('CHANGE TITLE')
                # ax[i,j].set_xlabel('Wavelength Channel ' + str(self.wavelengths[indY]))
                # ax[i,j].set_ylabel('Wavelength Channel ' + str(self.wavelengths[indX]))
        # ax.set_title('2D Marginal Plots ')
        # fig.savefig(self.mcmcDir + '2Dmarginal.png', dpi=300)

        ax[0].set_ylabel(r'$\lambda = $' + str(self.wavelengths[indset1[0]]) + ' nm')
        # ax[1,0].set_ylabel(r'$\lambda = $' + str(self.wavelengths[indset1[1]]) + ' nm')
        # ax[2,0].set_ylabel(r'$\lambda = $' + str(self.wavelengths[indset1[2]]) + ' nm')

        ax[0].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[0]]) + ' nm')
        ax[1].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[1]]) + ' nm')
        ax[2].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[2]]) + ' nm')
        # ax[2,0].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[0]]) + ' nm')
        # ax[2,1].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[1]]) + ' nm')
        # ax[2,2].set_xlabel(r'$\lambda = $' + str(self.wavelengths[indset2[2]]) + ' nm')
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')

    def mcmcPlots(self):

        self.plot2Dmarginal(indset1=[100,250,410], indset2=[30,101,260])
        self.diagnostics(indset=[30,40,90,100,150,160,250,260])
        self.kdcontour(100, 150)
        self.kdcontour(self.nx-2, self.nx-1)

    def diagnostics(self, indSet=[10,20,50,100,150,160,250,260,425,426]):
        # assume there are 10 elements in indSet
        # default: indSet = [10,20,50,100,150,160,250,260,425,426]
        if self.nx-2 not in indSet:
            indSet = indSet.extend([self.nx-2, self.nx-1]) 

        x_vals = np.load(self.mcmcDir + 'MCMC_x.npy')
        logpos = np.load(self.mcmcDir + 'logpos.npy')
        acceptance = np.load(self.mcmcDir + 'acceptance.npy')
        numPairs = int(len(indSet) / 2) 

        # plot 2D visualization
        # for i in range(numPairs):
        #     fig = self.twoDimVisual(indX=indSet[2*i], indY=indSet[2*i+1])
        #     fig.savefig(self.mcmcDir + '2D_' + str(indSet[2*i]) + '-' + str(indSet[2*i+1]) + '.png', dpi=300)
        #     fig = self.kdcontour(indX=indSet[2*i], indY=indSet[2*i+1])
        #     fig.savefig(self.mcmcDir + 'contour_' + str(indSet[2*i]) + '-' + str(indSet[2*i+1]) + '.png', dpi=300)

        # subplot setup
        fig1, axs1 = plt.subplots(numPairs, 2)
        fig2, axs2 = plt.subplots(numPairs, 2)
        xPlot = np.zeros(numPairs * 2, dtype=int)
        yPlot = np.zeros(numPairs * 2, dtype=int)
        xPlot[::2] = range(numPairs)
        xPlot[1::2] = range(numPairs)
        yPlot[1::2] = 1

        for i in range(len(indSet)):
            print('Diagnostics:',indSet[i])
            x_elem = x_vals[indSet[i],:]
            xp = xPlot[i]
            yp = yPlot[i]

            # plot trace
            axs1[xp,yp].plot(range(self.Nsamp), x_elem)
            axs1[xp,yp].set_title('Trace - Index ' + str(indSet[i]))

            # plot autocorrelation
            ac = self.autocorr(indSet[i])
            ac = ac[:int(len(ac)/2)]
            axs2[xp,yp].plot(range(1,len(ac)+1) * self.thinning, ac)
            axs2[xp,yp].set_title('Autocorrelation - Index ' + str(indSet[i]))

        fig1.set_size_inches(5, 7)
        fig1.tight_layout()
        fig1.savefig(self.mcmcDir + 'trace.png', dpi=300)
        fig2.set_size_inches(5, 7)
        fig2.tight_layout()
        fig2.savefig(self.mcmcDir + 'autocorr.png', dpi=300)
        
        # plot logpos
        plt.figure()
        plt.plot(logpos)
        plt.xlabel('Number of Samples')
        plt.ylabel('Log Posterior')
        plt.savefig(self.mcmcDir + 'logpos.png', dpi=300)

        # acceptance rate
        # print('Acceptance rate:', )
        acceptRate = np.mean(acceptance[self.burn:])
        binWidth = 1000
        numBin = int(self.Nsamp / binWidth)
        xPlotAccept = np.arange(binWidth, self.Nsamp+1, binWidth)
        acceptPlot = np.zeros(numBin)
        for i in range(numBin):
            acceptPlot[i] = np.mean(acceptance[binWidth*i : binWidth*(i+1)])
        plt.figure()
        plt.plot(xPlotAccept, acceptPlot)
        plt.xlabel('Number of Samples')
        plt.ylabel('Acceptance Rate')
        plt.title('Acceptance Rate = ' + str(acceptRate))
        plt.ylim([0, 1])
        plt.savefig(self.mcmcDir + 'acceptance.png', dpi=300)
        
    '''
