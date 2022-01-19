import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

class MCMCGibbs:
    '''
    Contains functions to perform MCMC sampling with approximate Gibbs
    '''

    def __init__(self, config):

        self.unpackConfig(config)
        self.nx = self.gamma_x.shape[0] # parameter dimension
        self.ny = self.noisecov.shape[0] # data dimension
        # self.nComp = self.nx - self.rank

        # self.x0 = np.zeros(self.rank) # initialize chain at zero
        # self.x0 = np.zeros(self.nx) # initialize chain at zero
        self.x0 = self.startX

        self.invGammaX = np.linalg.inv(self.gamma_x)
        self.invGammaX_ref = np.linalg.inv(self.gamma_x[:-2,:-2])
        self.invNoiseCov = np.linalg.inv(self.noisecov)

    def unpackConfig(self, config):
        # self.x0 = config["x0"]              # initial value of chain
        self.startX = config["startX"]      # initial value of chain
        self.Nsamp = config["Nsamp"]        # total number of MCMC samples
        self.burn = config["burn"]          # number of burn-in samples
        self.sd = config["sd"]              # proposal covariance parameter
        self.propcov = config["propcov"]    # initial proposal covariance
        self.lowbound = config["lowbound"]  # lower constraint for parameters
        self.upbound = config["upbound"]    # upper constraint for parameters
        self.rank = config["rank"]          # rank of problem
        self.mu_x = config["mu_x"]          # prior mean
        self.gamma_x = config["gamma_x"]    # prior covariance
        self.noisecov = config["noisecov"]  # data noise covariance
        self.fm = config["fm"]              # forward model
        self.geom = config["geom"]          # geometry model
        self.yobs = config["yobs"]          # radiance observation
        self.resultsDir = config["resultsDir"]    # directory to save data
        self.thinning = config["thinning"]# only save every ___ sample
        self.luts = config["luts"]          # look up tables for LUT parameters
        self.truth = config["truth"]
        self.bands = config["bands"]
        self.bandsX = config["bandsX"]

    def lookupLUT(self, point):
        ''' Returns the lookup table results for a atm point given self.luts (lookup table copied from isofit files) '''
        ret = {}
        for key, lut in self.luts.items():
            ret[key] = np.array(lut(point)).ravel()
        return ret

    def unpackLUTparam(self, atm):
        ''' Obtain relevant LUT parameters '''
        lutparam = self.lookupLUT(atm)
        rhoatm = lutparam['rhoatm']
        sphalb = lutparam['sphalb']
        transm = lutparam['transm']
        coszen = np.load('coszen.npy')   
        solar_irr = np.load('solar_irr.npy')   
        return rhoatm, sphalb, transm, coszen, solar_irr

    def linOper(self, sphalb, transm, coszen, solar_irr): # Conditioned on the atmospheric parameters!
        xMAP = self.startX[:self.ny]
        G = coszen / np.pi * np.diag(solar_irr * transm / (1 - sphalb * xMAP))
        return G

    def proposalAtm(self, mean, std):
        ''' Sample proposal from truncated normal '''
        a = (self.lowbound[-2:] - mean)/std
        b = (self.upbound[-2:] - mean)/std
        z = mean + std * truncnorm.rvs(a,b)
        return z

    # def proposalRef(self, mean, cov):
    #     ''' Sample proposal from a normal distribution '''
    #     zx = np.random.normal(0, 1, size=mean.size)
    #     z = mean + np.linalg.cholesky(cov) @ zx
        
    #     return z

    def proposal(self, mean, covCholesky):
        ''' Sample proposal from a normal distribution '''
        
        zx = np.random.normal(0, 1, size=mean.size)
        z = mean + covCholesky @ zx
        
        return z

    def logpos(self, x):
        ''' Calculate log posterior '''
        
        tPr = x - self.mu_x
        logprior = -1/2 * (tPr @ self.invGammaX @ tPr.T) 
        # logprior = -1/2 * (tPr[self.bandsX] @ self.invGammaX[self.bandsX,:][:,self.bandsX] @ tPr[self.bandsX].T) 
        # 
        meas = self.fm.calc_rdn(x, self.geom) # apply forward model
        tLH = self.yobs - meas
        # loglikelihood = -1/2 * (tLH @ self.invNoiseCov @ tLH.T)
        loglikelihood = -1/2 * (tLH[self.bands] @ self.invNoiseCov[self.bands,:][:,self.bands] @ tLH[self.bands].T)
        
        # print('\tLogPrior/LLH', logprior, loglikelihood)
        # # print('\t', logprior)
        # print('\ttPrNorm', np.linalg.norm(tPr))
        # print('\ttLHNorm', np.linalg.norm(tLH[self.bands]))


        # plt.plot(self.mu_x[self.bands], '.', label='Prior')
        # plt.plot(x[self.bands], '.',label='x')
        # plt.legend()
        # plt.show()

        # plt.plot(meas[self.bands], '.', label='Predicted')
        # plt.plot(self.yobs[self.bands], '.',label='Observed')
        # plt.legend()
        # plt.show()

        return logprior + loglikelihood

    def lognormal(self, x, mu, cov):
        ''' Calculate the log normal density '''
        return -1/2 * (x - mu).T @ np.linalg.inv(cov) @ (x - mu)
        
    def alpha(self, x, z):
        ''' Calculate acceptance ratio '''
        logposZ = self.logpos(z)
        logposX = self.logpos(x)
        ratio = logposZ - logposX

        # return both acceptance ratio and logpos
        return np.minimum(1, np.exp(ratio)), logposZ, logposX

    def alphaAsym(self, x, z, mu, cov):
        ''' Calculate acceptance ratio for asymmetric proposal '''
        logposZ = self.logpos(z)
        logposX = self.logpos(x)
        logqZ = self.lognormal(z[:-2], mu, cov)
        logqX = self.lognormal(x[:-2], mu, cov)
        ratio = logposZ - logposX - logqZ + logqX
        print('\t %10.3f | %10.3E %10.3E %10.3E %10.3E' % (ratio, logposZ, logposX, logqZ, logqX))

        # return both acceptance ratio and logpos
        return np.minimum(1, np.exp(ratio)), logposZ, logposX

    def checkConstraint(self, x):
        # x needs to have dimension = nx
        checkA = any(x[i] < self.lowbound[i] for i in self.bandsX) 
        checkB = any(x[i] > self.upbound[i] for i in self.bandsX) 
        if checkA or checkB:
            return False
        return True

    # def adaptmAnnotate(self, alg):
    #     ''' Run Adaptive-Metropolis MCMC algorithm - ANNOTATED'''

    #     # initialize vectors
    #     x_vals = np.zeros([self.nx, self.Nsamp]) # store all samples
    #     logpos = np.zeros(self.Nsamp) # store the log posterior values
    #     acceptAtm = np.zeros(self.Nsamp, dtype=int)
    #     acceptRef = np.zeros(self.Nsamp, dtype=int)
        
    #     # get prior reflectance - mean and covariance (no atm) (432x432)
    #     mu_x = self.mu_x[:-2]
    #     gamma_x = self.gamma_x[:-2,:-2]

    #     # initialize proposal covariance for atm only (2x2)
    #     self.propcovAtm = self.propcov[-2:,-2:]
    #     propStdAtm = np.sqrt(np.diag(self.propcovAtm))
    #     eps = 1e-10 # epsilon for proposal covariance calculation

    #     # initialize the chain at the MAP point
    #     x = self.x0       

    #     # obtain the linearized model G given the MAP atm parameters
    #     # yobs_adjust is the adjusted version of y = Gx + const, where yobs_adjust = y - const = Gx
    #     xAtm = x[-2:]  
    #     rhoatm, sphalb, transm, coszen, solar_irr = self.unpackLUTparam(xAtm)
    #     G = self.linOper(sphalb, transm, coszen, solar_irr)
    #     yobs_adjust = self.yobs - coszen / np.pi * solar_irr * rhoatm 

    #     for i in range(self.Nsamp):
            
    #         # create copy of x and propose new atm param
    #         zAtm = np.copy(x)
    #         zAtm[-2:] = self.proposalAtm(x[-2:], propStdAtm)

    #         # check acceptance rate for x with new atm param
    #         alphaAtm, logposZ, logposX = self.alpha(x, zAtm)
    #         if np.random.random() < alphaAtm:

    #             # update atm param of x
    #             x[-2:] = zAtm[-2:]
    #             logposX = logposZ
    #             acceptAtm[i] += 1

    #             # use this atm param to generate the new linearized forward model G, yobs_adjust
    #             xAtm = x[-2:]
    #             rhoatm, sphalb, transm, coszen, solar_irr = self.unpackLUTparam(xAtm)
    #             G = self.linOper(sphalb, transm, coszen, solar_irr)
    #             yobs_adjust = self.yobs - coszen / np.pi * solar_irr * rhoatm 
                
    #             # apply posterior mean and covariance equations
    #             # cholesky factorize the pos covariance for proposal
    #             gamma_refl = np.linalg.inv(G.T @ self.invNoiseCov @ G + self.invGammaX_ref)
    #             mu_refl = gamma_refl @ (G.T @ self.invNoiseCov @ yobs_adjust + self.invGammaX_ref @ mu_x)
    #             chol_gamma_refl = np.linalg.cholesky(gamma_refl) 

    #         # set reflectance to posterior mean OR propose new reflectance using mu_refl, gamma_refl
    #         zRef = np.copy(x)
    #         zRef[:-2] = mu_refl 
    #         # zRef[:-2] = self.proposal(mu_refl, chol_gamma_refl*0.2)

    #         # check acceptance rate
    #         alphaRef, logposZ, logposX = self.alpha(x, zRef)
    #         if np.random.random() < alphaRef:
    #             x[:-2] = zRef[:-2] 
    #             logposX = logposZ
    #             acceptRef[i] += 1
            
    #         x_vals[:,i] = x
    #         logpos[i] = logposX
            
    #         # print progress
    #         if (i+1) % 500 == 0: 
    #             print('Sample: ', i+1)
    #             print('   Atm Accept Rate: ', np.mean(acceptAtm[i-499:i]))
    #             print('   Ref Accept Rate: ', np.mean(acceptRef[i-499:i]))
    #             sys.stdout.flush()

    #         # # change proposal covariance
    #         if i == 999:
    #             self.propcovAtm = self.sd * (np.cov(x_vals[-2:,:1000]) + eps * np.identity(2))
    #             meanXprev = np.mean(x_vals[-2:,:1000],1)
    #         elif i >= 1000:
    #             meanX = i / (i + 1) * meanXprev + 1 / (i + 1) * x_vals[-2:,i]
    #             self.propcovAtm = (i-1) / i * self.propcovAtm + self.sd / i * (i * np.outer(meanXprev, meanXprev) - (i+1) * np.outer(meanX, meanX) + np.outer(x_vals[-2:,i], x_vals[-2:,i]) + eps * np.identity(2))
    #             meanXprev = meanX
                
    #     np.save(self.resultsDir + 'mcmcchain.npy', x_vals[:,::self.thinning])
    #     np.save(self.resultsDir + 'logpos.npy', logpos[::self.thinning])
    

    def adaptm(self, alg):
        ''' Run Adaptive-Metropolis MCMC algorithm '''
        x_vals = np.zeros([self.nx, self.Nsamp]) # store all samples
        # x_vals = np.zeros([self.rank, self.Nsamp]) # store all samples
        # x_vals_comp = np.zeros([self.nComp, self.Nsamp])

        logpos = np.zeros(self.Nsamp) # store the log posterior values
        acceptAtm = np.zeros(self.Nsamp, dtype=int)
        acceptRef = np.zeros(self.Nsamp, dtype=int)

        mu_x = self.mu_x[:-2]
        gamma_x = self.gamma_x[:-2,:-2]

        x = self.x0               
        xAtm = x[-2:]       
        mu_refl = x[:-2]
        chol_gamma_refl = np.linalg.cholesky(gamma_x) 

        self.propcovAtm = self.propcov[-2:,-2:]
        propStdAtm = np.sqrt(np.diag(self.propcovAtm))
        # propCholAtm = np.linalg.cholesky(self.propcovAtm)
        
        eps = 1e-10

        ### Try initializing like this ##############
        rhoatm, sphalb, transm, coszen, solar_irr = self.unpackLUTparam(xAtm)
        G = self.linOper(sphalb, transm, coszen, solar_irr)
        yobs_adjust = self.yobs - coszen / np.pi * solar_irr * rhoatm 
        
        gamma_y = G @ gamma_x @ G.T + self.noisecov
        inv_gamma_y = np.linalg.inv(gamma_y)
        gamma_refl = (np.identity(self.ny) - gamma_x @ G.T @ inv_gamma_y @ G) @ gamma_x
        mu_refl = gamma_refl @ (G.T @ self.invNoiseCov @ yobs_adjust + self.invGammaX_ref @ mu_x)
        chol_gamma_refl = np.linalg.cholesky(gamma_refl) 
        #################################################
        x[:-2] = mu_refl

        # setup = self.loadSetup2()

        
        for i in range(self.Nsamp):

            zAtm = np.copy(x)
            # zAtm[-2:] = self.proposal(x[-2:], propCholAtm)
            zAtm[-2:] = self.proposalAtm(x[-2:], propStdAtm)

            alphaAtm, logposZ, logposX = self.alpha(x, zAtm)

            # print('Atm:', alphaAtm, logposX, logposZ)

            if np.random.random() < alphaAtm:
                
                # self.plotRadiance(zAtm, x)

                x[-2:] = zAtm[-2:] 
                # print('xATM:', xAtm)

                logposX = logposZ
                acceptAtm[i] += 1

                # update linear operator conditioned on the atm
                xAtm = x[-2:]

                rhoatm, sphalb, transm, coszen, solar_irr = self.unpackLUTparam(xAtm)
                G = self.linOper(sphalb, transm, coszen, solar_irr)
                yobs_adjust = self.yobs - coszen / np.pi * solar_irr * rhoatm 
                gamma_refl = np.linalg.inv(G.T @ self.invNoiseCov @ G + self.invGammaX_ref)
                mu_refl = gamma_refl @ (G.T @ self.invNoiseCov @ yobs_adjust + self.invGammaX_ref @ mu_x)
                chol_gamma_refl = np.linalg.cholesky(gamma_refl) 

            factor = 0.14
            # factor=1
            gamma_prop = gamma_refl * (factor ** 2)
            chol_gamma_prop = chol_gamma_refl * factor
            zRef = np.copy(x)
            # zRef[:-2] = mu_refl 
            # zRef[:-2] = self.proposal(mu_refl, chol_gamma_prop)
            zRef[:-2] = self.proposal(x[:-2], chol_gamma_prop)

            # plt.plot(mu_refl, label='mean', alpha=0.8)
            # plt.plot(x[:-2], label='current', alpha=0.6)
            # plt.plot(zRef[:-2], label='proposal', alpha=0.4)
            # plt.ylim([0.2,0.25])
            # plt.legend()
            # plt.show()

            ###### if we want to calculate acceptance in three tries ######
            # refProp = self.proposal(mu_refl, chol_gamma_refl*0.2)
            # cutOff = [5,195,225,295,320,430]
            # for j in range(3):
            #     a = cutOff[j*2]
            #     b = cutOff[j*2+1]
            #     zRef = np.copy(x)
            #     zRef[a:b] = refProp[a:b]

            #     alphaRef, logposZ, logposX = self.alpha(x, zRef)
            #     # print('Refl:', alphaRef, logposX, logposZ)

            #     if np.random.random() < alphaRef:
            #         x[a:b] = zRef[a:b] # change the reflectances of x
            #         acceptRef[i] += 1/3
            ################################################################

            
            # print('xATM:', xAtm)
            alphaRef, logposZ, logposX = self.alpha(x, zRef)
            # alphaRef, logposZ, logposX = self.alphaAsym(x, zRef, mu_refl, gamma_prop)
            # print('Refl:', alphaRef, logposX, logposZ)
            
            if np.random.random() < alphaRef:
                x[:-2] = zRef[:-2] # change the reflectances of x
                logposX = logposZ
                acceptRef[i] += 1
            
            x_vals[:,i] = x
            logpos[i] = logposX
            
            # print progress
            if (i+1) % 500 == 0: 
                print('Sample: ', i+1)
                print('   Atm Accept Rate: ', np.mean(acceptAtm[i-499:i]))
                print('   Ref Accept Rate: ', np.mean(acceptRef[i-499:i]))
                print('xATM:', xAtm)
                propCholAtm = np.linalg.cholesky(self.propcovAtm) # update chol of propcov
                sys.stdout.flush()

            # # change proposal covariance
            if i == 999:
                self.propcovAtm = self.sd * (np.cov(x_vals[-2:,:1000]) + eps * np.identity(2))
                meanXprev = np.mean(x_vals[-2:,:1000],1)
            elif i >= 1000:
                meanX = i / (i + 1) * meanXprev + 1 / (i + 1) * x_vals[-2:,i]
                self.propcovAtm = (i-1) / i * self.propcovAtm + self.sd / i * (i * np.outer(meanXprev, meanXprev) - (i+1) * np.outer(meanX, meanX) + np.outer(x_vals[-2:,i], x_vals[-2:,i]) + eps * np.identity(2))
                meanXprev = meanX
                
        
        np.save(self.resultsDir + 'mcmcchain.npy', x_vals[:,::self.thinning])
        np.save(self.resultsDir + 'logpos.npy', logpos[::self.thinning])
        np.save(self.resultsDir + 'acceptAtm.npy', acceptAtm[::self.thinning])
        np.save(self.resultsDir + 'acceptRef.npy', acceptRef[::self.thinning])
    
    def calcMeanCov(self):
        x_vals = np.load(self.resultsDir + 'MCMC_x.npy')
        x_ref = x_vals[:, self.burn:]
        nx = x_ref.shape[0]
        mean = np.mean(x_ref, axis=1)
        cov = np.cov(x_ref)
        return mean, cov

    def autocorr(self, ind):
        x_vals = np.load(self.resultsDir + 'MCMC_x.npy')
        x_elem = x_vals[ind,:]
        Nsamp = int(min(self.Nsamp, 20000))
        meanX = np.mean(x_elem)
        varX = np.var(x_elem)
        ac = np.zeros(Nsamp-1)

        for k in range(Nsamp-1):
            cov = np.cov(x_elem[:Nsamp-k], x_elem[k:Nsamp])
            ac[k] = cov[1,0] / varX
        return ac

    def loadSetup1(self):
        from fileProcessing import FileProcessing
        from isofitSetup import Setup
        f = FileProcessing(setupDir='setup/' + 'ang20170228')
        f.loadWavelength('data/wavelengths.txt')
        f.loadReflectance('data/beckmanlawn/insitu.txt')
        f.loadRadiance('data/beckmanlawn/ang20171108t184227_data_v2p11_BeckmanLawn.mat')
        f.loadConfig('config/config_inversion.json')
        wv, ref, radiance, config = f.getFiles()
        radiance=0
        setup = Setup(wv, ref, radiance, config, resultsDir='Test1', setupDir='ang20170228')
        return setup

    def loadSetup2(self):
        from fileProcessing import FileProcessing
        from isofitSetup import Setup
        f = FileProcessing(setupDir='setup/' + 'ang20140612')
        f.loadWavelength('data/wavelengths.txt')
        f.loadReflectance('data/177/insitu.txt')
        f.loadRadiance('data/177/ang20140612t215931_data_dump.mat')
        f.loadConfig('config/config_inversion.json')
        wv, ref, radiance, config = f.getFiles()
        radiance=0
        setup = Setup(wv, ref, radiance, config, resultsDir='Test1', setupDir='ang20140612')
        return setup

    def plotRadiance(self, zAtm, x):
        plt.figure()
        meas1 = self.fm.calc_rdn(zAtm, self.geom) 
        meas2 = self.fm.calc_rdn(x, self.geom) 
        plt.plot(self.yobs, label='Observed')
        plt.plot(meas1, label='Proposed')
        plt.plot(meas2, label='Current')
        plt.legend()

    def plotTempRetrieval(self, setup, mu_refl):
        plt.figure()
        isofit_pos, gamma_pos = setup.invModel(self.yobs)
        plt.plot(mu_refl, label='Linear')
        plt.plot(isofit_pos[:-2], label='Isofit')
        plt.plot(self.truth[:-2], label='Truth')
        plt.ylim([0.05,0.3])
        plt.legend()

        
