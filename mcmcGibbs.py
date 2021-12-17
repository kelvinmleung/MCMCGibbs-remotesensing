import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

class MCMCGibbs:
    '''
    Contains functions to perform MCMC sampling
    Integrating LIS capabilities
    '''

    def __init__(self, config):

        self.unpackConfig(config)
        self.nx = self.gamma_x.shape[0] # parameter dimension
        self.ny = self.noisecov.shape[0] # data dimension
        self.nComp = self.nx - self.rank

        self.x0 = np.zeros(self.rank) # initialize chain at zero

        self.invGammaX = np.linalg.inv(self.gamma_x)
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


    def proposal(self, mean, covCholesky):
        ''' Sample proposal from a normal distribution '''
        zx = np.random.normal(0, 1, size=mean.size)
        z = mean + covCholesky @ zx
        return z

    def logpos(self, x):
        ''' Calculate log posterior '''
        # input x is zero-mean in LIS parameter space
        
        xFull = x + self.startX
        tPr = xFull - self.mu_x
        logprior = -1/2 * (tPr @ self.invGammaX @ tPr.T) 
        
        meas = self.fm.calc_rdn(xFull, self.geom) # apply forward model
        tLH = self.yobs - meas
        loglikelihood = -1/2 * (tLH @ self.invNoiseCov @ tLH.T)

        return logprior + loglikelihood 
        
    def alpha(self, x, z):
        ''' Calculate acceptance ratio '''
        logposZ = self.logpos(z)
        logposX = self.logpos(x)
        ratio = logposZ - logposX
        # return both acceptance ratio and logpos
        return np.minimum(1, np.exp(ratio)), logposZ, logposX

    def checkConstraint(self, x):
        # x needs to have dimension = nx
        checkA = any(x[i] < self.lowbound[i] for i in range(self.nx)) 
        checkB = any(x[i] > self.upbound[i] for i in range(self.nx)) 
        if checkA or checkB:
            return False
        return True

    def adaptm(self, alg):
        ''' Run Adaptive-Metropolis MCMC algorithm '''
        x_vals = np.zeros([self.rank, self.Nsamp]) # store all samples
        x_vals_comp = np.zeros([self.nComp, self.Nsamp])

        logpos = np.zeros(self.Nsamp) # store the log posterior values
        acceptAtm = np.zeros(self.Nsamp, dtype=int)
        acceptRef = np.zeros(self.Nsamp, dtype=int)

        x = self.x0
        # xRef = x[:-2]
        # xAtm = x[-2:]
        self.propcovAtm = self.propcov[-2:,-2:]
        propCholAtm = np.linalg.cholesky(self.propcovAtm)
        
        eps = 1e-10
        gamma = 0.01
        
        mu_x = self.mu_x[:-2]
        gamma_x = self.gamma_x[:-2,:-2]

        for i in range(self.Nsamp):
            # kl = 0
            # constraintSatisfy = False
            # while(constraintSatisfy == False):
            #     z = self.proposal(x, propChol)
            #     alpha, logposZ, logposX = self.alpha(x, z)

            #     # add component and check constraint
            #     zComp = self.proposal(np.zeros(self.nComp), np.identity(self.nComp))
            #     xFull = self.phi @ x + self.phiComp @ zComp + self.startX
            #     kl = kl + 1
                
            #     constraintSatisfy = self.checkConstraint(xFull)
            # print(kl)

            zAtm = np.copy(x)
            zAtm[-2:] = self.proposal(xAtm, propCholAtm)
            alphaAtm, logposZ, logposX = self.alpha(x, zAtm)

            if np.random.random() < alphaAtm:
                x = zAtm # change only the atm parameters in x
                logposX = logposZ
                acceptAtm[i] += 1

            xAtm = x[-2:] + self.startX[-2:]
            ## CALCULATE G USING xAtm
            
            gamma_y = G @ gamma_x @ G.T + self.noisecov
            inv_gamma_y = np.linalg.inv(gamma_y)
            gamma_refl = (np.identity(self.ny) - gamma_x @ G.T @ inv_gamma_y @ G) @ gamma_x
            mu_refl = gamma_refl @ (G.T @ self.noisecov @ self.yobs) + gamma_x @ mu_x

            zRef = np.copy(x)
            zRef[:-2] = self.proposal(mu_refl, np.linalg.cholesky(gamma_refl))
            alphaRef, logposZ, logposX = self.alpha(x, zRef)

            if np.random.random() < alphaRef:
                x = zRef # change the reflectances of x
                logposX = logposZ
                acceptRef[i] += 1


            x_vals[:,i] = x
            
            logpos[i] = logposX
            
            # print progress
            if (i+1) % 500 == 0: 
                print('Sample: ', i+1)
                print('   Atm Accept Rate: ', np.mean(acceptAtm[i-499:i]))
                print('   Ref Accept Rate: ', np.mean(acceptRef[i-499:i]))
                propCholAtm = np.linalg.cholesky(self.propcovAtm) # update chol of propcov
                sys.stdout.flush()

            # change proposal covariance
            if i == 999:
                self.propcovAtm = self.sd * (np.cov(x_vals[-2:,:1000]) + eps * np.identity(len(x)))
                meanXprev = np.mean(x_vals[-2:,:1000],1)
            elif i >= 1000:
                meanX = i / (i + 1) * meanXprev + 1 / (i + 1) * x_vals[-2:,i]
                self.propcovAtm = (i-1) / i * self.propcovAtm + self.sd / i * (i * np.outer(meanXprev, meanXprev) - (i+1) * np.outer(meanX, meanX) + np.outer(x_vals[:,i], x_vals[:,i]) + eps * np.identity(len(x)))
                meanXprev = meanX

        x_vals_full = x_vals + np.outer(self.startX, np.ones(self.Nsamp))
        
        np.save(self.resultsDir + 'mcmcchain.npy', x_vals_full[:,::self.thinning])
        np.save(self.resultsDir + 'logpos.npy', logpos[::self.thinning])
        # np.save(self.resultsDir + 'acceptance.npy', accept[::self.thinning])
    
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


        
