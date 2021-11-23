import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

class MCMCLIS:
    '''
    Contains functions to perform MCMC sampling
    Integrating LIS capabilities
    '''

    def __init__(self, config):

        self.unpackConfig(config)
        self.nx = self.gamma_x.shape[0] # parameter dimension
        self.ny = self.noisecov.shape[0] # data dimension
        self.nComp = self.nx - self.rank

        # initialize chain and proposal covariance
        # self.x0 = self.x0 - self.mu_x
        # self.x0 = self.x0 - self.MAP
        self.x0 = np.zeros(self.rank)

        self.invGammaX = np.linalg.inv(self.gamma_x)
        self.invNoiseCov = np.linalg.inv(self.noisecov)

        if self.LIS == True:
            # compute projection matrices
            self.phi, self.theta, self.proj, self.phiComp, self.thetaComp, self.projComp = self.LISproject() 

            # project x0 and proposal covariance to LIS subspace
            # self.x0 = self.theta.T @ self.x0
            self.propcov = self.theta.T @ self.propcov @ self.theta

    def unpackConfig(self, config):
        # self.x0 = config["x0"]              # initial value of chain
        self.startX = config["startX"]      # initial value of chain
        self.Nsamp = config["Nsamp"]        # total number of MCMC samples
        self.burn = config["burn"]          # number of burn-in samples
        self.sd = config["sd"]              # proposal covariance parameter
        self.propcov = config["propcov"]    # initial proposal covariance
        self.lowbound = config["lowbound"]  # lower constraint for parameters
        self.upbound = config["upbound"]    # upper constraint for parameters
        self.LIS = config["LIS"]            # LIS or no LIS
        self.rank = config["rank"]          # rank of problem
        self.mu_x = config["mu_x"]          # prior mean
        self.gamma_x = config["gamma_x"]    # prior covariance
        self.noisecov = config["noisecov"]  # data noise covariance
        self.fm = config["fm"]              # forward model
        self.geom = config["geom"]          # geometry model
        self.linop = config["linop"]        # linear operator
        self.yobs = config["yobs"]          # radiance observation
        self.mcmcDir = config["mcmcDir"]    # directory to save data
        self.thinning = config["thinning"]# only save every ___ sample

    def LISproject(self):
        ### Compute LIS projection matrices ###

        # compute Hessian, solve eigenvalue problem
        cholPr = np.linalg.cholesky(self.gamma_x) # cholesky decomp of prior covariance
        H = self.linop.T @ self.invNoiseCov @ self.linop # Hessian
        Hn = cholPr.T @ H @ cholPr 
        V = self.solveEig(Hn, plot=False, title='LIS Eigenvalue Decay')
        
        # LIS subspace
        VLIS = V[:,:self.rank] 
        phi = cholPr @ VLIS
        theta = np.linalg.inv(cholPr.T) @ VLIS
        proj = phi @ theta.T

        # complementary subspace
        VComp = V[:,self.rank:]
        phiComp = cholPr @ VComp
        thetaComp = np.linalg.inv(cholPr.T) @ VComp
        projComp = phiComp @ thetaComp.T

        return phi, theta, proj, phiComp, thetaComp, projComp

    def solveEig(self, matrix, plot=False, title='Eigenvalue Decay'):
        # solve eigenvalue problem, sort, plot
        print('Solving generalized eigenvalue problem...')
        eigvec, eigval, p = np.linalg.svd(matrix)
        idx = eigval.argsort()[::-1]
        eigval = eigval[idx]
        if plot == True:
            plt.semilogy(eigval, 'b')
            plt.title(title)
            plt.show()
        return eigvec[:,idx]

    def logpos(self, x):
        ''' Calculate log posterior '''
        # input x is zero-mean in LIS parameter space
        if self.LIS == True:
            # logprior = -1/2 * x.dot(x)
            tPr = x + self.theta.T @ (self.startX - self.mu_x)
            logprior = -1/2 * tPr.dot(tPr)
            xFull = self.phi @ x + self.startX
        else:
            tPr = x + self.startX - self.mu_x
            logprior = -1/2 * (tPr @ self.invGammaX @ tPr.T) 
            xFull = x + self.startX

        meas = self.fm.calc_rdn(xFull, self.geom) # apply forward model
        tLH = self.yobs - meas
        loglikelihood = -1/2 * (tLH @ self.invNoiseCov @ tLH.T)

        # for fixed atm
        # xFull = np.concatenate((x, [0.05,1.75]))
        # meas = self.fm.calc_rdn(xFull, self.geom) # apply forward model
        # tLH = self.yobs - meas
        # loglikelihood = -1/2 * tLH.dot(np.linalg.solve(self.noisecov, tLH))
        
        return logprior + loglikelihood 

    def proposal(self, mean, covCholesky):
        ''' Sample proposal from a normal distribution '''
        zx = np.random.normal(0, 1, size=mean.size)
        z = mean + covCholesky @ zx
        return z

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
        accept = np.zeros(self.Nsamp, dtype=int)

        x = self.x0
        xComp = np.zeros(self.nComp)
        propChol = np.linalg.cholesky(self.propcov) # cholesky decomp of proposal cov
        eps = 1e-10
        gamma = 0.01

        for i in range(self.Nsamp):
            kl = 0
            constraintSatisfy = False
            while(constraintSatisfy == False):
                z = self.proposal(x, propChol)
                alpha, logposZ, logposX = self.alpha(x, z)

                # add component and check constraint
                zComp = self.proposal(np.zeros(self.nComp), np.identity(self.nComp))
                xFull = self.phi @ x + self.phiComp @ zComp + self.startX
                kl = kl + 1
                
                constraintSatisfy = self.checkConstraint(xFull)
            print(kl)
            # if self.checkConstraint(xFull) == False:
            #     alpha = 0
            #     i = i - 1

            if np.random.random() < alpha:
                x = z 
                xComp = zComp
                logposX = logposZ
                accept[i] = 1

            x_vals[:,i] = x
            x_vals_comp[:,i] = xComp 
            logpos[i] = logposX
            
            # print progress
            if (i+1) % 500 == 0: 
                print('Sample: ', i+1)
                print('   Accept Rate: ', np.mean(accept[i-499:i]))
                propChol = np.linalg.cholesky(self.propcov) # update chol of propcov
                # print(np.linalg.norm(propChol))
                sys.stdout.flush()
                # flush 

            # change proposal covariance
            if i == 999:
                self.propcov = self.sd * (np.cov(x_vals[:,:1000]) + eps * np.identity(len(x)))
                meanXprev = np.mean(x_vals[:,:1000],1)
            elif i >= 1000:
                meanX = i / (i + 1) * meanXprev + 1 / (i + 1) * x_vals[:,i]
                self.propcov = (i-1) / i * self.propcov + self.sd / i * (i * np.outer(meanXprev, meanXprev) - (i+1) * np.outer(meanX, meanX) + np.outer(x_vals[:,i], x_vals[:,i]) + eps * np.identity(len(x)))
                meanXprev = meanX

        # post processing, store MCMC chain
        if self.LIS == True:
            x_vals_full = self.phi @ x_vals + self.phiComp @ x_vals_comp
        else:
            x_vals_full = x_vals
        x_vals_full = x_vals_full + np.outer(self.startX, np.ones(self.Nsamp))
        
        np.save(self.mcmcDir + 'MCMC_x.npy', x_vals_full[:,::self.thinning])
        np.save(self.mcmcDir + 'logpos.npy', logpos[::self.thinning])
        np.save(self.mcmcDir + 'acceptance.npy', accept[::self.thinning])
    
    def plotProposal(self, z):
        if self.LIS == True:
            plt.plot(self.phi @ z + self.startX)
        else:
            # plt.plot(z + self.mu_x)
            plt.plot(z + self.startX)
        plt.ylim([-0.1, 0.8])
        plt.show(block=False)
        plt.pause(0.001)
        plt.close()

    def calcMeanCov(self):
        x_vals = np.load(self.mcmcDir + 'MCMC_x.npy')
        x_ref = x_vals[:, self.burn:]
        nx = x_ref.shape[0]
        mean = np.mean(x_ref, axis=1)
        cov = np.cov(x_ref)
        return mean, cov

    def autocorr(self, ind):
        x_vals = np.load(self.mcmcDir + 'MCMC_x.npy')
        x_elem = x_vals[ind,:]
        Nsamp = int(min(self.Nsamp, 20000))
        meanX = np.mean(x_elem)
        varX = np.var(x_elem)
        ac = np.zeros(Nsamp-1)

        for k in range(Nsamp-1):
            cov = np.cov(x_elem[:Nsamp-k], x_elem[k:Nsamp])
            ac[k] = cov[1,0] / varX
        return ac


        
