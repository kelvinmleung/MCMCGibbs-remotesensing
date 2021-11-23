import scipy as s
import numpy as np
import matplotlib.pyplot as plt

class Analysis:
    '''
    Contains functions for eigenanalysis of the linearized forward
    model.
    '''

    def __init__(self, setup, regression):
        
        print('Initializing analysis...')

        # configure directories
        self.sampleDir = setup.sampleDir
        self.regDir = setup.regDir
        self.analysisDir = setup.analysisDir

        # setup parameters
        # self.plotbands = setup.plotbands
        self.wl = setup.wavelengths
        self.reflectance = setup.reflectance
        self.truth = setup.truth
        self.radiance = setup.radianceSim
        self.noisecov = setup.noisecov
        self.bands = setup.bands
        self.bandsX = setup.bandsX

        self.mu_x = setup.mu_x
        self.gamma_x = setup.gamma_x
        self.fm = setup.fm
        self.geom = setup.geom        

        # get data sets
        self.scaleX = regression.X_train
        self.scaleY = regression.Y_train

        # load data sets
        X_train = np.load(self.sampleDir + 'X_train.npy')
        Y_train = np.load(self.sampleDir + 'Y_train.npy')
        self.X = X_train
        self.Y = Y_train

        self.phi_tilde = np.load(self.regDir + 'phi.npy')
        self.nx = self.X.shape[1]
        self.ny = self.Y.shape[1]
        Nsamp = self.X.shape[0]

        self.varX = regression.varX
        self.varY = regression.varY

        # likelihood covariance
        error = self.scaleY -  self.scaleX @ self.phi_tilde.T
        gamma_ygx_tilde = np.cov(error.T)

        # transform tilde to non-tilde
        sigma_x_power = np.diag(self.varX ** -0.5)
        sigma_y_power = np.diag(self.varY ** 0.5)
        self.phi = np.real(sigma_y_power @ self.phi_tilde @ sigma_x_power)
        self.gamma_ygx = np.real(sigma_y_power @ gamma_ygx_tilde @ sigma_y_power)
        # calculate marginal covariance
        self.gamma_y = self.phi @ self.gamma_x @ self.phi.T + self.gamma_ygx

    def eigPCA(self):
        #eigvecPCA, eigvalPCA, s1 = np.linalg.svd(self.gamma_y)

        mu, gamma_xgy = self.posterior(self.radiance)

        # plt.figure()
        # plt.plot(mu[:425])
        eigvecPCA, eigvalPCA, s1 = np.linalg.svd(self.gamma_x - gamma_xgy)
        #eigvecPCA, eigvalPCA, s1 = np.linalg.svd(self.gamma_y)

        idx = eigvalPCA.argsort()[::-1]   
        eigvalPCA = eigvalPCA[idx]
        eigvecPCA = eigvecPCA[:,idx]
        
        return eigvalPCA, eigvecPCA

    def eigLIS(self):
        cholPr = np.linalg.cholesky(self.gamma_x)
        H = self.phi.T @ np.linalg.inv(self.gamma_ygx) @ self.phi
        Hn = cholPr.T @ H @ cholPr
        eigvecLIS, eigvalLIS, s1 = np.linalg.svd(Hn) # z

        eigvecLIS = cholPr @ eigvecLIS # v

        idx = eigvalLIS.argsort()[::-1]   
        eigvalLIS = eigvalLIS[idx]
        eigvecLIS = eigvecLIS[:,idx]

        return eigvalLIS, eigvecLIS

    def eigPlots(self, eigval, eigvec, rank=427, title='LIS'):
        plt.figure()
        plt.semilogy(eigval) # - np.ones(self.nx)
        plt.title(title + ' Eigenvalue Decay')
        plt.grid()

        xPlot = range(self.nx)
        plt.figure()
        for i in range(5):
            plt.plot(xPlot, eigvec[:,i],label=str(i+1))
        plt.title('Leading Eigendirections')
        plt.xlabel('Index')
        plt.legend()
        plt.grid()

        plt.figure()
        ploty = np.zeros(self.nx)
        for i in range(rank):
            ploty = ploty + eigval[i] / (1 + eigval[i]) * (eigvec[:,i] ** 2)
        plt.plot(xPlot[self.nx-2:], ploty[self.nx-2:], 'ro')
        plt.semilogy(xPlot, ploty)
        plt.title('Sum of Leading Eigendirections, rank='+str(rank))
        plt.xlabel('Index')
        plt.ylabel(r'$\frac{\lambda_i}{1 + \lambda_i} v_{ij}^2$')
        plt.grid()

        indSort = ploty.argsort()[::-1]   
        sortPlotY = ploty[indSort]
        sortPlotInd = np.array(xPlot)[indSort]

        plt.figure()
        barWidth = 1
        bars = np.concatenate((sortPlotY[:rank], ploty[self.nx-2:]))
        r1 = np.arange(len(bars))
        # Make the plot
        plt.bar(r1, bars, color='#557f2d', width=barWidth, edgecolor='white')
        plt.yscale('log')
        # Add xticks on the middle of the group bars
        plt.xlabel('Index Number', fontweight='bold')
        plt.ylabel(r'$\lambda_i v_{ij}^2$')
        plt.xticks(range(len(bars)), [str(sortPlotInd[i]) for i in range(rank)] + ['AOD', 'H20'])

    def eigLISdata(self):
        L = np.linalg.cholesky(self.gamma_ygx)
        invL = np.linalg.inv(L)
        eigvecLIS, eigvalLIS, s1 = np.linalg.svd(invL @ self.gamma_y @ invL.T) # z
        eigvecLIS = invL.T @ eigvecLIS
        idx = eigvalLIS.argsort()[::-1]   
        eigvalLIS = eigvalLIS[idx]
        eigvecLIS = eigvecLIS[:,idx]
        
        return eigvalLIS, eigvecLIS

    def eigPCAdata(self):

        eigvecPCA, eigvalPCA, s1 = np.linalg.svd(self.gamma_y)
        idx = eigvalPCA.argsort()[::-1]   
        eigvalPCA = eigvalPCA[idx]
        eigvecPCA = eigvecPCA[:,idx]

        return eigvalPCA, eigvecPCA
    
    def posterior(self, yobs):
        # full linear model
        gamma_xgy = (np.identity(self.nx) - self.gamma_x @ self.phi.T @ np.linalg.inv(self.gamma_y) @ self.phi) @ self.gamma_x
        mu_xgy = gamma_xgy @ (self.phi.T @ np.linalg.inv(self.gamma_ygx) @ yobs + np.linalg.inv(self.gamma_x) @ self.mu_x)
        return mu_xgy, gamma_xgy
    
    def comparePosParam(self, gamma_xgy, maxdim):
        eigvalPCA, eigvecPCA = self.eigPCA()
        eigvalLIS, eigvecLIS = self.eigLIS()

        dims = range(5,maxdim,5)
        mu_errorPCA = np.zeros(len(dims))
        mu_errorLIS = np.zeros(len(dims))
        gamma_errorPCA = np.zeros(len(dims))
        gamma_errorLIS = np.zeros(len(dims))

        mu_PCA = np.zeros(self.nx)
        mu_LIS = np.zeros(self.nx)

        invX = np.linalg.inv(self.gamma_x)
        invYGX = np.linalg.inv(self.gamma_ygx)
        invXGY = np.linalg.inv(gamma_xgy) 
            
        for i in range(len(dims)):
            print('Dimension:', dims[i])
            gamma_xgy_PCA = self.gamma_x
            gamma_xgy_LIS = self.gamma_x

            for j in range(dims[i]):
                gamma_xgy_PCA = gamma_xgy_PCA - eigvalPCA[j] * np.outer(eigvecPCA[:,j], eigvecPCA[:,j].T)
                gamma_xgy_LIS = gamma_xgy_LIS - eigvalLIS[j] / (eigvalLIS[j] + 1) * np.outer(eigvecLIS[:,j], eigvecLIS[:,j].T)
            
            N = 100
            for j in range(N):
                mu_PCA = gamma_xgy_PCA @ (self.phi.T @ invYGX @ self.Y[j,:] + invX @ self.mu_x)
                mu_LIS = gamma_xgy_LIS @ (self.phi.T @ invYGX @ self.Y[j,:] + invX @ self.mu_x)
                
                mu_errorPCA[i] = mu_errorPCA[i] + 1/N * np.abs(mu_PCA - self.X[j,:]).T @ invXGY @ np.abs(mu_PCA - self.X[j,:])
                mu_errorLIS[i] = mu_errorLIS[i] + 1/N * np.abs(mu_LIS - self.X[j,:]).T @ invXGY @ np.abs(mu_LIS - self.X[j,:])
            
            # eigsPCA = s.linalg.eigh(gamma_xgy, gamma_xgy_PCA, eigvals_only=True)
            # eigsLIS = s.linalg.eigh(gamma_xgy, gamma_xgy_LIS, eigvals_only=True)
            eigsPCA = s.linalg.eigh(self.gamma_x, gamma_xgy_PCA, eigvals_only=True)
            eigsLIS = s.linalg.eigh(self.gamma_x, gamma_xgy_LIS, eigvals_only=True)

            for j in range(eigsPCA.size):
                gamma_errorPCA[i] = gamma_errorPCA[i] + (np.log(eigsPCA[j])) ** 2
                gamma_errorLIS[i] = gamma_errorLIS[i] + (np.log(eigsLIS[j])) ** 2

            gamma_errorPCA[i] = np.sqrt(gamma_errorPCA[i])
            gamma_errorLIS[i] = np.sqrt(gamma_errorLIS[i])
        
        np.save(self.analysisDir + 'gamma_errorPCAparam.npy',gamma_errorPCA)
        np.save(self.analysisDir + 'gamma_errorLISparam.npy',gamma_errorLIS)
        np.save(self.analysisDir + 'mu_errorPCAparam.npy', mu_errorPCA)
        np.save(self.analysisDir + 'mu_errorLISparam.npy', mu_errorLIS)


    def comparePosData(self, gamma_xgy, maxdim):
        eigvalPCA, eigvecPCA = self.eigPCAdata()
        eigvalLIS, eigvecLIS = self.eigLISdata()
        dims = range(5,maxdim,5)
        #maxdim = self.nx-2
        mu_errorPCA = np.zeros(len(dims))
        mu_errorLIS = np.zeros(len(dims))
        gamma_errorPCA = np.zeros(len(dims))
        gamma_errorLIS = np.zeros(len(dims))
        # PCA and LIS
        for i in range(len(dims)):
            print('Dimension:', dims[i])
            W = eigvecPCA[:,:dims[i]]
            V = eigvecLIS[:,:dims[i]]
            phi_PCA = self.phi.T @ W
            phi_LIS = self.phi.T @ V

            invX = np.linalg.inv(self.gamma_x)
            invYGX_PCA = np.linalg.inv(W.T @ self.gamma_ygx @ W)
            invYGX_LIS = np.linalg.inv(V.T @ self.gamma_ygx @ V)
            invXGY = np.linalg.inv(gamma_xgy) 
            
            gamma_xgy_PCA = np.linalg.inv(phi_PCA @ invYGX_PCA @ phi_PCA.T + invX)
            gamma_xgy_LIS = np.linalg.inv(phi_LIS @ invYGX_LIS @ phi_LIS.T + invX)
            
            # monte carlo for mu_xgy
            mu_PCA = np.zeros(self.nx)
            mu_LIS = np.zeros(self.nx)

            N = 100
            print('\tRunning Bayes risk calculation...')
            for j in range(N):
                #print('\tMonte Carlo: Iteration', j+1)
                mu_PCA = gamma_xgy_PCA @ (phi_PCA @ invYGX_PCA @ (W.T @ self.Y[j,:]) + invX @ self.mu_x)
                mu_LIS = gamma_xgy_LIS @ (phi_LIS @ invYGX_LIS @ (V.T @ self.Y[j,:]) + invX @ self.mu_x)
                
                mu_errorPCA[i] = mu_errorPCA[i] + 1/N * np.abs(mu_PCA - self.X[j,:]).T @ invXGY @ np.abs(mu_PCA - self.X[j,:])
                mu_errorLIS[i] = mu_errorLIS[i] + 1/N * np.abs(mu_LIS - self.X[j,:]).T @ invXGY @ np.abs(mu_LIS - self.X[j,:])
            print('\tBayes risk completed.')
            
            gammaxgy = gamma_xgy
            eigsPCA = s.linalg.eigh(gammaxgy, gamma_xgy_PCA, eigvals_only=True)
            eigsLIS = s.linalg.eigh(gammaxgy, gamma_xgy_LIS, eigvals_only=True)

            for j in range(eigsPCA.size):
                gamma_errorPCA[i] = gamma_errorPCA[i] + (np.log(eigsPCA[j])) ** 2
                gamma_errorLIS[i] = gamma_errorLIS[i] + (np.log(eigsLIS[j])) ** 2

            gamma_errorPCA[i] = np.sqrt(gamma_errorPCA[i])
            gamma_errorLIS[i] = np.sqrt(gamma_errorLIS[i])

        plt.figure()
        plt.semilogy(dims, gamma_errorPCA, label='PCA')
        plt.semilogy(dims, gamma_errorLIS, label='LIS')
        plt.xlabel('Dimension of Subspace')
        plt.ylabel('Forstner Distance')
        plt.title('Forstner Distance in Posterior Covariance')
        plt.legend()
        plt.grid()
        
        plt.figure()
        plt.semilogy(dims, mu_errorPCA, label='PCA')
        plt.semilogy(dims, mu_errorLIS, label='LIS')
        plt.xlabel('Dimension of Subspace')
        plt.ylabel('Bayes Risk')
        plt.title('Bayes Risk in Posterior Mean')
        plt.legend()
        plt.grid()
        
        np.save(self.analysisDir + 'gamma_errorPCAdata.npy',gamma_errorPCA)
        np.save(self.analysisDir + 'gamma_errorLISdata.npy',gamma_errorLIS)
        np.save(self.analysisDir + 'mu_errorPCAdata.npy', mu_errorPCA)
        np.save(self.analysisDir + 'mu_errorLISdata.npy', mu_errorLIS)
        
    '''
    def posterior_plots(self, dim, eigvecPCA, eigvecLIS, gamma_xgy, mu_xgy):
        # plot for dim = 30
        #dim = 30
        
        W = eigvecPCA[:,:dim]
        V = eigvecLIS[:,:dim]
        phi_PCA = self.phi @ W
        phi_LIS = self.phi @ V

        gamma_xgy_PCA = np.linalg.inv(phi_PCA @ np.linalg.inv(W.T @ self.gamma_ygx @ W) @ phi_PCA.T + np.linalg.inv(self.gamma_x))
        
        gamma_xgy_LIS = np.linalg.inv(phi_LIS @ np.linalg.inv(V.T @ self.gamma_ygx @ V) @ phi_LIS.T + np.linalg.inv(self.gamma_x))

        # Maybe need to plot the mu_xgy calculated using self.radiance (radiance "truth")
    '''
    def plotcontour(self, gamma, title):
        X_plot = np.arange(1,self.ny+1,1)
        Y_plot = np.arange(1,self.ny+1,1)
        X_plot, Y_plot = np.meshgrid(X_plot, Y_plot)
        plt.figure()
        plt.contourf(X_plot,Y_plot,gamma[:self.nx-2,:self.nx-2])
        plt.title(title)
        plt.axis('equal')
        plt.colorbar()
        
    def linModErrors(self):
        ##### Errors in Linear vs Isofit models for reflectances #####

        n = 10
        relerror = np.zeros([self.nx-2,n])
        for i in range(n):
            y_isofit = self.Y[i,:]
            y_lasso = (self.phi_tilde.T @ self.scaleX[i,:].T) * np.sqrt(self.r.varY) + self.r.meanY
            relerror[:,i] = abs(y_isofit - y_lasso) / abs(y_isofit)

        plt.figure()
        plt.semilogy(self.wl, relerror, linewidth=0.5)
        plt.title('Error in radiance using certain reflectance samples')
        plt.xlabel('Wavelength')
        plt.ylabel('Relative Error')
        plt.grid()

    def reconstruct_gammay(self, dim, eigvalPCA, eigvecPCA, eigvalLIS, eigvecLIS):
        #dim = 30
        basisPCA = eigvecPCA[:,:dim].T
        basisLIS = eigvecLIS[:,:dim].T

        temp = basisPCA @ (self.truthrad - self.r.meanY)
        Y_PCA = basisPCA.T @ temp + self.r.meanY

        temp = basisLIS @ (self.truthrad - self.r.meanY)
        Y_LIS = basisLIS.T @ temp + self.r.meanY

        plt.figure()
        plt.plot(self.wl, self.truthrad,label='Truth')
        plt.plot(self.wl, Y_PCA,label='PCA')
        plt.plot(self.wl, Y_LIS,label='LIS')
        plt.title('Reconstruction using first '+str(dim)+ ' eigenvectors')
        plt.xlabel('Wavelength')
        plt.ylabel('Radiance')
        plt.legend()
        plt.grid()

        self.plotcontour(self.gamma_y, 'Covariance Gamma_y')

        L = np.linalg.cholesky(self.gamma_ygx)
        gamma_y_PCA = np.zeros([self.ny, self.ny])
        gamma_y_LIS = np.zeros([self.ny,self.ny])

        for i in range(dim):
            gamma_y_PCA = gamma_y_PCA + eigvalPCA[i] * np.outer(eigvecPCA[:,i], eigvecPCA[:,i].T)
            gamma_y_LIS = gamma_y_LIS + eigvalLIS[i] * np.outer(eigvecLIS[:,i], eigvecLIS[:,i].T)

        gamma_y_LIS = L @ gamma_y_LIS @ L.T

        self.plotcontour(gamma_y_PCA, 'Gamma_y - Reconstruction using '+str(dim)+' PCA eigenvectors')
        self.plotcontour(gamma_y_LIS, 'Gamma_y - Reconstruction using '+str(dim)+' LIC eigenvectors')

        # error norm difference of reconstructed gamma_y
        errorPCA = np.zeros(100)
        errorLIS = np.zeros(100)
        gamma_y_PCA = np.zeros([self.ny, self.ny])
        gamma_y_LIS = np.zeros([self.ny, self.ny])
        for i in range(100):
            print('Iteration:', i+1)
            gamma_y_PCA = gamma_y_PCA + eigvalPCA[i] * np.outer(eigvecPCA[:,i], eigvecPCA[:,i].T)
            gamma_y_LIS = gamma_y_LIS + eigvalLIS[i] * np.outer(eigvecLIS[:,i], eigvecLIS[:,i].T)
            
            errorPCA[i] = np.linalg.norm(gamma_y_PCA - self.gamma_y)/np.linalg.norm(self.gamma_y)
            errorLIS[i] = np.linalg.norm(L @ gamma_y_LIS @ L.T - self.gamma_y)/np.linalg.norm(self.gamma_y)

        # reconstruction error (relative) for 1 to 100 eigenvectors
        plt.figure()
        plt.semilogy(range(100), errorPCA, label='PCA')
        plt.semilogy(range(100), errorLIS, label='LIS')
        plt.title('Gamma_y reconstruction error comparison')
        plt.grid()
        plt.legend()
        


        
