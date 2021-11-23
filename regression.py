import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import scale, StandardScaler
from sklearn.neighbors import KernelDensity

class Regression:
    '''
    Contains functions to perform Lasso Regression to 
    create a linear approximation of the Isofit forward model.
    '''

    def __init__(self, setup):
        
        # directory to store regression results
        self.regDir = setup.regDir

        # setup parameters
        # self.plotbands = setup.plotbands
        self.wavelengths = setup.wavelengths
        self.reflectance = setup.reflectance
        self.truth = setup.truth
        self.radiance = setup.radianceSim
        self.bands = setup.bands
        self.bandsX = setup.bandsX

        # load data sets
        self.sampleDir = setup.sampleDir
        X_train = np.load(self.sampleDir + 'X_train.npy')
        Y_train = np.load(self.sampleDir + 'Y_train.npy')
        X_test = np.load(self.sampleDir + 'X_test.npy')
        Y_test = np.load(self.sampleDir + 'Y_test.npy')
        
        # plt.figure(1) 
        # for i in range(100):   
        #     plt.plot(Y_train[np.random.randint(0,24000),:432])
        # # plt.figure(2)
        # # for i in range(1000):
        # #     ii = np.random.randint(0,24000)
        # #     jj = np.random.randint(0,24000)
        # #     plt.plot(X_train[ii,432], X_train[jj,433],'b.')
        # # plt.title('Atmospheric Parameters')
        # plt.show()

        # scale the data
        self.scalerX = StandardScaler().fit(X_train)
        self.scalerY = StandardScaler().fit(Y_train)
        self.X_train = self.scalerX.transform(X_train)
        self.Y_train = self.scalerY.transform(Y_train)
        self.X_test = self.scalerX.transform(X_test)
        self.Y_test = self.scalerY.transform(Y_test)

        # get mean and variance for future transformations
        self.meanX = self.scalerX.mean_
        self.varX = self.scalerX.var_
        self.meanY = self.scalerY.mean_
        self.varY = self.scalerY.var_

        # scale truth as well
        self.N = self.X_train.shape[0]
        # ny = self.radiance.size
        self.nx = setup.nx
        self.ny = setup.ny
        self.reflectance_scaled = (self.reflectance - self.meanX[:self.nx-2]) / np.sqrt(self.varX[:self.nx-2])
        self.truth_scaled = (self.truth - self.meanX) / np.sqrt(self.varX)
        
    def reglasso(self, param, yElem):

        # train with the reduced X_train
        linreg = Lasso(alpha=param, max_iter=5000) 
        linreg.fit(self.X_train, self.Y_train[:,yElem])
        phiReduce = linreg.coef_
        pred = linreg.predict(self.X_train)
        trainError = mean_squared_error(self.Y_train[:,yElem], pred)

        # make phi back into 427
        # nx = self.X_train.shape[1]
        # phi = np.zeros(nx)
        # j = 0
        # for i in range(nx):
        #     if i in self.bandsX:
        #         phi[i] = phiReduce[j] 
        #         j = j + 1
        phi = phiReduce
        
        N = self.X_test.shape[0]
        genError = 0
        for i in range(N):
            genError = genError + 1/N * np.linalg.norm(self.Y_test[i,yElem] - phi.dot(self.X_test[i,:]))
        return phi, trainError, genError

    def fullLasso(self, params):
        # perform lasso for all wavelengths

        nx = self.X_train.shape[1]
        ny = self.Y_train.shape[1]

        y_lasso = np.zeros(ny)
        GE = np.zeros(ny)
        TE = np.zeros(ny)
        phi = np.zeros([ny,nx])
        
        for i in range(ny):
            print('Regression: Element', i+1, str(self.wavelengths[i]) + 'nm')
            phi[i,:], te, ge = self.reglasso(params[i], i)
            scaled = phi[i,:].dot(self.truth_scaled) 
            y_lasso[i] = scaled * np.sqrt(self.varY[i]) + self.meanY[i]
            TE[i] = te
            GE[i] = ge
        
        np.save(self.regDir + 'y_lasso.npy', y_lasso)
        np.save(self.regDir + 'lassoTE.npy', TE)
        np.save(self.regDir + 'lassoGE.npy', GE)
        np.save(self.regDir + 'phi.npy',phi)      

    def plotFullLasso(self):

        y_lasso = np.load(self.regDir + 'y_lasso.npy')
        TE = np.load(self.regDir + 'lassoTE.npy')
        GE = np.load(self.regDir + 'lassoGE.npy')
        phi = np.load(self.regDir + 'phi.npy')

        plt.figure(31)
        plt.plot(self.wavelengths, self.radiance, 'navy',linewidth=2, label='Isofit Forward Model')
        plt.plot(self.wavelengths, y_lasso, 'orange',linewidth=1, label='Linear Model')
        plt.xlabel('Wavelength')
        plt.ylabel('Radiance')
        plt.title('Lasso Regression - Radiance')
        plt.grid()
        plt.legend()

        plt.figure(34)
        # self.plotbands(GE, 'navy', linewidth=1, label='Generalization Error', axis='semilogy')
        # self.plotbands(TE, 'orange', linewidth=1, label='Training Error', axis='semilogy')
        plt.semilogy(GE[self.bands], 'navy', linewidth=1, label='Generalization Error')
        plt.semilogy(TE[self.bands], 'orange', linewidth=1, label='Training Error')
        plt.xlabel('Wavelength')
        plt.ylabel('Error')
        plt.title('Lasso Regression - Error')
        plt.grid()
        plt.legend()

        relerror = abs(self.radiance - y_lasso) / abs(self.radiance)
        plt.figure(35)
        # self.plotbands(relerror, 'navy',linewidth=1, label='Lasso,p=1e-3',axis='semilogy')
        plt.semilogy(relerror[self.bands], 'navy',linewidth=1, label='Lasso,p=1e-3')
        plt.xlabel('Wavelength')
        plt.ylabel('Error')
        plt.title('Lasso Regression - Relative Error')
        plt.grid()
        plt.legend()

        plt.figure(36)
        plt.spy(phi, color='b', precision=1e-15,markersize=2)
        plt.title('Sparsity Plot - G')


    def plotlog(self, plotx, ploty, fig, title='', xLabel='', yLabel=''):
        plt.figure(fig)
        plt.loglog(plotx, ploty, linewidth=2)
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.legend()
        plt.grid()
        return

    def tuneLasso(self, params, yElem, plot=True):
        nParam = np.array(params).size
        trainError = np.zeros(nParam)
        genError = np.zeros(nParam)

        for j in range(nParam):
            print('Regression parameter:', params[j])
            [phi, te, ge] = self.reglasso(params[j], yElem)
            trainError[j] = te
            genError[j] = ge
            print('\t Gen Error:', ge, '\t Train Error:', te)
        if plot==True:
            self.plotlog(params,trainError, 11, title='Lasso - Training Error', xLabel='Regularization Parameter', yLabel='Error')
            self.plotlog(params,genError,12, title='Lasso - Generalization Error', xLabel='Regularization Parameter', yLabel='Error')

        indMin = np.argmin(genError)
        return trainError, genError, params[indMin]

    def plotRegSample(self, fig, yElem, phi):

        # plot the training + test data, plus the line
        plt.figure(fig)
        plt.plot(self.X_train[:,yElem], self.Y_train[:,yElem],'bx',label='Training Data')
        plt.plot(self.X_test[:,yElem], self.Y_test[:,yElem],'rx', label='Test Data')
        linearApprox = np.zeros(10000)
        for i in range(10000):
            linearApprox[i] = phi.dot(self.X_train[i,:])

        polynomial = np.polyfit(self.X_train[:,yElem], linearApprox, 1)
        xLine = np.linspace(-5,5,50)
        plt.plot(xLine, np.poly1d(polynomial)(xLine))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Lasso Regression at '+str(round(self.wavelengths[yElem]))+'nm, alpha=1e-3')
        plt.grid()
        plt.legend()
        plt.xlim([1,2])
        plt.ylim([1,2])
    
    def distFromTruth(self):
        # plots a distribution of distance of samples from truth

        X_train = np.diag(np.sqrt(self.varX)) @ self.X_train.T + np.outer(self.meanX, np.ones(self.N))
        X_train = X_train.T
        dist = np.zeros(self.N)[:, np.newaxis]
        for i in range(self.N):
            dist[i] = np.linalg.norm(self.truth - X_train[i,:])

        X_plot = np.linspace(0,10,1000)[:, np.newaxis]
        kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(dist)
        log_dens = kde.score_samples(X_plot)
        plt.figure()
        plt.fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
        plt.title('Distance of Training data from truth - Gaussian Kernel')
        plt.xlabel('Euclidean Distance')

        kde = KernelDensity(kernel='tophat', bandwidth=0.75).fit(dist)
        log_dens = kde.score_samples(X_plot)
        plt.figure()
        plt.fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
        plt.title('Distance of Training data from truth - Tophat Kernel')
        plt.xlabel('Euclidean Distance')
        
        
