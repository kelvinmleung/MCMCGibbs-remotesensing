import numpy as np
import json
from scipy.io import loadmat
from fsplit.filesplit import Filesplit
from spectral import *
import matplotlib.pyplot as plt



class FileProcessing:


    def __init__(self, setupDir):
        self.setupDir = setupDir + '/'
        print('\n')

    def loadWavelength(self, wvFile): #'setup/data/wavelengths.txt'
        # these wavelengths must correspond to the reflectances (and not radiances)
        fileLoad = np.loadtxt(self.setupDir + wvFile).T
        if fileLoad.shape[0] == 2:
            wv, fwhm = fileLoad
        elif fileLoad.shape[0] == 3:
            ind, wv, fwhm = fileLoad
            wv = wv * 1000
        # mat = loadmat(wvFile)
        # self.wv = mat['wl'][0]
        self.wv = wv# * 1000

    def loadReflectance(self, refFile):
        data = np.loadtxt(self.setupDir + refFile).T
        wvRaw = data[0]
        refRaw = data[1]
        self.ref = np.interp(self.wv, wvRaw, refRaw)
        return self.ref

    def loadRadiance(self, datamatfile):
        mat = loadmat(self.setupDir + datamatfile)
        self.radiance = mat['meas'][0]

    def loadConfig(self, configFile):
        with open(self.setupDir + configFile, 'r') as f:
            self.config = json.load(f)
            
    def loadSurfModel(self, surfModel):
        mat = loadmat(self.setupDir + surfModel)
        means = mat['means']
        covs = mat['covs']
        return means, covs

    
    def getFiles(self):
        return self.wv, self.ref, self.radiance, self.config

    def splitFile(self, filename, output):
        fs = Filesplit()
        def split_cb(f, s):
            print("file: {0}, size: {1}".format(f, s))
        fs.split(file=filename, split_size=9000000000, output_dir=output, callback=split_cb)

    def mergeFile(self, inputdir):
        fs = Filesplit()
        def merge_cb(f, s):
            print("file: {0}, size: {1}".format(f, s))
        fs.merge(input_dir=inputdir, callback=merge_cb)

    def thinMCMCFile(self, inputdir, thinning):
        x_vals =  np.load(inputdir + 'MCMC_x.npy', mmap_mode='r')
        x_vals_thin = x_vals[:,::thinning]
        np.save(inputdir + 'MCMC_x_thin.npy', x_vals_thin)

    # def readENVI(self):
    #     img = open_image('../results/david_insitu/filtered_ocean.hdr')
    #     arr = img.load()
    #     print(arr.shape)
    #     plt.figure()
    #     for i in range(10):
    #         ref = np.squeeze(arr[:,0,i])
    #         plt.plot(ref)
    #     plt.show()



        


        