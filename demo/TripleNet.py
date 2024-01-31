import sys
sys.path.append("C:\\Users\\mrodri02\\Documents\\Projet ARIA\\projet S1 12-01-2024")
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy.integrate import *
from aotools import *
from scipy.fft import *
from scipy.linalg import inv
import os
import psf_sim
import ExpData
from tqdm import tqdm
from functools import partial

def floor4(x, i=4):
    if i==1:
        return np.floor(x/2)
    else:
        return floor4(np.floor(x/2), i=i-1)

## Define TripleNET (En Pytorch)

class TripleNet(nn.Module):
    def set_exp_parameters(self, xmax=5000, zmax=10500, width=0.6, nb=50, nz=50, N=24, ON=1.25, n0=1.33, lamb=1028, alpha=1.8/3.8, r_pupil=3.8, noise='poisson', noise_rate=1037, norm_mode='max', freq_res=1, tilt=False, distrib='unif'):
        self.xmax=xmax #en nm
        self.zmax=zmax #en nm
        self.width=width #en rad
        self.nb=nb #sans unite
        self.nz=nz #sans unite
        self.N=N #sans unite
        self.ON=ON #sans unite
        self.n0=n0 #sans unite
        self.lamb=lamb #en nm
        self.alpha=alpha #sans unite
        self.r_pupil=r_pupil #en mm
        self.norm_mode=norm_mode
        self.noise=noise
        self.noise_rate=noise_rate # sans unite
        self.tilt=tilt # booleen
        self.distrib=distrib # 'unif' ou 'L2'


        self.freq_res=freq_res
        assert self.freq_res%2==0 or self.freq_res==1

        self.radius_k=self.ON/self.lamb #en nm¨{-1}
        self.sigma = self.radius_k*self.alpha
        self.R=self.nb/(2*self.xmax) # en nm^{-1}
        self.kx = np.linspace(-self.R,self.R,self.nb*freq_res) # en nm^{-1}
        self.ky = np.linspace(-self.R,self.R,self.nb*freq_res)
        self.z= np.linspace(-self.zmax/2,self.zmax/2, self.nz)

        self.zCoeffs=np.zeros(self.N+4, dtype=np.float32)
        self.tzCoeffs=torch.from_numpy(self.zCoeffs)
        self.I=np.zeros([self.nb,self.nb,self.nz], dtype=np.float32)
        self.tI=torch.from_numpy(self.I)

        n=int(floor4(self.nb))
        m=int(floor4(self.nz))
        self.fcXY=nn.Linear(128*n*n, 512)
        self.fcXZ=nn.Linear(128*n*m, 512)
        self.fcYZ=nn.Linear(128*n*m, 512)

        if self.tilt:
            self.fc5=nn.Linear(64,self.N+2)
        else:
            self.fc5=nn.Linear(64,self.N)

        assert self.noise in ['poisson', 'normal', None], "noise has to be either 'poisson', 'normal' or None."

        if self.noise=='poisson':
            assert self.noise_rate!=None, "noise_rate needs to be specified if noise != None."
            assert self.noise_rate>0, "noise_rate has to be > 0"

            self.trans=partial(psf_sim.poisson_noise, self.noise_rate)
        elif self.noise=='normal':
            assert self.noise_rate!=None, "noise_rate needs to be specified if noise != None."

            self.trans=partial(psf_sim.gaussian_noise, self.noise_rate)
        else:
            self.trans=None


    def sim_batch(self, batch_size):
        batchXY=torch.zeros([batch_size,1,self.nb, self.nb])
        batchXZ=torch.zeros([batch_size, 1, self.nb, self.nz])
        batchYZ=torch.zeros([batch_size, 1, self.nb, self.nz])
        batch=[batchXY, batchXZ, batchYZ]

        batch_coeffs=torch.zeros([batch_size, self.N])
        if self.tilt:
            batch_m=torch.normal(torch.zeros([batch_size,2]),self.radius_k/2*torch.ones([batch_size,2])) # tilt dans l'espace de fourier
        else:
            batch_m=torch.zeros([batch_size,2])

        for j in range(batch_size):

            if self.distrib=='unif':
                self.zCoeffs[4:] = np.random.rand(self.N)*self.width*2-self.width
            elif self.distrib=='L2':
                self.zCoeffs[4:] = psf_sim.Unif_boule(self.N)*self.width
            else:
                raise ValueError("distrib should be either 'unif' or 'L2'.")

            _,_,I_temp = psf_sim.I_exp(self.kx,self.ky, self.z,self.nb, self.nz, self.sigma, self.radius_k, self.zCoeffs, self.n0, self.lamb, m=batch_m[j,:].numpy())
            I_temp=I_temp[self.freq_res*self.nb//2-self.nb//2:self.freq_res*self.nb//2+self.nb//2, self.freq_res*self.nb//2-self.nb//2:self.freq_res*self.nb//2+self.nb//2, :]
            I_temp=I_temp/np.max(np.abs(I_temp))
            self.I[:,:,self.nz//2]=I_temp[:,:,self.nz//2]
            self.I[:,self.nb//2,:]=I_temp[:,self.nb//2,:]
            self.I[self.nb//2,:,:]=I_temp[self.nb//2,:,:]
            batch[0][j, 0,:,:]=self.trans(self.tI[:,:,self.nz//2])
            batch[1][j, 0,:,:]=self.trans(self.tI[:,self.nb//2,:])
            batch[2][j, 0,:,:]=self.trans(self.tI[self.nb//2,:,:])

            batch_coeffs[j]=self.tzCoeffs[4:]

        if self.tilt:
            return batch, batch_coeffs, batch_m*self.lamb*self.r_pupil/self.ON # conversion du tilt en mm dans l'espace reel
        else:
            return batch, batch_coeffs

    def sim_Dataset(self, ds_path, dataset_size, batch_size, test=False):
        """
        Compute a Dataset of simulated PSF with respect to the experimental parameters (to change them, call the set_exp_parameters() method).

        Inputs :
        ds_path : path of the directory in which the Dataset will be saved.
        dataset_size : number of samples to compute.
        batch_size : to gain memory, the samples are saved within mini-batches of size batch_size rather than individually. Each file "sample{i}.pt" contains a mini-batch. It doesn't need to match the batch size used for training.
        test : if True, compute a test Dataset in addition to the training one. Default is False.

        Outputs :

        trainset : dataset of simulated PSF and associated Zernike coefficients for training. You can access to the samples the same way you get list items. Samples are lists of 3 2D tensors (XY, XZ, YZ, in this order). XY size is nb x nb, and XZ, YZ shapes are nb x nz

        testset : same as trainset but used only for testing. Is returned only if test==True.

        Example :
        >>> TN=TripleNet()
        trainset=TN.sim_Dataset('path/to/dataset/', 1000, 100)
        sample, target = trainset[10]
        # sample[0] is XY plane
        # sample[1] is XZ plane
        # sample[2] is YZ plane
        # target is the tensor of Zernike coefficients used to compute sample.
        """
        return psf_sim.GenerateDataset(ds_path, dataset_size, batch_size, zmax=self.zmax, xmax=self.xmax, width=self.width, nb=self.nb, nz=self.nz, N=self.N, ON=self.ON, n0=self.n0, lamb=self.lamb, alpha=self.alpha, r_pupil=self.r_pupil, mode='TripleNet', test=test, norm_mode=self.norm_mode, noise=self.noise, noise_rate=self.noise_rate, freq_res=self.freq_res, tilt=self.tilt, distrib=self.distrib)

    def load_Dataset(self, ds_path):
        """
        Load a Dataset already saved on your computer.

        Inputs :

        ds_path : path of the directory in which the Dataset is saved.

        Outputs :

        trainset : dataset of simulated PSF and associated Zernike coefficients for training. You can access to the samples the same way you get list items. Samples are lists of 3 2D tensors (XY, XZ, YZ, in this order). XY size is nb x nb, and XZ, YZ shapes are nb x nz

        testset : same as trainset but used only for testing. Is returned only if there is actually data in 'Dataset/Test/' directory.

        Example :
        >>> TN=TripleNet()
        trainset=TN.load_Dataset('path/to/dataset/')
        sample, target = trainset[10]
        # sample[0] is XY plane
        # sample[1] is XZ plane
        # sample[2] is YZ plane
        # target is the tensor of Zernike coefficients used to compute sample.
        """

        list_train_samples=os.listdir(os.path.join(ds_path, 'Train','Samples'))
        list_train_coeffs=os.listdir(os.path.join(ds_path, 'Train', 'Coeffs'))
        list_test_samples=os.listdir(os.path.join(ds_path, 'Test','Samples'))
        list_test_coeffs=os.listdir(os.path.join(ds_path, 'Test', 'Coeffs'))

        if list_test_samples:

            if self.tilt:
                list_train_tilts=os.listdir(os.path.join(ds_path, 'Train', 'Tilts'))
                list_test_tilts=os.listdir(os.path.join(ds_path, 'Test','Tilts'))
                trainset=psf_sim.TiltTripleNetDataset(list_train_coeffs, list_train_samples, list_train_tilts, os.path.join(ds_path, "Train"), transform=self.trans)
                testset=psf_sim.TiltTripleNetDataset(list_test_coeffs, list_test_samples, list_test_tilts, os.path.join(ds_path, "Test"), transform=self.trans)
            else:
                trainset=psf_sim.TripleNetDataset(list_train_coeffs, list_train_samples, os.path.join(ds_path, "Train"), transform=self.trans)
                testset=psf_sim.TripleNetDataset(list_test_coeffs, list_test_samples, os.path.join(ds_path, "Test"), transform=self.trans)
            return trainset, testset
        else:
            if self.tilt:
                list_train_tilts=os.listdir(os.path.join(ds_path, 'Train', 'Tilts'))
                trainset=psf_sim.TiltTripleNetDataset(list_train_coeffs, list_train_samples, list_train_tilts, os.path.join(ds_path, "Train"), transform=self.trans)
            else:
                trainset=psf_sim.TripleNetDataset(list_train_coeffs, list_train_samples, os.path.join(ds_path, "Train"), transform=self.trans)
            return trainset


    def __init__(self, device='cpu', tilt=False):
        super(TripleNet, self).__init__()
        self.set_exp_parameters()
        n=int(floor4(self.nb))
        m=int(floor4(self.nz))
        self.device = device
        self.tilt=tilt

        self.convXY11=nn.Conv2d(1,8,5, padding='same')
        self.convXY12=nn.Conv2d(8,8,5, padding='same')
        self.convXY13=nn.Conv2d(8,8,5, padding='same')
        self.convXY21=nn.Conv2d(8,16,5, padding='same')
        self.convXY22=nn.Conv2d(16,16,5, padding='same')
        self.convXY23=nn.Conv2d(16,16,5, padding='same')
        self.convXY31=nn.Conv2d(16,32,5, padding='same')
        self.convXY32=nn.Conv2d(32,32,5, padding='same')
        self.convXY33=nn.Conv2d(32,32,5, padding='same')
        self.convXY41=nn.Conv2d(32,64,3, padding='same')
        self.convXY42=nn.Conv2d(64,64,3, padding='same')
        self.convXY43=nn.Conv2d(64,64,3, padding='same')
        self.convXY51=nn.Conv2d(64,128,3, padding='same')
        self.convXY52=nn.Conv2d(128,128,3, padding='same')
        self.convXY53=nn.Conv2d(128,128,3, padding='same')

        self.convXZ11=nn.Conv2d(1,8,5, padding='same')
        self.convXZ12=nn.Conv2d(8,8,5, padding='same')
        self.convXZ13=nn.Conv2d(8,8,5, padding='same')
        self.convXZ21=nn.Conv2d(8,16,5, padding='same')
        self.convXZ22=nn.Conv2d(16,16,5, padding='same')
        self.convXZ23=nn.Conv2d(16,16,5, padding='same')
        self.convXZ31=nn.Conv2d(16,32,5, padding='same')
        self.convXZ32=nn.Conv2d(32,32,5, padding='same')
        self.convXZ33=nn.Conv2d(32,32,5, padding='same')
        self.convXZ41=nn.Conv2d(32,64,3, padding='same')
        self.convXZ42=nn.Conv2d(64,64,3, padding='same')
        self.convXZ43=nn.Conv2d(64,64,3, padding='same')
        self.convXZ51=nn.Conv2d(64,128,3, padding='same')
        self.convXZ52=nn.Conv2d(128,128,3, padding='same')
        self.convXZ53=nn.Conv2d(128,128,3, padding='same')

        self.convYZ11=nn.Conv2d(1,8,5, padding='same')
        self.convYZ12=nn.Conv2d(8,8,5, padding='same')
        self.convYZ13=nn.Conv2d(8,8,5, padding='same')
        self.convYZ21=nn.Conv2d(8,16,5, padding='same')
        self.convYZ22=nn.Conv2d(16,16,5, padding='same')
        self.convYZ23=nn.Conv2d(16,16,5, padding='same')
        self.convYZ31=nn.Conv2d(16,32,5, padding='same')
        self.convYZ32=nn.Conv2d(32,32,5, padding='same')
        self.convYZ33=nn.Conv2d(32,32,5, padding='same')
        self.convYZ41=nn.Conv2d(32,64,3, padding='same')
        self.convYZ42=nn.Conv2d(64,64,3, padding='same')
        self.convYZ43=nn.Conv2d(64,64,3, padding='same')
        self.convYZ51=nn.Conv2d(64,128,3, padding='same')
        self.convYZ52=nn.Conv2d(128,128,3, padding='same')
        self.convYZ53=nn.Conv2d(128,128,3, padding='same')

        self.pool=nn.MaxPool2d((2,2))
        self.fcXY=nn.Linear(128*n*n, 512)
        self.fcXZ=nn.Linear(128*n*m, 512)
        self.fcYZ=nn.Linear(128*n*m, 512)

        self.fc1=nn.Linear(512*3,512)
        self.fc2=nn.Linear(512, 512)
        self.fc3=nn.Linear(512,256)
        self.fc4=nn.Linear(256,64)

        if self.tilt:
            self.fc5=nn.Linear(64,self.N+2)
        else:
            self.fc5=nn.Linear(64,self.N)
    def forward(self, x):
        xy=x[0]
        xz=x[1]#*self.maskXZ
        yz=x[2]#*self.maskYZ
        activation=torch.tanh

        xy=activation(self.convXY11(xy))
        xy=activation(self.convXY12(xy))
        xy=activation(self.convXY13(xy))
        xy=self.pool(xy)
        xy=activation(self.convXY21(xy))
        xy=activation(self.convXY22(xy))
        xy=activation(self.convXY23(xy))
        xy=self.pool(xy)
        xy=activation(self.convXY31(xy))
        xy=activation(self.convXY32(xy))
        xy=activation(self.convXY33(xy))
        xy=self.pool(xy)
        xy=activation(self.convXY41(xy))
        xy=activation(self.convXY42(xy))
        xy=activation(self.convXY43(xy))
        xy=self.pool(xy)
        xy=activation(self.convXY51(xy))
        xy=activation(self.convXY52(xy))
        xy=activation(self.convXY53(xy))

        xz=activation(self.convXZ11(xz))
        xz=activation(self.convXZ12(xz))
        xz=activation(self.convXZ13(xz))
        xz=self.pool(xz)
        xz=activation(self.convXZ21(xz))
        xz=activation(self.convXZ22(xz))
        xz=activation(self.convXZ23(xz))
        xz=self.pool(xz)
        xz=activation(self.convXZ31(xz))
        xz=activation(self.convXZ32(xz))
        xz=activation(self.convXZ33(xz))
        xz=self.pool(xz)
        xz=activation(self.convXZ41(xz))
        xz=activation(self.convXZ42(xz))
        xz=activation(self.convXZ43(xz))
        xz=self.pool(xz)
        xz=activation(self.convXZ51(xz))
        xz=activation(self.convXZ52(xz))
        xz=activation(self.convXZ53(xz))

        yz=activation(self.convYZ11(yz))
        yz=activation(self.convYZ12(yz))
        yz=activation(self.convYZ13(yz))
        yz=self.pool(yz)
        yz=activation(self.convYZ21(yz))
        yz=activation(self.convYZ22(yz))
        yz=activation(self.convYZ23(yz))
        yz=self.pool(yz)
        yz=activation(self.convYZ31(yz))
        yz=activation(self.convYZ32(yz))
        yz=activation(self.convYZ33(yz))
        yz=self.pool(yz)
        yz=activation(self.convYZ41(yz))
        yz=activation(self.convYZ42(yz))
        yz=activation(self.convYZ43(yz))
        yz=self.pool(yz)
        yz=activation(self.convYZ51(yz))
        yz=activation(self.convYZ52(yz))
        yz=activation(self.convYZ53(yz))

        xy=torch.flatten(xy, 1)
        xz=torch.flatten(xz, 1)
        yz=torch.flatten(yz, 1)

        xy=activation(self.fcXY(xy))
        xz=activation(self.fcXZ(xz))
        yz=activation(self.fcYZ(yz))

        batch,nxy=xy.shape
        _,nxz=xz.shape
        _,nyz=yz.shape

        # X=torch.zeros([batch,nxy+nxz+nyz])
        # X[:,:nxy]=xy
        # X[:,nxy:nxy+nxz]=xz
        # X[:,nxy+nxz:]=yz

        X=torch.cat((xy, xz, yz), dim=1)

        X=torch.tanh(self.fc1(X))
        X=torch.tanh(self.fc2(X))
        X=torch.tanh(self.fc3(X))
        X=torch.tanh(self.fc4(X))
        X=self.fc5(X)

        if self.tilt:
            return X[:,:2], X[:,2:]
        else:
            return X

    def train_on_simulation(self, batch_size, nb_it, epochs=1, lr=1e-4, mu=1/4, return_err=True, save=False, save_path=None):
        optimizer=optim.Adam(self.parameters(), lr=lr)
        criterion=nn.MSELoss()
        #criterion=nn.L1Loss()
        print('Training...')
        list_err=[]
        k=10
        for epoch in range(epochs):
            tepoch=tqdm(range(nb_it), unit='batch', desc=f"Epoch {epoch+1}/{epochs}", mininterval=1)
            running_loss=0.
            global_error=0.
            i=0
            list_err.append([])
            for i in tepoch:

                if self.tilt:
                    inputs, targets, m_targets=self.sim_batch(batch_size)
                    m_targets=m_targets.to(self.device)
                else:
                    inputs, targets=self.sim_batch(batch_size)

                targets=targets.to(self.device)
                inputs[0]=inputs[0].to(self.device)
                inputs[1]=inputs[1].to(self.device)
                inputs[2]=inputs[2].to(self.device)
                optimizer.zero_grad()

                outputs=self.forward(inputs)
                if self.tilt:
                    loss=criterion(outputs[1], targets)+mu*criterion(outputs[0], m_targets)
                else:
                    loss=criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % k == k-1:    # print every 100 mini-batches
                    global_error=running_loss/k
                    running_loss = 0.
                    list_err[-1].append(np.sqrt(global_error))
                if i>=k:
                    tepoch.set_postfix(loss=np.sqrt(global_error), refresh=False)
                i+=1
            tepoch.close()

            if save:
                torch.save(self.state_dict(),os.path.join(save_path,f'state_at_epoch_{epoch}.pt'))
        print('Finished Training')

        if return_err==True:
            return list_err

    def train_on_DS(self, trainloader, epochs=1, lr=1e-4, mu=1/4, return_err=True, save=False, save_path=None):
        optimizer=optim.Adam(self.parameters(), lr=lr)
        criterion=nn.MSELoss()
        #criterion=nn.L1Loss()
        print('Training...')
        list_err=[]
        k=10 # to print every k mini-batches
        for epoch in range(epochs):
            tepoch=tqdm(trainloader, unit='batch', desc=f"Epoch {epoch+1}/{epochs}", mininterval=1)
            running_loss=0.
            global_error=0.
            i=0
            list_err.append([])
            for inputs, targets in tepoch:

                if self.tilt:
                    coeffs_targets, m_targets = targets
                    coeffs_targets=coeffs_targets.to(device)
                    m_targets=m_targets.to(self.device)
                else:
                    targets=targets.to(self.device)

                inputs[0]=inputs[0].to(self.device)
                inputs[1]=inputs[1].to(self.device)
                inputs[2]=inputs[2].to(self.device)


                optimizer.zero_grad()

                outputs=self.forward(inputs)
                if self.tilt:
                    loss=criterion(outputs[1], coeffs_targets)+mu*criterion(outputs[0], m_targets)
                else:
                    loss=criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % k == k-1:    # print every k mini-batches
                    global_error=running_loss/k
                    running_loss = 0.
                    list_err[-1].append(np.sqrt(global_error))
                if i>=k:
                    tepoch.set_postfix(loss=np.sqrt(global_error), refresh=False)
                i+=1
            tepoch.close()

            if save:
                torch.save(self.state_dict(),os.path.join(save_path,f'state_at_epoch_{epoch}.pt'))
        print('Finished Training')

        if return_err==True:
            return list_err