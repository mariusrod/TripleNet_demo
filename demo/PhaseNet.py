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
from tqdm import tqdm
from functools import partial


def floor5(x, i=5):
    if i==1:
        return np.floor(x/2)
    else:
        return floor5(np.floor(x/2), i=i-1)

## Define PHASENET (En Pytorch)

class PhaseNet(nn.Module):
    def set_exp_parameters(self, xmax=2500, zmax=3500, width=0.4, nb=50, nz=50, N=22, ON=1.25, n0=1.33, lamb=1028, alpha=1.8/3.8, noise='poisson', noise_rate=1037, norm_mode='max', distrib='unif'):

        """
        Do not use this method after training the network (a network trained with some parameters may give wrong answers according to other parameters.
        """

        self.xmax=xmax
        self.zmax=zmax
        self.width=width
        self.nb=nb
        self.nz=nz
        self.N=N
        self.ON=ON
        self.n0=n0
        self.lamb=lamb
        self.alpha=alpha
        self.norm_mode=norm_mode
        self.noise=noise
        self.noise_rate=noise_rate
        self.distrib=distrib

        self.radius_k=self.ON/self.lamb
        self.sigma = self.radius_k*self.alpha
        self.R=self.nb/(2*self.xmax) # en nm^{-1}
        self.kx = np.linspace(-self.R,self.R,self.nb) # en nm^{-1}
        self.ky = np.linspace(-self.R,self.R,self.nb)
        self.z= np.linspace(-self.zmax/2,self.zmax/2, self.nz)

        self.zCoeffs=np.zeros(self.N+4, dtype=np.float32)
        self.tzCoeffs=torch.from_numpy(self.zCoeffs)
        self.I=np.zeros([self.nb,self.nb,self.nz], dtype=np.float32)
        self.tI=torch.from_numpy(self.I)

        # Adapt the following two layers :
        n=int(floor5(self.nb))
        self.fc2=nn.Linear(64*self.nz*n*n,64)
        self.fc3=nn.Linear(64,self.N)

    def sim_batch(self, batch_size):
        batch=torch.zeros([batch_size,1,self.nb, self.nb, self.nz])
        batch_coeffs=torch.zeros([batch_size, self.N])
        for j in range(batch_size):
            if self.distrib=='unif':
                self.zCoeffs[4:] = np.random.rand(self.N)*self.width*2-self.width
            elif self.distrib=='L2':
                self.zCoeffs[4:] = psf_sim.Unif_boule(self.N)*self.width
            _,_,I_temp = psf_sim.I_exp(self.kx,self.ky, self.z,self.nb, self.nz, self.sigma, self.radius_k, self.zCoeffs, self.n0, self.lamb)
            self.I[:,:,:]=I_temp/np.max(np.abs(I_temp))
            batch[j, :,:,:]=self.tI
            batch_coeffs[j]=self.tzCoeffs[4:]
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

        trainset : dataset of simulated PSF and associated Zernike coefficients for training. You can access to the samples the same way you get list items. Samples are 3D tensors of shape nb*nb*nz.

        testset : same as trainset but used only for testing. Is returned only if test==True.

        Example :
        >>> TN=TripleNet()
        trainset=TN.sim_Dataset('path/to/dataset/', 1000, 100)
        sample, target = trainset[10]
        # sample is the 3D PSF
        # target is the tensor of Zernike coefficients used to compute sample.
        """
        return psf_sim.GenerateDataset(ds_path, dataset_size, batch_size, zmax=self.zmax, xmax=self.xmax, width=self.width, nb=self.nb, nz=self.nz, N=self.N, ON=self.ON, n0=self.n0, lamb=self.lamb, alpha=self.alpha, mode='PhaseNet', test=test, norm_mode=self.norm_mode, noise=self.noise, noise_rate=self.noise_rate, distrib=self.distrib)

    def load_Dataset(self, ds_path):
        """
        Load a Dataset already saved on your computer.

        Inputs :

        ds_path : path of the directory in which the Dataset is saved.

        Outputs :

        trainset : dataset of simulated PSF and associated Zernike coefficients for training. You can access to the samples the same way you get list items. Samples are 3D tensors of shape nb*nb*nz.

        testset : same as trainset but used only for testing. Is returned only if there is actually data in 'Dataset/Test/' directory.

        Example :
        >>> TN=TripleNet()
        trainset=TN.load_Dataset('path/to/dataset/')
        sample, target = trainset[10]
        # sample is the 3D PSF
        # target is the tensor of Zernike coefficients used to compute sample.
        """

        assert self.noise in ['poisson', 'normal', None], "noise has to be either 'poisson', 'normal' or None."

        if self.noise=='poisson':
            assert self.noise_rate!=None, "noise_rate needs to be specified if noise != None."
            assert self.noise_rate>0, "noise_rate has to be > 0"

            trans=partial(psf_sim.poisson_noise, self.noise_rate)
        elif self.noise=='normal':
            assert self.noise_rate!=None, "noise_rate needs to be specified if noise != None."

            trans=partial(psf_sim.gaussian_noise, self.noise_rate)
        else:
            trans=None

        list_train_samples=os.listdir(os.path.join(ds_path, 'Train','Samples'))
        list_train_coeffs=os.listdir(os.path.join(ds_path, 'Train', 'Coeffs'))
        list_test_samples=os.listdir(os.path.join(ds_path, 'Test','Samples'))
        list_test_coeffs=os.listdir(os.path.join(ds_path, 'Test', 'Coeffs'))
        trainset=psf_sim.PhaseNetDataset(list_train_coeffs, list_train_samples, os.path.join(ds_path, "Train"), transform=trans)
        if list_test_samples:
            testset=psf_sim.PhaseNetDataset(list_test_coeffs, list_test_samples, os.path.join(ds_path, "Test"), transform=trans)
            return trainset, testset
        else:
            return trainset

    def __init__(self, device='cpu'):
        super(PhaseNet, self).__init__()
        self.set_exp_parameters()
        n=int(floor5(self.nb))
        #m=int(floor5(self.nz))
        self.device = device

        self.conv11=nn.Conv3d(1,8,3, padding='same')
        self.conv12=nn.Conv3d(8,8,3, padding='same')
        self.conv21=nn.Conv3d(8,16,3, padding='same')
        self.conv22=nn.Conv3d(16,16,3, padding='same')
        self.conv31=nn.Conv3d(16,32,3, padding='same')
        self.conv32=nn.Conv3d(32,32,3, padding='same')
        self.conv41=nn.Conv3d(32,64,3, padding='same')
        self.conv42=nn.Conv3d(64,64,3, padding='same')
        self.conv51=nn.Conv3d(64,128,3, padding='same')
        self.conv52=nn.Conv3d(128,128,3, padding='same')
        self.pool=nn.MaxPool3d((2,2,1))
        self.fc1=nn.Linear(128,64)
        #self.fc2=nn.Linear(64,64)
        self.fc2=nn.Linear(64*self.nz*n*n,64)
        self.fc3=nn.Linear(64,self.N)
    def forward(self, x):
        activation=torch.tanh

        x=activation(self.conv11(x))
        x=activation(self.conv12(x))
        x=self.pool(x)

        x=activation(self.conv21(x))
        x=activation(self.conv22(x))
        x=self.pool(x)

        x=activation(self.conv31(x))
        x=torch.tanh(self.conv32(x))
        x=self.pool(x)

        x=activation(self.conv41(x))
        x=activation(self.conv42(x))
        x=self.pool(x)

        x=activation(self.conv51(x))
        x=activation(self.conv52(x))
        x=self.pool(x)

        x=torch.transpose(x,1,2)
        x=torch.transpose(x,2,3)
        x=torch.transpose(x,3,4)
        x=activation(self.fc1(x))

        x=torch.flatten(x,1)
        x=activation(self.fc2(x))
        #x=torch.tanh(self.fc3(x))
        x=self.fc3(x)
        return x

    def train_on_simulation(self, batch_size, nb_it, epochs=1, lr=1e-4, return_err=True, save=False, save_path=None):
        optimizer=optim.Adam(self.parameters(), lr=lr)
        #criterion=nn.L1Loss()
        criterion=nn.MSELoss()
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

                inputs, targets=self.sim_batch(batch_size)

                inputs=inputs.to(self.device)
                targets=targets.to(self.device)

                optimizer.zero_grad()

                outputs=self.forward(inputs)
                loss=criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % k == k-1:    # print every 100 mini-batches
                    global_error=running_loss/k
                    running_loss = 0.
                    list_err[-1].append(np.sqrt(global_error))
                if i>=k-1:
                    tepoch.set_postfix(loss=np.sqrt(global_error), refresh=False)
                i+=1
            tepoch.close()
            if save:
                torch.save(self.state_dict(),os.path.join(save_path,f'state_at_epoch_{epoch}.pt'))

        print('Finished Training')

        if return_err==True:
            return list_err

    def train_on_DS(self, trainloader,epochs=1, lr=1e-4, return_err=True, save=False):
        optimizer=optim.Adam(self.parameters(), lr=lr)
        #criterion=nn.L1Loss()
        criterion=nn.MSELoss()
        print('Training...')
        list_err=[]
        k=10
        for epoch in range(epochs):
            tepoch=tqdm(trainloader, unit='batch', desc=f"Epoch {epoch+1}/{epochs}", mininterval=1)
            running_loss=0.
            global_error=0.
            i=0
            list_err.append([])
            for inputs, targets in tepoch:

                inputs=inputs.to(self.device)
                targets=targets.to(self.device)

                optimizer.zero_grad()

                outputs=self.forward(inputs)
                loss=criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % k == k-1:    # print every 100 mini-batches
                    global_error=running_loss/k
                    running_loss = 0.
                    list_err[-1].append(np.sqrt(global_error))
                if i>=k-1:
                    tepoch.set_postfix(loss=np.sqrt(global_error), refresh=False)
                i+=1
            tepoch.close()

            if save:
                torch.save(self.state_dict(),os.path.join(save_path,f'state_at_epoch_{epoch}.pt'))
        print('Finished Training')

        if return_err==True:
            return list_err

