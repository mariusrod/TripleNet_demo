import numpy as np

import torch as torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy.integrate import *
from aotools import *
from scipy.fft import *
from scipy.linalg import inv
import os
from tqdm import tqdm
from functools import partial
import shutil as shutil

class PhaseNetDataset(Dataset):
    """
    Dataset of appropriate format for PhaseNet. It is used to put a DataSet structure on an existing dataset. The samples are 3D tensors representing of simulated PSFs, and the targets are the corresponding Zernike coefficients. The samples must be saved in a 'Samples' directory, and Zernike coefficients in a 'Coeffs' directory. The samples and coeffs file names are passed through the two lists sample_names and target_names.
    A PhaseNetDataset object is subscriptable, and can be passed to a PyTorch dataloader.

    Attributes :
    self.target_names : list of the Zernike coeffs file names
    self.sample_names : list of the samples file names
    self.dir : dataset directory's path
    self.transform : samples transform for data augmentation (ex : poisson noise, ...). Default is None.
    self.target_transform : coeffs transform for data augmentation. Default is None
    self.batch_size : batch size of the saved samples and coeffs.

    Methods :
    __init__ : called when initializing a PhaseNetDataset object.
    __len__ ; called when len() function is used.
    __getitem__: used to make PhaseNetDataset objects subscriptable. Returns
separately a sample and its corresponding coeffs.
    """

    def __init__(self, target_names, sample_names, dir, transform=None, target_transform=None):
        self.target_names=target_names
        self.sample_names=sample_names
        self.dir = dir
        self.transform = transform
        self.target_transform = target_transform

        test_sample = torch.load(os.path.join(self.dir, "Samples", self.sample_names[0]))
        self.batch_size=test_sample.shape[0]
    def __len__(self):
        return len(self.target_names)*self.batch_size # each sample file contains self.batch_size samples.

    def __getitem__(self, idx):
        path_sample = os.path.join(self.dir, "Samples" ,self.sample_names[idx//self.batch_size]) # The samples has to be saved in a 'Samples' directory !
        sample_batch = torch.load(path_sample)
        sample = sample_batch[idx % self.batch_size, :, :]
        l,m,n=sample.shape
        sample=torch.reshape(sample, (1,l,m,n))

        path_zCoeffs = os.path.join(self.dir, "Coeffs" ,self.target_names[idx//self.batch_size]) # The samples has to be saved in a 'Coeffs' directory !
        zCoeffs_batch = torch.load(path_zCoeffs)
        zCoeffs = zCoeffs_batch[idx % self.batch_size, :]


        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            zCoeffs = self.target_transform(zCoeffs)
        return sample, zCoeffs

class TripleNetDataset(Dataset):
    """
    Dataset of appropriate format for TripleNet (with tilt==False). It is used to put a DataSet structure on an existing dataset. The samples are 3D tensors representing of simulated PSFs, and the targets are the corresponding Zernike coefficients. The samples must be saved in a 'Samples' directory, and Zernike coefficients in a 'Coeffs' directory. The samples and coeffs file names are passed through the two lists sample_names and target_names.
    A TripleNetDataset object is subscriptable, and can be passed to a PyTorch dataloader.

    Attributes :
    self.target_names : list of the Zernike coeffs file names
    self.sample_names : list of the samples file names
    self.dir : dataset directory's path
    self.transform : samples transform for data augmentation (ex : poisson noise, ...). Default is None.
    self.target_transform : coeffs transform for data augmentation. Default is None
    self.batch_size : batch size of the saved samples and coeffs.

    Methods :
    __init__ : called when initializing a TripleNetDataset object.
    __len__ ; called when len() function is used.
    __getitem__: used to make TripleNetDataset objects subscriptable. Returns
separately a sample and its corresponding coeffs.
    """
    def __init__(self, target_names, sample_names, dir, transform=None, target_transform=None):
        self.target_names=target_names
        self.sample_names=sample_names
        self.dir = dir
        self.transform = transform
        self.target_transform = target_transform

        test_sample, _, _= torch.load(os.path.join(self.dir, "Samples", self.sample_names[0]))
        self.batch_size=test_sample.shape[0]
    def __len__(self):
        return len(self.target_names)*self.batch_size

    def __getitem__(self, idx):
        path_sample = os.path.join(self.dir, "Samples" ,self.sample_names[idx//self.batch_size]) # each sample file contains self.batch_size samples.
        batchXY, batchXZ, batchYZ=torch.load(path_sample)

        sampleXY=batchXY[idx % self.batch_size, :,:]
        sampleXZ=batchXZ[idx % self.batch_size, :,:]
        sampleYZ=batchYZ[idx % self.batch_size, :,:]

        if self.transform:
            sampleXY=self.transform(sampleXY)
            sampleXZ=self.transform(sampleXZ)
            sampleYZ=self.transform(sampleYZ)

        sampleXY=sampleXY[None, :, :]
        sampleXZ=sampleXZ[None, :, :]
        sampleYZ=sampleYZ[None, :, :]

        path_zCoeffs = os.path.join(self.dir, "Coeffs" ,self.target_names[idx//self.batch_size]) # The samples has to be saved in a 'Coeffs' directory !
        batch_zCoeffs = torch.load(path_zCoeffs)

        zCoeffs=batch_zCoeffs[idx % self.batch_size, :]

        if self.target_transform:
            zCoeffs = self.target_transform(zCoeffs)

        sample = [sampleXY, sampleXZ, sampleYZ]

        return sample, zCoeffs

class TiltTripleNetDataset(Dataset):
    """
    Dataset of appropriate format for TripleNet (with tilt==True). It is used to put a DataSet structure on an existing dataset. The samples are 3D tensors representing of simulated PSFs, and the targets are the corresponding Zernike coefficients and tilts. The samples must be saved in a 'Samples' directory, Zernike coefficients in a 'Coeffs' directory and tilts in a 'Tilts' directory. The samples, coeffs and tilts file names are passed through the two lists sample_names, target_names and tilt_names.
    A TiltTripleNetDataset object is subscriptable, and can be passed to a PyTorch dataloader.

    Attributes :
    self.target_names : list of the Zernike coeffs file names
    self.sample_names : list of the samples file names
    self.tilt_names : list of the tilts file names
    self.dir : dataset directory's path
    self.transform : samples transform for data augmentation (ex : poisson noise, ...). Default is None.
    self.target_transform : coeffs transform for data augmentation. Default is None
    self.tilt_transform : tilts transform for data augmentation. Default is None
    self.batch_size : batch size of the saved samples and coeffs.

    Methods :
    __init__ : called when initializing a TiltTripleNetDataset object.
    __len__ ; called when len() function is used.
    __getitem__: used to make TiltTripleNetDataset objects subscriptable. Returns
separately a sample and its corresponding coeffs.
    """
    def __init__(self, target_coeffs_names, sample_names, target_tilt_names, dir, transform=None, coeff_transform=None, tilt_transform=None):
        self.target_names=target_coeffs_names
        self.sample_names=sample_names
        self.tilt_names=target_tilt_names
        self.dir = dir
        self.transform = transform
        self.coeff_transform = coeff_transform
        self.tilt_transform=tilt_transform

        test_sample, _, _= torch.load(os.path.join(self.dir, "Samples", self.sample_names[0]))
        self.batch_size=test_sample.shape[0]
    def __len__(self):
        return len(self.target_names)*self.batch_size

    def __getitem__(self, idx):
        path_sample = os.path.join(self.dir, "Samples" ,self.sample_names[idx//self.batch_size])
        batchXY, batchXZ, batchYZ=torch.load(path_sample)

        sampleXY=batchXY[idx % self.batch_size, :,:]
        sampleXZ=batchXZ[idx % self.batch_size, :,:]
        sampleYZ=batchYZ[idx % self.batch_size, :,:]

        if self.transform:
            sampleXY=self.transform(sampleXY)
            sampleXZ=self.transform(sampleXZ)
            sampleYZ=self.transform(sampleYZ)

        sampleXY=sampleXY[None, :, :]
        sampleXZ=sampleXZ[None, :, :]
        sampleYZ=sampleYZ[None, :, :]

        path_zCoeffs = os.path.join(self.dir, "Coeffs" ,self.target_names[idx//self.batch_size])
        batch_zCoeffs = torch.load(path_zCoeffs)
        zCoeffs=batch_zCoeffs[idx % self.batch_size, :]

        if self.coeff_transform:
            zCoeffs = self.coeff_transform(zCoeffs)

        path_tilts = os.path.join(self.dir, "Tilts" ,self.tilt_names[idx//self.batch_size])
        batch_tilts = torch.load(path_tilts)
        tilt=batch_tilts[idx % self.batch_size, :]

        if self.tilt_transform:
            tilt = self.tilt_transform(tilt)


        sample = [sampleXY, sampleXZ, sampleYZ]
        target = [tilt, zCoeffs]

        return sample, target

def E_theorique_incident(x,y,sigma, m=np.array([0,0])):
    """
    Gaussian theoretical incident beam.
    """
    return np.exp(-((x-m[0])**2+(y-m[1])**2)/(2*sigma**2))

def E_pupil(kx,ky, z, nb, nz, sigma, radius_k, zCoeffs, n0, lamb, m=np.array([0,0])):
    """
    Pupil function
    """
    dkx=kx[1]-kx[0]
    Rpix=int(radius_k//dkx)

    KX,KY, Z = np.meshgrid(kx,ky, z)
    KX_center=KX[len(kx)//2-Rpix:len(kx)//2+Rpix,len(ky)//2-Rpix:len(ky)//2+Rpix,:]
    KY_center=KY[len(kx)//2-Rpix:len(kx)//2+Rpix,len(ky)//2-Rpix:len(ky)//2+Rpix,:]
    Z_center=Z[len(kx)//2-Rpix:len(kx)//2+Rpix,len(ky)//2-Rpix:len(ky)//2+Rpix,:]

    array_temp = np.zeros((len(kx),len(ky), nz),dtype = 'complex_')

    phase = phaseFromZernikes(zCoeffs,Rpix*2)
    phase = phase.reshape((Rpix*2,Rpix*2,1))
    phase = phase.dot(np.ones([1,nz]))

    C=circle(Rpix,Rpix*2)
    C=C.reshape((Rpix*2,Rpix*2,1))
    C=C.dot(np.ones([1,nz]))

    mask = C*np.exp(1j*phase)*np.exp(-2j*np.pi*Z_center*np.sqrt((n0/lamb)**2-(KX_center*C)**2-(KY_center*C)**2))

    array_temp[len(kx)//2-Rpix:len(kx)//2+Rpix,len(ky)//2-Rpix:len(ky)//2+Rpix,:]=mask

    return array_temp*E_theorique_incident(KX, KY, sigma, m=m)


def E_exp(kx,ky, z,nb, nz,sigma, radius_k, zCoeffs, n0, lamb, m=np.array([0,0])):
    """
    Simulated PSF with aberation zCoeffs
    """
    N,M=len(kx), len(ky)
    x=fftshift(fftfreq(N,kx[1]-kx[0]))
    y=fftshift(fftfreq(M,ky[1]-ky[0]))
    xmax=-2*x[0]
    ymax=-2*y[0]
    Ep=E_pupil(kx,ky, z, nb, nz, sigma, radius_k, zCoeffs, n0, lamb, m=m)
    E = fftshift(fft2(Ep, axes=(0,1)), axes=(0,1))/((xmax*ymax)*(2*np.pi*sigma**2))
    return x,y,E


def I_exp(kx,ky, z,nb, nz,sigma, radius_k, zCoeffs, n0, lamb, m=np.array([0,0])):
    """
    simulated PSF intensity.
    """
    x,y,E = E_exp(kx,ky, z,nb,nz,sigma, radius_k, zCoeffs, n0, lamb, m=m)
    return x,y,np.abs(E)**2

def norm_factor(kx,ky, z,nb, nz,sigma, radius_k, n0, lamb, norm_mode):
    _,_,I=I_exp(kx,ky, z,nb,nz,sigma, radius_k, np.array([0.]), n0, lamb)

    if norm_mode=='max_uniform':
        return np.max(np.abs(I))
    elif norm_mode=='L2_uniform':
        return np.linalg.norm(np.abs(I), ord=2)
    elif norm_mode=='L2_uniform':
        return np.linalg.norm(np.abs(I), ord=1)
    else:
        raise NotImplementedError("La norme demandée n'est pas prise en charge")

def non_uniform_norm(T, norm_mode): #ne marche pas encore très bien pour L1 et L2 ==> se poser la question de norme 2D ou 3D ???
    if norm_mode=='max':
        return T/np.max(np.abs(T))
    elif norm_mode=='L2':
        return T/np.linalg.norm(T, ord=2)
    elif norm_mode=='L2':
        return T/np.linalg.norm(T, ord=1)
    else:
        raise NotImplementedError("La norme demandée n'est pas prise en charge")

def poisson_noise(noise_rate, T):
    return torch.sqrt(torch.poisson(T**2*noise_rate)/noise_rate)

def gaussian_noise(noise_rate, T):
    bias=0.015
    return T+torch.normal(bias*torch.ones_like(T), noise_rate*torch.ones_like(T))

def Unif_boule(N):
    """
    Pick a random point in the dim N unit sphere.
    """
    X=np.random.normal(np.zeros(N), np.ones(N))
    X=X/np.linalg.norm(X)
    R=np.random.rand()
    return R**(1/N)*X

def GenerateDataset(path, dataset_size, batch_size, zmax=10500, xmax=5000, width=0.61, nb=50, nz=50, N=11, ON=1.25, n0=1.33, lamb=1028, alpha=1.8/3.8, r_pupil=3.8, mode='PhaseNet', test=True, norm_mode='max', noise=None, noise_rate=None, freq_res=1, tilt=False, distrib='unif'):
    """Génère une base de données de PSF simulées selon les paramètres expérimentaux. La base de données est sauvegardée dans un dossier "DataSet" au chemin d'accès fourni. DataSet se décompose en deux sous-dossiers "Train" et "Test" contenant respectivement les données entrainement et de test. Dans chacun d'eux sont séparés les PSF simulées ("Samples") des coefficients de Zernike associés ("Coeffs").

    Inputs :

    path : chemin d'accès du dossier où enregistrer le dataset
    dataset_size : nombre d'échantillons produits
    width : valeur absolue maximale des coefficients de Zernike (en unités de lambda)
    nb : longueur et largeur de l'image en sortie (en pixels)
    nz : profondeur de l'image (en nombre de frames)
    N : nombre d'aberrations prises en compte (à l'exception de piston, tip, tilt, et defocus)
    ON : ouverture numérique
    n0 : indice de réfraction
    lamb : longueur d'onde du rayon incident (en nm)
    alpha : rapport sigma_x/r_pupille, où sigma_x est l'écart-type du rayon gaussien incident, et r_pupille le rayon de la pupille arrière du microscope.
    r_pupil : rayon de la pupille arriere du microscope (en mm)
    zmax : profondeur de l'image 3D à générer (en nm)
    xmax : largeur/longueur de l'image 3D à générer (en nm)
    mode : 'PhaseNet'/'TripleNet' pour des données d'entrainement de PhaseNet/TripleNet
    test : Si test==True, génère un dataset de test de même taille dataset_size.
    norm_mode : mode utilisé pour la normalisation. 'max' pour normaliser le maximum, 'L2' pour utiliser la norme euclidienne, 'L1' pour la norme L1. Rajouter le suffixe '_uniform' (ex : 'max_uniform') pour normaliser par rapport à la norme d'une gaussienne de référence (psf sans abérrations)
    noise : pour ajouter dubruit aux données simulées. Peut être 'normal', ou 'poisson'. Par défaut, noise=None.
    noise_rate : intensité du bruit. Doit être > 0 si noise!=None. Par défaut, noise_rate=None.
    freq_res : resulution in Fourier space is 1/(freq_res*xmax). Must be an even integer or 1.

    Outputs :

    trainset (et testset si test==True) : base(s) de données pytorch pouvant être prise(s) en charge par un dataLoader.


    """

    # ######## Create all the necessary directories ############
    if test:
        ds_names=['Train', 'Test']
    else:
        ds_names=['Train']

    try :
        os.mkdir(os.path.join(path, 'Dataset'))
    except:
        pass
    try:
        os.mkdir(os.path.join(path, 'Dataset','Test'))
    except:
        pass
    try:
        os.mkdir(os.path.join(path, 'Dataset', 'Train'))
    except:
        pass
    try:
        os.mkdir(os.path.join(path, 'Dataset', 'Test', 'Samples'))
    except:
        list=os.listdir(os.path.join(path, 'Dataset', 'Test','Samples'))
        for filename in list:
            os.remove(os.path.join(path, 'Dataset', 'Test', 'Samples', filename))
    try:
        os.mkdir(os.path.join(path, 'Dataset', 'Test', 'Coeffs'))
    except:
        list=os.listdir(os.path.join(path, 'Dataset', 'Test', 'Coeffs'))
        for filename in list:
            os.remove(os.path.join(path, 'Dataset', 'Test', 'Coeffs', filename))
    try:
        os.mkdir(os.path.join(path, 'Dataset', 'Train', 'Samples'))
    except:
        list=os.listdir(os.path.join(path, 'Dataset', 'Train', 'Samples'))
        for filename in list:
            os.remove(os.path.join(path, 'Dataset', 'Train', 'Samples', filename))
    try:
        os.mkdir(os.path.join(path, 'Dataset', 'Train', 'Coeffs'))
    except:
        list=os.listdir(os.path.join(path, 'Dataset', 'Train', 'Coeffs'))
        for filename in list:
            os.remove(os.path.join(path, 'Dataset', 'Train', 'Coeffs', filename))
    if tilt:
        try:
            os.mkdir(os.path.join(path, 'Dataset', 'Train', 'Tilts'))
        except:
            list=os.listdir(os.path.join(path, 'Dataset', 'Train', 'Tilts'))
            for filename in list:
                os.remove(os.path.join(path, 'Dataset', 'Train', 'Tilts', filename))
        try:
            os.mkdir(os.path.join(path, 'Dataset', 'Test', 'Tilts'))
        except:
            list=os.listdir(os.path.join(path, 'Dataset', 'Test', 'Tilts'))
            for filename in list:
                os.remove(os.path.join(path, 'Dataset', 'Test', 'Tilts', filename))
    else:
        try:
            shutil.rmtree(os.path.join(path, 'Dataset', 'Train', 'Tilts'))
            shutil.rmtree(os.path.join(path, 'Dataset', 'Test', 'Tilts'))
        except:
            pass

    assert freq_res%2==0 or freq_res==1

    # ###########################################################
    radius_k=ON/lamb
    sigma = radius_k*alpha
    R=nb/(2*xmax) # en nm^{-1}
    kx = np.linspace(-R,R,freq_res*nb) # en nm^{-1}
    ky = np.linspace(-R,R,freq_res*nb)
    z= np.linspace(-zmax/2,zmax/2, nz)

    assert distrib in ['unif', 'L2'], "distrib should be either 'unif' or 'L2'."
    if distrib=='unif':
        def RandomCoeffs():
            return np.random.rand(N)*width*2-width
    elif distrib=='L2':
        def RandomCoeffs():
            return Unif_boule(N)*width

    assert norm_mode in ['max', 'L1', 'L2', 'max_uniform', 'L1_uniform', 'L2_uniform'], "norm_mode has to be either 'max', 'L1', 'L2', 'max_uniform', 'L1_uniform', or 'L2_uniform'."

    Unif=norm_mode.split('_')
    if Unif[-1]=='uniform':
        factor=norm_factor(kx, ky, z, nb, nz, sigma, radius_k, n0, lamb, norm_mode)
        norm=lambda T:T/factor
    else:
        norm=lambda T: non_uniform_norm(T, norm_mode)

    assert noise in ['poisson', 'normal', None], "noise has to be either 'poisson', 'normal' or None."

    if noise=='poisson':
        assert noise_rate!=None, "noise_rate needs to be specified if noise != None."
        assert noise_rate>0, "noise_rate has to be > 0"

        trans=partial(poisson_noise, noise_rate)
    elif noise=='normal':
        assert noise_rate!=None, "noise_rate needs to be specified if noise != None."

        trans=partial(gaussian_noise, noise_rate)
    else:
        trans=None

    if mode=='PhaseNet':

        zCoeffs=np.zeros([batch_size,N+4], dtype=np.float32)
        tzCoeffs=torch.from_numpy(zCoeffs)
        I=np.zeros([batch_size,nb,nb,nz], dtype=np.float32)
        tI=torch.from_numpy(I)
        print('Compute Dataset...')
        Samples=[]
        Coeffs=[]
        Tilts=[]
        for name in ds_names:
            for i in tqdm(range(dataset_size), mininterval=1):

                zCoeffs[i % batch_size,4:] = RandomCoeffs()
                _,_,I[i % batch_size,:,:,:] = I_exp(kx,ky, z,nb, nz, sigma, radius_k, zCoeffs[i% batch_size,:], n0, lamb)
                I[i % batch_size,:,:,:]=norm(I[i % batch_size,:,:,:])
                if i % batch_size == batch_size-1:
                    torch.save(tI, os.path.join(path, 'Dataset', name, 'Samples', f'sample{i//batch_size}.pt'))
                    torch.save(tzCoeffs[:,4:], os.path.join(path, 'Dataset', name, 'Coeffs', f'coeffs{i//batch_size}.pt'))

                    if name == 'Train':
                        Samples.append(f'sample{i//batch_size}.pt')
                        Coeffs.append(f'coeffs{i//batch_size}.pt')
                else:
                    pass


        trainset=PhaseNetDataset(Coeffs, Samples, os.path.join(path, 'Dataset', 'Train'), transform=trans)
        if test:
            testset=PhaseNetDataset(Coeffs, Samples, os.path.join(path, 'Dataset', 'Test'), transform=trans)
        else:
            pass

    elif mode=='TripleNet':

        zCoeffs=np.zeros([batch_size,N+4], dtype=np.float32)
        tzCoeffs=torch.from_numpy(zCoeffs)
        IXY=np.zeros([batch_size,nb,nb], dtype=np.float32)
        tIXY=torch.from_numpy(IXY)
        IXZ=np.zeros([batch_size,nb,nz], dtype=np.float32)
        tIXZ=torch.from_numpy(IXZ)
        IYZ=np.zeros([batch_size,nb,nz], dtype=np.float32)
        tIYZ=torch.from_numpy(IYZ)
        batch_tilts=torch.zeros([batch_size,2])
        print('Compute Dataset...')
        Samples=[]
        Coeffs=[]
        Tilts=[]

        for name in ds_names:
            for i in tqdm(range(dataset_size), mininterval=1):
                zCoeffs[i % batch_size,4:] = RandomCoeffs()
                if tilt:
                    batch_tilts[i % batch_size,:]=torch.normal(torch.zeros([2]),radius_k/2*torch.ones([2])) # tilt dans l'espace de fourier

                _,_,I_temp = I_exp(kx,ky, z,nb, nz, sigma, radius_k, zCoeffs[i % batch_size,:], n0, lamb, m=batch_tilts[i % batch_size,:].numpy())
                I_temp=I_temp[freq_res*nb//2-nb//2:freq_res*nb//2+nb//2,freq_res*nb//2-nb//2:freq_res*nb//2+nb//2,:]
                I_temp=norm(I_temp)
                IXY[i % batch_size,:]=I_temp[:,:,nz//2]
                IXZ[i % batch_size,:]=I_temp[:,nb//2,:]
                IYZ[i % batch_size,:]=I_temp[nb//2,:,:]
                if i % batch_size == batch_size-1:
                    torch.save([tIXY, tIXZ, tIYZ], os.path.join(path, 'Dataset', name, 'Samples', f'sample{i//batch_size}.pt'))
                    torch.save(tzCoeffs[:,4:], os.path.join(path, 'Dataset', name, 'Coeffs', f'coeffs{i//batch_size}.pt'))

                    if name=='Train':
                        Samples.append(f'sample{i//batch_size}.pt')
                        Coeffs.append(f'coeffs{i//batch_size}.pt')

                    if tilt:
                        torch.save(batch_tilts*lamb*r_pupil/ON, os.path.join(path, 'Dataset', name, 'Tilts', f'tilt{i//batch_size}.pt')) # On enregistre les tilts convertis dans l'espace des distances réelles.
                        if name=='Train':
                            Tilts.append(f'tilt{i//batch_size}.pt')

        if tilt:
            trainset=TiltTripleNetDataset(Coeffs, Samples, Tilts, os.path.join(path, 'Dataset', 'Train'), transform=trans)
            if test:
                testset=TiltTripleNetDataset(Coeffs, Samples, Tilts, os.path.join(path, 'Dataset', 'Test'), transform=trans)
        else:
            trainset=TripleNetDataset(Coeffs, Samples, os.path.join(path, 'Dataset', 'Train'), transform=trans)
            if test:
                testset=TripleNetDataset(Coeffs, Samples, os.path.join(path, 'Dataset', 'Test'), transform=trans)
    else :
        raise NotImplementedError("Only two modes allowed : 'PhaseNet' and 'TripleNet'.")

    if test:
        return trainset, testset
    else:
        return trainset
