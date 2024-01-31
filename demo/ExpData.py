import numpy as np
import torch as torch
import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_img(x, ax, norm=None):
    """
    Affiche une image passée en entrée.
    """
    try :
        x=x/norm
    except:
        if type(x)==np.ndarray:
            max=np.max(x)
            min=np.min(x)
            x=(x-min)/(max-min)

        elif type(x)==torch.Tensor:
            max=torch.max(x)
            min=torch.min(x)
            x=(x-min)/(max-min)

    assert type(x)==torch.Tensor or type(x)==numpy.ndarray, "type non pris en charge."

    if type(x)==np.ndarray:
        assert np.all(x >= 0) and np.all(x <= 1), "Les valeurs de l'image doivent être entre 0 et 1!"
    elif type(x)==torch.Tensor:
        assert torch.all(x >= 0) and torch.all(x <= 1), "Les valeurs de l'image doivent être entre 0 et 1!"

    ax.imshow(x, vmin=0, vmax=1, interpolation="nearest", cmap='gray')

def nested_change(item, func):
    if isinstance(item, list):
        return [nested_change(x, func) for x in item]
    return func(item)

def txt_to_tensor(file_path):
    _,ext=os.path.splitext(file_path)
    if ext=='.txt':
        with open(file_path, 'r') as file:
            contents = file.read()
            l=[r.split('\t')[:-1] for r in contents.split('\n')[:-1]]
            l=nested_change(l, np.float32)

            t=torch.tensor(l, dtype=torch.float32)
        return t
    else:
        if ext=='':
            raise ValueError('The path given does not point on a file.')
        else:
            raise ValueError(f'Expected extension .txt but found {ext}')

def plot_txt(file_path, ax, norm=None):
    t=txt_to_tensor(file_path)
    plot_img(t, ax, norm=norm)


