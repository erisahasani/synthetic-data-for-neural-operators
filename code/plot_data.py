import numpy as np
import torch
from time import sleep
import matplotlib.pyplot as plt

"""
Once data is generated, can use this to plot the generated data
"""

def plot_data_1d(f,title, name):
    size = len(f)
    cor = np.linspace(0,1,size)
    plt.figure()
    plt.plot(cor,f)
    plt.title(title, loc='left')
    plt.savefig(f'{name}.jpg')
    plt.close()

def plot_data_2d(f,title,name):
    plt.figure()
    plt.pcolormesh(f)
    plt.colorbar()
    plt.title(title, loc='left')
    plt.savefig(f'{name}.jpg')
    plt.close()


if __name__ == '__main__':
    x_test = torch.load('synthetic_data_25_x.pt').type(torch.FloatTensor)
    y_test = torch.load('synthetic_data_25_y.pt').type(torch.FloatTensor)
    print(x_test.shape)
    print(y_test.shape)  

    for test in range(0,len(x_test)):
        plot_data_2d(x_test[test,0,:,:],f'input function f',f'input_output')
        sleep(1)
        plot_data_2d(y_test[test,:,:],f'output function u',f'input_output')
        sleep(1)
