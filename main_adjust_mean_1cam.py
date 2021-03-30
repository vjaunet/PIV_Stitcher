#!/usr/local/bin/python3.9
#
#
#     Usage :
#        ./main_adjust_mean.py --avg ../DATA_Davis/Ma1080/ \
#                              --f ../DATA_Davis/Ma1080/B00001_AvgV.vc7
#                              --Mj 1.080
#
#
##########################################################

import argparse

import importlib
import pivFunctions
importlib.reload(pivFunctions)

import numpy as np

from glob import glob
import re

import matplotlib.pyplot as plt

##############################################@@@@@@@@
def Mj_NPR(NPR):
    g=1.4
    return (2.0/(g-1)*(NPR**((g-1)/g)-1))**0.5

def Uj_Mj(Mj,T0=293):
    g=1.4
    R=287
    return np.sqrt(g*R*T0/(1. + (g-1.)/2.*Mj**2))*Mj

############################################################################@@@@

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#          Args parser
parser = argparse.ArgumentParser()
parser.add_argument('--avg' ,action='store', dest='folderAvg')
parser.add_argument('--f' ,action='store', dest='fileAvg')
parser.add_argument('--Mj'  ,action='store', dest='Mj',type=float)

args = parser.parse_args()


#######################################################
# Main program core
#######################################################
folderAvg= args.folderAvg  #
fileAvg= args.fileAvg      #input DAVIS mean file path
Mj=args.Mj

print(fileAvg)

# convert whole file ------------------------------------
file_mean=fileAvg

grid_dict=pivFunctions.create_grid_1cam(file_mean)
x1,y1,u1,v1= pivFunctions.readPIVvc7(file_mean)

#scaling
u1 = u1/Uj_Mj(Mj)
v1 = v1/Uj_Mj(Mj)

#arranging
mean_data = pivFunctions.arrange_PIV_1cam(u1,v1,grid_dict)

# output the file =======================================

x = np.tile(mean_data["X"],mean_data["Y"].size)
y = mean_data["Y"].repeat(mean_data["X"].size)

plt.figure(1)
plt.contourf(u1)
plt.show()

plt.figure(2)
plt.contourf(mean_data['X'],mean_data['Y'],mean_data['Um'][:,:])
plt.show()

#print(mean_data['Um'].shape)

dataout = np.vstack((x.flatten()/10, \
                     y.flatten()/10, \
                     mean_data["Um"].flatten(),\
                     mean_data["Vm"].flatten())).T
print('shape: ',mean_data['Um'].shape)

np.savetxt('DATA_Stitched'+'/Mj_%4d' % int(Mj*1000) + '_mean.dat',
           dataout, fmt='%.6e',
           header='#x/D y/D U/Uj V/Uj',
           delimiter="\t")
