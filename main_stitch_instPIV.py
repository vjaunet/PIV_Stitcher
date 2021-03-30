#!/usr/bin/python3.5
#
#
#
#
##########################################################

import argparse

import importlib
import pivFunctions
import PIVdatamod
importlib.reload(pivFunctions)
importlib.reload(PIVdatamod)

from scipy.io  import loadmat
import numpy as np

from openpiv import filters,validation

from glob import glob
import re

##############################################@@@@@@@@
def Mj_NPR(NPR):
    g=1.4
    return (2.0/(g-1)*(NPR**((g-1)/g)-1))**0.5

def Uj_Mj(Mj,T0=293):
    g=1.4
    R=287
    return np.sqrt(g*R*T0/(1. + (g-1.)/2.*Mj**2))*Mj


def convertPIV(Mj,folderAvg,filesINST,outfilename):
    #== Reorganize the mean PIV fields to get the interpolation grid
    #print(folderAvg+'/B00001_AvgV.vc7')
    data_SUCRE,grid_dict = pivFunctions.getPIV_AvgRms(folderAvg+'/B00001_AvgV.vc7',
                                                      folderAvg+'/B00002_StdevV.vc7')

    #== preparing the PIV data format
    gridType=67
    dx=(grid_dict['Xint'][1]-grid_dict['Xint'][0])/10
    dy=(grid_dict['Yint'][1]-grid_dict['Yint'][0])/10
    x0=grid_dict['Xint'][0]/10
    y0=grid_dict['Yint'][0]/10

    nx=1200; ny=401; nc=2; ns=0; fs=1
    dataPIVout=PIVdatamod.PIVdata()
    dataPIVout.create(gridType,dx,dy,x0,y0,nx,ny,nc,ns,fs)
    dataPIVout.write_header(outfilename)

    Uj = Uj_Mj(Mj)
    u_out=np.zeros([nx,ny,nc],dtype='float32')

    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #     MAIN FUNCTION CORE
    ifile=0
    for inst_PIVfile in filesINST:
        #read instantaneous data
        u1,v1,u2,v2 = pivFunctions.read_INSTPIV_fieldsOnly(inst_PIVfile)

        u1, v1, mask = validation.local_median_val(
            u1, v1, u_threshold = 15, v_threshold = 5, size=1 )
        u1, v1  = filters.replace_outliers(
            u1, v1, method='localmean', max_iter=5, kernel_size=2)

        # u1, v1, mask = validation.global_std(
        #     u1, v1, std_threshold = 3.5)
        # u1, v1  = filters.replace_outliers(
        #     u1, v1, method='localmean', max_iter=5, kernel_size=2)

        u2, v2, mask = validation.local_median_val(
            u2, v2, u_threshold = 15, v_threshold = 5, size=1)
        u2, v2  = filters.replace_outliers(
            u2, v2, method='localmean', max_iter=5, kernel_size=2)

        # u2, v2, mask = validation.global_std(
        #     u2, v2, std_threshold = 3.5)
        # u2, v2  = filters.replace_outliers(
        #     u2, v2, method='localmean', max_iter=5, kernel_size=2)

        #interpolate the data
        data_inst = pivFunctions.arrange_PIV(u1,v1,u2,v2,grid_dict)

        #Store the data into PIVdata format
        u_out[:,:,0] = data_inst['Um'].T/Uj
        u_out[:,:,1] = data_inst['Vm'].T/Uj
        ifile+=1
        if (ifile % 1000 == 0):
            print(Mj,':',ifile)

        # output the data into the PIV data format
        dataPIVout.append_sample(u_out)

    #end function convertPIV()

############################################################################@@@@

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#          Args parser
parser = argparse.ArgumentParser()
parser.add_argument('--avg' ,action='store', dest='folderAvg')
parser.add_argument('--ext' ,action='store', dest='ext')
parser.add_argument('--inst',action='store', dest='folderInst')
parser.add_argument('--Mj'  ,action='store', dest='Mj',type=float)

args = parser.parse_args()


#######################################################
# Main program core
#######################################################
folderInst=args.folderInst
folderAvg= args.folderAvg
Mj=args.Mj

print(folderAvg)


#convert whole file ------------------------------------
outfile=folderInst + '/Mj_%4d' % int(Mj*1000) + args.ext + '_test.bin'
filesInst = sorted(glob(folderInst+'/B0000?'+'.vc7'))
convertPIV(Mj,folderAvg,filesInst,outfile)

print('test Written')

outfile=folderInst + '/Mj_%4d' % int(Mj*1000) + args.ext + '.bin'
filesInst = sorted(glob(folderInst+'/B*'+'.vc7'))
convertPIV(Mj,folderAvg,filesInst,outfile)
