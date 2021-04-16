#!/usr/local/bin/python3
#
# Stitching des fichiers moyennes de davis
#
# usage :./main_stitch_instPIV.py --favg DATA_Davis/Ma1080
#                                 --fin DATA_davis/Ma1080
#                                 --fout DATA_Stitched
#                                 --Mj 1.080
#
#
##########################################################

import argparse

import importlib
import pivFunctions
importlib.reload(pivFunctions)

import PIVdatamod
importlib.reload(PIVdatamod)

from openpiv import filters,validation

from scipy.io  import loadmat
import numpy as np

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


def convertPIV(Mj,FolderAvg,FolderIn,outfilename):
    #== Reorganize the mean PIV fields to get the interpolation grid
    # get the grid from the average data
    stitchGrid_dict = pivFunctions.create_StitchGrid(folderAvg+'/B00001_AvgV.vc7',1200,501)
#== preparing the PIV data format
    gridType=67
    dx=(stitchGrid_dict['Xint'][1]-stitchGrid_dict['Xint'][0])/10
    dy=(stitchGrid_dict['Yint'][1]-stitchGrid_dict['Yint'][0])/10
    x0=stitchGrid_dict['Xint'][0]/10
    y0=stitchGrid_dict['Yint'][0]/10

    nx=1200; ny=501; nc=2; ns=0; fs=1
    dataPIVout=PIVdatamod.PIVdata()
    dataPIVout.create(gridType,dx,dy,x0,y0,nx,ny,nc,ns,fs)
    dataPIVout.write_header(outfilename)

    Uj = Uj_Mj(Mj)
    u_out=np.zeros([nx,ny,nc],dtype='float32')

    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #     MAIN FUNCTION CORE
    filesINST=glob(FolderIn+'/B?????.vc7')
    ifile=0
    for inst_PIVfile in filesINST:
        #read instantaneous data
        u1,v1,u2,v2 = pivFunctions.readPIVvc7_fieldsOnly(inst_PIVfile)

        # u1, v1, mask = validation.local_median_val(
        #     u1, v1, u_threshold = 15, v_threshold = 5, size=1 )
        # u1, v1  = filters.replace_outliers(
        #     u1, v1, method='localmean', max_iter=5, kernel_size=2)

        # u2, v2, mask = validation.local_median_val(
        #     u2, v2, u_threshold = 15, v_threshold = 5, size=1)
        # u2, v2  = filters.replace_outliers(
        #     u2, v2, method='localmean', max_iter=5, kernel_size=2)

        #interpolate the data
        data_inst = pivFunctions.arrange_PIV(u1,v1,u2,v2,stitchGrid_dict)

        #Store the data into PIVdata format
        u_out[:,:,0] = data_inst['U'].T/Uj
        u_out[:,:,1] = data_inst['V'].T/Uj
        ifile+=1
        if (ifile % 1000 == 0):
            print(Mj,':',ifile)

        # output the data into the PIV data format
        dataPIVout.append_sample(u_out)
    ## === end function convertPIV()

############################################################################@@@@

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#          Args parser
parser = argparse.ArgumentParser()
parser.add_argument('--favg' ,action='store', dest='folderAvg')
parser.add_argument('--fin' ,action='store', dest='folderInst')
parser.add_argument('--fout',action='store', dest='folderOUT')
parser.add_argument('--Mj'  ,action='store', dest='Mj',type=float)

args = parser.parse_args()


#######################################################
# Main program core
#######################################################
folderAvg= args.folderAvg
folderInst= args.folderInst
folderOUT= args.folderOUT
Mj=args.Mj
print(folderAvg)


#convert whole file ------------------------------------
outfile=folderOUT + '/Mj_%4d' % int(Mj*1000) +'/Mj_%4d' % int(Mj*1000) + '.bin'
convertPIV(Mj,folderAvg,folderInst,outfile)
