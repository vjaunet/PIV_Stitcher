#!/usr/local/bin/python3
#
# Stitching des fichiers moyennes de davis
#
# usage :./main_stitch_meanPIV.py --fin ../DATA_Davis/Ma1080
#                                 --fout DATA_Stitched
#                                 --Mj 1.080
#
#
##########################################################

import argparse

import importlib
import pivFunctions
importlib.reload(pivFunctions)

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


def convertPIV(Mj,FolderIn,outfilename):
    #== Reorganize the mean PIV fields to get the interpolation grid
    # get the grid from the average data
    stitchGrid_dict = pivFunctions.create_StitchGrid(folderAvg+'/B00001_AvgV.vc7',1200,501)
    dataMean = pivFunctions.stitchPIV(folderAvg+'/B00001_AvgV.vc7',stitchGrid_dict)
    dataRMS  = pivFunctions.stitchPIV(folderAvg+'/B00002_StdDevV.vc7',stitchGrid_dict)



    x = np.tile(dataMean["X"],dataMean["Y"].size)
    y = dataMean["Y"].repeat(dataMean["X"].size)
    dataout = np.vstack((x.flatten()/10, \
                       y.flatten()/10, \
                       dataMean["U"].flatten(),\
                       dataMean["V"].flatten(),\
                       dataRMS["U"].flatten(),
                       dataRMS["V"].flatten())).T

    print(dataout.shape)

    np.savetxt(outfilename,
               dataout, fmt='%.6e',
               header='#x/D y/D U/Uj V/Uj Urms/Uj Vrms/Uj',
               delimiter="\t")

    ## === end function convertPIV()

############################################################################@@@@

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#          Args parser
parser = argparse.ArgumentParser()
parser.add_argument('--fin' ,action='store', dest='folderAvg')
parser.add_argument('--fout',action='store', dest='folderOUT')
parser.add_argument('--Mj'  ,action='store', dest='Mj',type=float)

args = parser.parse_args()


#######################################################
# Main program core
#######################################################
folderAvg= args.folderAvg
folderOUT= args.folderOUT
Mj=args.Mj
print(folderAvg)


#convert whole file ------------------------------------
outfile=folderOUT + '/Mj_%4d' % int(Mj*1000) + '.dat'
convertPIV(Mj,folderAvg,outfile)
