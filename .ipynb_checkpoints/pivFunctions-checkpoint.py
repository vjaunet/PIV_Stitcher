#
#   some useful PIV functions
#
#====================================================================

import ReadIM
import numpy as np

from scipy.interpolate import interp2d,interp1d
from scipy.optimize import minimize
from scipy.ndimage  import rotate

#import matplotlib.pyplot as plt
from math import atan2,pi

#============================================================
# High level functions
#============================================================
def create_StitchGrid(file_name,nx_int=1200,ny_int=501):
    """ 
        Create a grid based on a specific data file.
        For example, call create_StitchGrid() on average values
        and call stitchPIV() on instantaneous data.
    """
    #=== read the vc7 file
    x1,y1,u1,v1,x2,y2,u2,v2,buff= readPIVvc7(file_name)

    #=== Find rotation and trasnlation in vertical 
    # direction on both PIV fields
    alpha1,y0 = rotatePIV(x1,y1,u1)
    y1 = y1 - y0
    print("Camera1: alpha=",alpha1,"Delta_y=",y0)
    
    alpha2,y0 = rotatePIV(x2,y2,u2)
    y2 = y2 - y0
    print("Camera2: alpha=",alpha2,"Delta_y=",y0)
 
    #=== find Dx between first and second field
     #rotate the PIVfields
    u1 = rotate(u1,alpha1,reshape=False)
    v1 = rotate(v1,alpha1,reshape=False)

    u2 = rotate(u2,alpha2,reshape=False)
    v2 = rotate(v2,alpha2,reshape=False)

    # remove padded areas due to the rotation
    x1_1=x1[0:-2]
    u1=u1[:,0:-2]
    v1=v1[:,0:-2]

    x2_1=x2[1:-4]
    u2=u2[:,1:-4]
    v2=v2[:,1:-4]
    
    Dx = translatePIV(x1_1,y1,u1,x2_1,y2,u2)
    print("Dx =",Dx)

    #=== Creation of the interpolation grid
    x_int = np.linspace(x1[0],x2[-1],nx_int)
    ydeb= -min(abs(y1[0]),abs(y1[-1]))
    yfin= -ydeb
    y_int = np.linspace(ydeb,yfin,ny_int)
          
    return { 'Xint':x_int,'Yint':y_int,
             'x1':x1,'y1':y1,
             'x2':x2,'y2':y2,
             'alpha1':alpha1,'alpha2':alpha2, 'Dx':Dx}


def stitchPIV(file_name,stitchPIV_grid_dict):
    """ 
        Stitching process using a grid defined by using 
        the create_StitchGrid() function     
    """
    u1,v1,u2,v2=readPIVvc7_fieldsOnly(file_name)
    dict_mean = arrange_PIV(u1,v1,u2,v2,stitchPIV_grid_dict)
    return dict_mean




#============================================================
# READING DATA
#============================================================
def readPIVvc7(filename):
    """ Read DaVis vc7 file """
    ###############################################################
    # Be careful data might be transposed and, x and y inverted
    # it depends on the calibration wich read internally by Davis
    ##############################################################

    buff, piv_atts=  ReadIM.extra.get_Buffer_andAttributeList(filename)
    u1,v1,u2,v2 = readPIVvc7_fieldsOnly(filename)

    #create the grids
    x1=np.linspace(buff.scaleY.offset +
                   buff.scaleY.factor*buff.ny*buff.vectorGrid,
                   buff.scaleY.offset,
                   buff.ny)

    
    y1=np.linspace(buff.scaleX.offset,
                   buff.scaleX.offset +
                   buff.scaleX.factor*buff.nx*buff.vectorGrid,
                   buff.nx)
    
    if buff.nf == 1:

        ReadIM.DestroyBuffer(buff)
        ReadIM.DestroyAttributeListSafe(piv_atts)

        return (x1, y1,
                dataout[0,:,:].T,
                dataout[1,:,:].T,
                buff)

    elif buff.nf == 2:

        #New Davis 10 procedure for storing average data.
        x2 = x1
        y2 = y1    
        
        # Careful : need to extract a portion of the field 
        # Davis stores the entire grid of data for each camera
        # although each camera may not contribute to the entire fov.
        # Davis puts zeros where the camera doesn't see.
        x1 = x1[0:615]
        y1 = y1[55:-55]

        x2 = x2[420:1202]
        y2 = y2[50:-50]

        ReadIM.DestroyBuffer(buff)
        ReadIM.DestroyAttributeListSafe(piv_atts)
        
        
        #plt.contourf(x2,y2,u2)
        #plt.plot(u2[:,775])
        #plt.plot(u2[::-1,775])

        return (x1, y1, u1, v1,
                x2, y2, u2, v2, buff)

def readPIVvc7_fieldsOnly(filename):
    """  
            Read the vc7 file
            return Velocity only 
    """
    buff, piv_atts=  ReadIM.extra.get_Buffer_andAttributeList(filename)
    data, buff = ReadIM.extra.buffer_as_array(buff)

    data = -1.*data*buff.scaleI.factor

    #transpose, the software expects (ny,nx tables)
    data = np.transpose(data)
    data = np.moveaxis(data,-1,0)

    u1 = data[1,:,::-1]
    v1 = data[0,:,::-1]
    u2 = data[3,:,::-1]
    v2 = data[2,:,::-1]

    # Careful : need to extract a portion of the field 
    # Davis stores the entire grid of data for each camera
    # although each camera may not contribute to the entire fov.
    # Davis puts zeros where the camera doesn't see.
    u1 = u1[55:-55,0:615]
    v1 = v1[55:-55,0:615]

    u2 = u2[50:-50,420:1202]
    v2 = v2[50:-50,420:1202]
    
    ReadIM.DestroyBuffer(buff)
    ReadIM.DestroyAttributeListSafe(piv_atts)

    #return only velocities
    return (u1,v1,u2,v2)

def read_INSTPIV_fieldsOnly(filename):
    """ read the vc7 file
        for Davis version lower than Davis 10
    """ 
    buff, piv_atts=  ReadIM.extra.get_Buffer_andAttributeList(filename)
    data,_ = ReadIM.extra.buffer_as_array(buff)

    if (data.shape[0]<=buff.nf*2):
        raise ValueError('Davis data format not supported by this routine. Try readPIVvc7')
        
    # get the velocity - Frame 1
    # we discard the choices > 3 that correspond to secondary peaks.
    choices = np.where(data[0] > 3, 3, data[0])

    u1 = np.empty(choices.shape)
    v1 = np.empty(choices.shape)
    # the different data buffers contain the successively computed
    # displacements in a cumuluative way
    for (i,j),choice in np.ndenumerate(choices):
        u1[i,j]=np.sum(data[1:2*int(choice)+1:2][:,i,j])*buff.scaleI.factor
        v1[i,j]=np.sum(data[2:2*int(choice)+2:2][:,i,j])*buff.scaleI.factor

    # get the velocity - Frame 2
    # we discard the choices > 3 that correspond to secondary peaks.
    choices = np.where(data[10] > 3, 3, data[10])

    u2 = np.empty(choices.shape)
    v2 = np.empty(choices.shape)
    # the different data buffers contain the successively computed
    # displacements in a cumuluative way
    for (i,j),choice in np.ndenumerate(choices):
        u2[i,j]=np.sum(data[11:2*int(choice)+11:2][:,i,j])*buff.scaleI.factor
        v2[i,j]=np.sum(data[12:2*int(choice)+12:2][:,i,j])*buff.scaleI.factor
            
    ReadIM.DestroyBuffer(buff)
    ReadIM.DestroyAttributeListSafe(piv_atts)

    #return only velocities
    return u1,v1,u2,v2

#============================================================
#      interpolation and optimisation
#============================================================

#============================================================
def rotatePIV(x,y,u):
    """ Rotate PIV fields to find the jet axis
    """
    res= minimize(symPIV ,0, args=(y,u[:,30]))
    dy0=res.x

    res= minimize(symPIV ,0, args=(y,u[:,-30]))
    dy1=res.x

    alpha = atan2(dy1-dy0, x[-30]-x[30])*180./pi
    y0= (dy0+dy1)/2

    return alpha,y0


#============================================================
def symPIV(dy,y,u):
    """ Function to minimize in order to find
        the symmetry axis of a velocity profile
    """
    y=y-dy

    #== creating a reversed vertical axis
    ydeb= -min(abs(y[0]),abs(y[-1]))
    yfin= -ydeb

    #== interpolating the input profile on both
    # original and reversed axes
    yint = np.linspace(ydeb,yfin,400)
    f = interp1d( y,u, kind='cubic')
    g = interp1d(-y,u, kind='cubic')
    up = f(yint)
    um = g(yint)

    #== compute and return the difference between original
    # and reversed velocity profiles
    err=sum(abs(up-um))
    return err

#============================================================
#============================================================
def arrange_PIV(u1,v1,u2,v2,grid_dict):
    """
    Arranging (rotating and translating) the velocity fields
    according to the optimized grid and rotation obtained via a
    prior call to create_grids()
    """

    # for convenience....
    x_int = grid_dict["Xint"]
    y_int = grid_dict["Yint"]
    x1 = grid_dict["x1"]
    x2 = grid_dict["x2"]
    y1 = grid_dict["y1"]
    y2 = grid_dict["y2"]
    alpha1 = grid_dict["alpha1"]
    alpha2 = grid_dict["alpha2"]
    Dx = grid_dict["Dx"]
    
    #rotate the PIVfields
    u1 = rotate(u1,alpha1,reshape=False)
    v1 = rotate(v1,alpha1,reshape=False)

    u2 = rotate(u2,alpha2,reshape=False)
    v2 = rotate(v2,alpha2,reshape=False)

    # remove padded areas due to the rotation
    x1=x1[0:-2]
    u1=u1[:,0:-2]
    v1=v1[:,0:-2]

    x2=x2[1:-4]
    u2=u2[:,1:-4]
    v2=v2[:,1:-4]
    
    U_rot=interpolPIV(x_int,y_int,
                      x1,y1,u1,
                      x2,y2,u2,Dx)
    V_rot=interpolPIV(x_int,y_int,
                      x1,y1,v1,
                      x2,y2,v2,Dx)

    #removing last values if necessary
    x_int=x_int[0:]
    U_rot=U_rot[:,0:]
    V_rot=V_rot[:,0:]

    #put everything in a dict()
    return {'X':x_int,
            'Y':y_int,
            'U':U_rot,
            'V':V_rot}

#========================================================
def interpolPIV(x_int,y_int, x1,y1,u1, x2,y2,u2,Dx):
    """ ==== 2D interpoaltion of a vector field =====
           x_int, y_int is the grid to interpoalte on
    """
    
    f = interp2d(x1, y1, u1, kind='cubic')
    u_int1 = f(x_int,y_int)

    g = interp2d(x2+Dx, y2, u2, kind='cubic')
    u_int2 = g(x_int,y_int)

    test=np.where(x_int<=x2[0])
    imax = max(test[0])
    U_int = np.concatenate((u_int1[:,0:imax],
                            u_int2[:,imax:]),1)

    return U_int

def translatePIV(x1,y1,u1,x2,y2,u2):
        """
            Find the horizontal offset between the two cameras
        """          
        res= minimize(resDxPIV ,0, args=(x1,y1,u1,x2,y2,u2))
        dx=res.x
        return dx
    
def resDxPIV(dx,x1,y1,u1,x2,y2,u2):
    """ 
        Returns residuals to be minimized to find the most 
        precise Dx between both cameras
    """
    f2d = interp2d(x1, y1, u1, kind='cubic')
    x_int = x1[-3:] + dx
    y_int = y2
    u_int1 = f2d(x_int,y_int)

    f2d = interp2d(x2, y2, u2, kind='cubic')
    x_int = x1[-3:]
    u_int2 = f2d(x_int,y_int)

    err = np.sum(abs(u_int1[:,:] - u_int2[:,:]))
    return err



#=================================================================================
# check plots
# import matplotlib.pyplot as plt
# plt.figure(21);
# plt.plot(y1,u1[:,10],  'r',-y1,u1[:,10],  'b')
# plt.plot(y1,u1[:,400],  'r',-y1,u1[:,400],  'b')

# sanity check
# import matplotlib.pyplot as plt
# plt.figure(20);
# plt.plot(y_int,U_rot[:,10],  'r',-y_int,U_rot[:,10],  'b')
# plt.plot(y_int,U_rot[:,300], 'r',-y_int,U_rot[:,300], 'b')
# plt.plot(y_int,U_rot[:,600], 'r',-y_int,U_rot[:,600], 'b')
# plt.plot(y_int,U_rot[:,1100],'r',-y_int,U_rot[:,1100],'b')
# plt.grid()
# plt.show()

# check plot
# import matplotlib.pyplot as plt
# fig,axs = plt.subplots(1,1, sharex=True)
# axs.pcolor(x2[1:50],y2,u2[:,1:50],cmap='RdBu')


# check plot
# import matplotlib.pyplot as plt
# fig,axs = plt.subplots(1,1, sharex=True)
# axs.pcolor(x_int[1:-1:5],y_int,U_rot[:,1:-1:5],cmap='RdBu')
