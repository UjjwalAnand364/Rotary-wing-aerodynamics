import numpy as np
from Params import vertical_stabilizers,horizontal_stabilizers
from helper_functions import *

def yaw_vertical_fin(delE, rho, V_inf, Cd=0.01):

    Cl_alpha=vertical_stabilizers["Cl_alpha"]
    Cl0=vertical_stabilizers["Cl0"]
    S = vertical_stabilizers["verti_area"]
    
    x_arm = vertical_stabilizers["x_arm"]
    z_arm = vertical_stabilizers["z_arm"]

    Cl = Cl0 + Cl_alpha * np.radians(delE)
    L = 0.5 * rho * V_inf**2 * S * Cl
    yaw = L * x_arm
    roll = L * z_arm

    return yaw,roll

def elevator_pitch(delR, rho, V_inf):
    Cl_alpha=horizontal_stabilizers["Cl_alpha"]
    Cl0=horizontal_stabilizers["Cl0"]
    S = horizontal_stabilizers["horiz_area"]
    
    x_arm = horizontal_stabilizers["x_arm"]

    Cl = Cl0 + Cl_alpha * np.radians(delR)
    L = 0.5 * rho * V_inf**2 * S * Cl
    pitch = L * x_arm

    return 2 * pitch
