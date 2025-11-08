from Params import rotor_aero,fuselage,payload
from Solver import *
from helper_functions import *
from calc_functions import *
from Control_surfaces import yaw_vertical_fin, elevator_pitch
import numpy as np
from typing import Callable, Dict
from scipy.optimize import least_squares


def stall_check(R_root,R_tip,theta_fn,phi_fn):
    r_all = np.linspace(R_root, R_tip, 80)       # dense discretization in radius
    sigh_all = np.linspace(0, 2*np.pi, 90)       # dense discretization in azimuth

    for r in r_all:
        for sigh_val in sigh_all:
            alpha = theta_fn(r,sigh_val) - phi_fn(r, sigh_val)
            if alpha > rotor_aero["alpha_stall"]:
                print(f"Stall detected for main rotor at r = {r:.3f} m, sigh = {sigh_val:.3f} rad, alpha = {alpha:.3f} deg")
                return {"stall_status": 1, "r": r, "sigh": sigh_val, "alpha": alpha, "out_of_power": False}
    
    return {"stall_status": 0}


def find_moments(rotor, cyclic_c, cyclic_s, coll, R_root, R_tip, alpha_tpp, I, Omega, b, rho, V_inf, B_dot, phi_fn, v_fn, c_fn, B_fn):

    theta_fn = lambda r,sigh: pitch_x_forward(rotor, r,sigh, cyclic_c, cyclic_s, coll) 
    Cl_fn = lambda r,sigh: airfoil_lift(
        Cl0=rotor_aero["Cl0"],
        Cl_alpha=rotor_aero["Cl_alpha"],
        alpha0=rotor_aero["alpha0"],
        alpha = theta_fn(r,sigh) - phi_fn(r,sigh),
        alpha_stall=rotor_aero["alpha_stall"]
    )

    stall=stall_check(R_root,R_tip,theta_fn,phi_fn)
    if stall["stall_status"]==1:
        return stall
    # B_fn,B0_final=beta_fn(rho, alpha_tpp, Omega, V_inf, v_fn, c_fn, Cl_fn, R_root, R_tip, I)

    res = iterative_solver_cyclic(
        b=b, rho=rho,  
        Ut_fn=lambda r,sigh: Omega * r + V_inf*np.cos(alpha_tpp)*np.sin(sigh),
        Up_fn=lambda r,sigh: V_inf*np.sin(alpha_tpp) + v_fn(r,sigh) + r*B_dot + V_inf*np.sin(B_fn(sigh))*np.cos(sigh),
        v_fn=v_fn, c_fn=c_fn, Cl_fn=Cl_fn, 
        R_root=R_root, R_tip=R_tip,
        N0 = 25, max_iter = 4, tol=1e-3
    )
    return res


def find_cyclic_bounds(
    Mp_init, Mr_init, pitch_hfin, roll_hfin,
    theta_c_init, theta_s_init, coll_init,
    alpha_tpp, I, Omega, V_inf, B_dot,
    b, rho, R_root, R_tip, TOGW,
    v_fn, c_fn, phi_fn, B_fn
    ):
    #print("Mp_init:",Mp_init,"Mr_init:",Mr_init)
    #print(theta_c_init,theta_s_init)

    # Upper/Lower bound for pitch
    cyclic_c = 6.5+theta_c_init if Mp_init > 0 else -6.5+theta_c_init
    cyclic_s=theta_s_init
    
    out = find_moments(rotor, cyclic_c, cyclic_s, coll_init, R_root, R_tip, alpha_tpp, I, Omega, b, rho, V_inf, B_dot, phi_fn, v_fn, c_fn, B_fn)
    #print("out Mp:",out["Mp"])
    if (out["Mp"] - pitch_hfin + fuselage["d_from_shaft"] * TOGW * 9.8) * Mp_init > 0:
        cyclic_c += (np.sign(Mp_init) * 2.5)


    # Upper/Lower bound for Roll
    cyclic_s_2 = -6.5+theta_s_init if Mr_init >= 0 else 6.5+theta_s_init
    cyclic_c_2 = theta_c_init

    out = find_moments(rotor, cyclic_c_2, cyclic_s_2, coll_init, R_root, R_tip, alpha_tpp, I, Omega, b, rho, V_inf, B_dot, phi_fn, v_fn, c_fn, B_fn)
    #print("out Mr:",out["Mr"])
    if (out["Mr"] - roll_hfin) * Mr_init > 0:
        cyclic_s_2 -= (np.sign(Mr_init) * 2.5)

    return cyclic_c,cyclic_s_2


def find_cyclic(
    bound_c, bound_s, bound_0,
    theta_c_init, theta_s_init, coll_init,
    Mp_init, Mr_init, T_init,
    pitch_hfin, roll_vfin,
    alpha_tpp, I, Omega, V_inf, B_dot,
    b, rho, R_root, R_tip, TOGW,
    v_fn, c_fn, phi_fn, B_fn
    ):
    """
    Efficient 2D trim using Nelder-Mead and caching.

    simulate_forward_flight: function(theta1c, theta1s) -> {"Mr":..., "Mp":...}
    theta_init: starting guess (rad, rad)
    """
        
    # cyclic_c bounds
    if Mp_init < 0:
        lower_c, upper_c = bound_c, theta_c_init+0.5
    else:
        lower_c, upper_c = theta_c_init-0.5, bound_c
    # cyclic_s bounds
    if Mr_init < 0:
        lower_s, upper_s = theta_s_init-0.5, bound_s
    else:
        lower_s, upper_s = bound_s, theta_s_init+0.5
    # collective bounds
    if T_init < TOGW*9.8:
        lower_0, upper_0 = coll_init-5, bound_0
    else:
        lower_0, upper_0 = bound_0, coll_init+5

    def equations(vars):
        c, s, coll = vars
        out = find_moments(rotor, c, s, coll, R_root, R_tip, alpha_tpp, I, Omega, b, rho, V_inf, B_dot, phi_fn, v_fn, c_fn, B_fn)
    
        Mp = (out["Mp"] - pitch_hfin + fuselage["d_from_shaft"] * TOGW * 9.8) 
        Mr = (out["Mr"] - roll_vfin) 
        T = (out["T"]-TOGW*9.8) 

        return [Mp, Mr, T]
    
    # Bounds 
    lower_bounds = [lower_c, lower_s, lower_0]
    upper_bounds = [upper_c, upper_s, upper_0]
    # print(lower_c,upper_c)
    # print(lower_s,upper_s)
    # print(lower_0,upper_0)
    # Initial guess (can be anything inside bounds)
    initial_guess = [theta_c_init, theta_s_init, coll_init]

    result = least_squares(equations, initial_guess, bounds=(lower_bounds, upper_bounds), x_scale='jac')
    cyclic_c_trim, cyclic_s_trim, coll_trim = result.x

    return cyclic_c_trim, cyclic_s_trim, coll_trim

    
def trim_cyclic(rotor, tol_mode,
        theta_c_init, theta_s_init, coll_init, delE, delR, 
        TOGW, alpha_tpp, V_inf,V_climb, b, rho, B_fn, t_horizon_s
    ):

    I_values=compute_MoI(fuselage,payload)
    I_roll=I_values["I_x"]
    I_pitch=I_values["I_y"]
    theta_max_deg=5.0
    omega_max=1

    # compute allowed moments
    if tol_mode == "angle":
        theta_max = np.deg2rad(theta_max_deg)
        M_roll_allow  = 2 * I_roll  * theta_max / (t_horizon_s ** 2)
        M_pitch_allow = 2 * I_pitch * theta_max / (t_horizon_s ** 2)
    elif tol_mode == "rate":
        omega_max = np.deg2rad(omega_max)
        M_roll_allow  = I_roll  * omega_max
        M_pitch_allow = I_pitch * omega_max
    else:
        raise ValueError("tol_mode must be 'angle' or 'rate'")
    M_roll_allow = abs(M_roll_allow)
    M_pitch_allow = abs(M_pitch_allow)
    #print("Mp_allow:",M_pitch_allow,"Mr_allow:",M_roll_allow)

    R_tip  = rotor["Rt"]
    R_root = rotor["Rr"]
    C_tip = rotor["chord_tip"]
    C_root = rotor["chord_root"]

    Omega = engine["omega"]
    a = rotor_aero["Cl_alpha"]

    B1c=alpha_tpp
    B_dot=B1c*Omega
    # B0_old=0

    mu =  V_inf*np.cos(alpha_tpp)/ (Omega * R_tip)  # advance ratio
    I = rho*a*(C_tip + C_root)*R_root**4/(2*rotor["lock_number"] ) # Lock number

    Lambda_induced_forward = lambda r,sigh: lambda_i_forward(mu,r,R_tip,sigh,alpha_tpp,Omega,rho,V_inf,V_climb)
    # B_fn = lambda sigh: B0_old + B1c*np.cos(sigh)

    v_fn = lambda r,sigh: induced_velocity_forward(Lambda_induced_forward(r,sigh),Omega, R_tip, V_inf, alpha_tpp)
    phi_fn = lambda r,sigh: compute_phi_forward(V_inf,v_fn(r,sigh),Omega,alpha_tpp,sigh,r,B_fn(sigh),B_dot,R_root)

    c_fn = lambda r: chord_r(rotor,r)
    AR = (R_tip - R_root) / ((rotor["chord_root"] + rotor["chord_tip"]) / 2)
    
    out = find_moments(rotor, theta_c_init, theta_s_init, coll_init, R_root, R_tip, alpha_tpp, I, Omega, b, rho, V_inf, B_dot, phi_fn, v_fn, c_fn,B_fn)


    _,roll_vfin = yaw_vertical_fin(delE, rho, V_inf)
    pitch_hfin = elevator_pitch(delR, rho, V_inf)

    #print(out["Mr"],out["Mp"],roll_vfin,pitch_hfin)
    Mr_init = out["Mr"] - roll_vfin
    Mp_init = out["Mp"] - pitch_hfin + fuselage["d_from_shaft"] * TOGW * 9.8


    bound_c, bound_s=find_cyclic_bounds(
        Mp_init, Mr_init, pitch_hfin, roll_vfin,
        theta_c_init, theta_s_init, coll_init,
        alpha_tpp, I, Omega, V_inf, B_dot,
        b, rho, R_root, R_tip, TOGW, 
        v_fn, c_fn, phi_fn, B_fn
    )
    bound_0 = coll_init
    if out["T"] > TOGW*9.8:
        bound_0 -= 5
    else:
        bound_0 += 5

    cyclic_c_trim, cyclic_s_trim, coll_trim = find_cyclic(
            bound_c, bound_s, bound_0,
            theta_c_init, theta_s_init, coll_init,
            Mp_init, Mr_init, out["T"],
            pitch_hfin, roll_vfin,
            alpha_tpp, I, Omega, V_inf, B_dot,
            b, rho, R_root, R_tip, TOGW,
            v_fn, c_fn, phi_fn, B_fn
    )

    Cl_fn = lambda r, sigh: airfoil_lift(
        Cl0=rotor_aero["Cl0"],
        Cl_alpha=rotor_aero["Cl_alpha"],
        alpha0=rotor_aero["alpha0"],
        alpha=pitch_x_forward(rotor, r, sigh, cyclic_c_trim, cyclic_s_trim, coll_trim) - phi_fn(r, sigh),
        alpha_stall=rotor_aero["alpha_stall"]
    )
    # B_fn,B0_final=beta_fn(rho, alpha_tpp, Omega, V_inf, v_fn, c_fn, Cl_fn, R_root, R_tip, I)
    Cd_fn = lambda r,sigh: airfoil_drag(
        Cd0=rotor_aero["Cd0"],
        Cl=Cl_fn(r,sigh),
        e=rotor_aero["e"],
        AR=AR
    )
    res=iterative_solver_forward(
        b=b, rho=rho,
        Ut_fn=lambda r,sigh: Omega * r + V_inf*np.cos(alpha_tpp)*np.sin(sigh),
        Up_fn=lambda r,sigh: V_inf*np.sin(alpha_tpp) + v_fn(r,sigh) + r*B_dot + V_inf*np.sin(B_fn(sigh))*np.cos(sigh),
        c_fn=c_fn, Cl_fn=Cl_fn, v_fn=v_fn,
        phi_fn=phi_fn, Cd_fn=Cd_fn,
        R_root=R_root, R_tip=R_tip,
        N0 = 25, max_iter = 4, tol=1e-3
    )

    res["cyclic_c"]   = -cyclic_c_trim
    res["cyclic_s"]   = cyclic_s_trim
    res["collective"] = coll_trim
    res["Mp"] += fuselage["d_from_shaft"] * TOGW * 9.8

    return res


   