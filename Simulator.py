import numpy as np
import pandas as pd
from Solver import *
from helper_functions import *
from Params import *
from Balance_main_and_tail import balance_main_and_tail
from calc_functions import *

''' 
def Sim_Start_Hover_Climb(rotor, tail_rotor, engine, flight_condition):
    
    b = rotor["b"]
    altitude = flight_condition["altitude"]
    rho = atmosphere(altitude, delta_ISA=flight_condition["delta_ISA"])["rho"]
    R_tip  = rotor["Rt"]
    R_root = rotor["Rr"]
    C_tip = rotor["chord_tip"]
    C_root = rotor["chord_root"]
    Omega = engine["omega"]


    sigma = solidity(b, C_root,C_tip, R_tip)
    a = rotor_aero["Cl_alpha"]

    V_val = flight_condition["velocity"][2] # vertical velocity component (w)
    lambda_c =  V_val/ (Omega * R_tip)  # advance ratio

    # Pitch angle as a function of r
    theta_fn = lambda r: pitch_x(rotor, r)

    # Inflow ratio lambda as a function of r, with tip loss factor
    Lambda_tiploss = lambda r: solve_lambda_tiploss(flight_condition, sigma, a, b, theta_fn(r), r, lambda_c, R_tip, R_root)
    lembda_fn = lambda r: Lambda_tiploss(r)[0]
    F_fn = lambda r: Lambda_tiploss(r)[1]
    
    #[print(Lambda_tiploss(r)) for r in np.linspace(R_root, R_tip, 5)] # debug print
    
    # Induced velocity as a function of r
    v_fn = lambda r: induced_velocity(lembda_fn(r), Omega, R_tip, V_val)
 
    # Inflow angle phi as a function of r
    phi_fn = lambda r: compute_phi(V_val, v_fn(r), Omega, r, R_root)

    # Chord as a function of r
    c_fn = lambda r: chord_r(rotor,r)

    

    # Lift coefficient as a function of r
    Cl_fn = lambda r: airfoil_lift(
        Cl0=rotor_aero["Cl0"],
        Cl_alpha=rotor_aero["Cl_alpha"],
        alpha0=rotor_aero["alpha0"],
        alpha = theta_fn(r) - phi_fn(r),
        alpha_stall=rotor_aero["alpha_stall"]

    )

    # Aspect ratio estimate for drag calculation?
    AR = (R_tip - R_root) / ((C_root + C_tip) / 2)

    # Drag coefficient as a function of r
    Cd_fn = lambda r: airfoil_drag(
        Cd0=rotor_aero["Cd0"],
        Cl=Cl_fn(r),
        e=rotor_aero["e"],
        AR=AR
    )


    # Check stall across all radii (not just samples)
    r_all = np.linspace(R_root, R_tip, 200)   # dense discretization
    for r in r_all:
        alpha = theta_fn(r) - phi_fn(r)
        if alpha > rotor_aero["alpha_stall"]:
            print(f"Stall detected at r = {r:.3f} m, alpha = {alpha:.3f} deg")
            return ({"stall_status": 1, "r": r, "alpha": alpha},)
        
        
    # Now run the iterative thrust calculation
    res = iterative_solver_hover_climb(
        b=b, rho=rho,
        Ut_fn=lambda r: Omega * r,
        Up_fn=lambda r: V_val + v_fn(r),
        c_fn=c_fn, Cl_fn=Cl_fn, phi_fn=phi_fn, Cd_fn=Cd_fn,
        R_root=R_root, R_tip=R_tip,
        N0 = 200, max_iter= 100, tol=1e-3
    )

    # Print scalar variables
    print("b =", b)
    print("rho =", rho)
    print("altitude =", flight_condition["altitude"])
    print("R_tip =", R_tip)
    print("R_root =", R_root)
    print("Omega =", Omega)
    print("sigma =", sigma)
    print("a =", a)
    print("V_val =", V_val)
    print("AR =", AR)


    print("\nConverged T (Thrust) =", res["T"], "N")
    print("Converged D (Rotor Drag)   =", res["D"], "N")
    print("Converged Q (Torque) =", res["Q"], "Nm")
    print("Converged P (Power)  =", res["P"], "W")
        
        
    return (res,)

'''

"""
-------------Forward Simulation Start----------------
"""


def Sim_Start_Forward(rotor, tail_rotor, engine, flight_condition, tol2, t_horizon_s, delE =0, delR= 0):

    b = rotor["b"]
    altitude = flight_condition["altitude"]
    rho = atmosphere(altitude, delta_ISA=flight_condition["delta_ISA"])["rho"]
    R_tip  = rotor["Rt"]
    R_root = rotor["Rr"]
    C_tip = rotor["chord_tip"]
    C_root = rotor["chord_root"]
    Omega = engine["omega"]
    TOGW = fuselage["Empty_Weight"] + payload["weight"] + flight_condition["fuel_weight"]

    # Cyclic values
    coll=rotor["collective"]
    theta_1c=rotor["cyclic_c"]
    theta_1s=rotor["cyclic_s"]
    
    

    #sigma = solidity(b, C_root,C_tip, R_tip)
    a = rotor_aero["Cl_alpha"]

    V_inf = flight_condition["velocity"][0] + flight_condition['wind'][0] # horizontal velocity component (w) 
    V_climb = flight_condition['velocity'][2] ## z in vertical climb

    drag = fuselage_drag(fuselage, rho, flight_condition["velocity"])

    alpha_tpp = alphaTPP(drag,TOGW)    #in radians
    B1c = alpha_tpp 
    print("alpha_tpp (rad):", alpha_tpp)

    def B_dot(sigh):
        return -B1c*np.sin(sigh)*Omega
  
    
    B0_old = 0. ## Assuming no coning for now

    mu =  V_inf*np.cos(alpha_tpp)/ (Omega * R_tip)  # advance ratio

    I = rho*a*(C_tip + C_root)*R_root**4/(2*rotor["lock_number"] ) # Lock number

    # Pitch angle as a function of r
    theta_fn = lambda r,sigh: pitch_x_forward(rotor, r, sigh, theta_1c,theta_1s,coll,) 
                                                
    # Inflow ratio lambda as a function of r, with tip loss factor
    Lambda_induced_forward = lambda r, sigh: lambda_i_forward(mu, r, R_tip, sigh, alpha_tpp, Omega, rho, V_inf,V_climb)

    
    #[print(Lambda_tiploss(r)) for r in np.linspace(R_root, R_tip, 5)] # debug print
    
    # Induced velocity as a function of r
    v_fn = lambda r,sigh: induced_velocity_forward(Lambda_induced_forward(r,sigh), Omega, R_tip, V_inf,alpha_tpp)
    B_fn = lambda sigh: B0_old + B1c*np.cos(sigh)
 
    # Inflow angle phi as a function of r
    phi_fn = lambda r,sigh: compute_phi_forward(V_inf, v_fn(r,sigh), Omega, alpha_tpp,sigh,r,B_fn(sigh),B_dot(sigh),R_root)

    # Chord as a function of r
    c_fn = lambda r: chord_r(rotor,r)

    # Lift coefficient as a function of r
    Cl_fn = lambda r,sigh: airfoil_lift(
        Cl0=rotor_aero["Cl0"],
        Cl_alpha=rotor_aero["Cl_alpha"],
        alpha0=rotor_aero["alpha0"],
        alpha = theta_fn(r,sigh) - phi_fn(r,sigh),
        alpha_stall=rotor_aero["alpha_stall"]
    )

    # Aspect ratio estimate for drag calculation?
    AR = (R_tip - R_root) / ((C_root + C_tip) / 2)

    # Drag coefficient as a function of r
    Cd_fn = lambda r,sigh: airfoil_drag(
        Cd0=rotor_aero["Cd0"],
        Cl=Cl_fn(r,sigh),
        e=rotor_aero["e"],
        AR=AR
    )
    
    # Check stall across all radii and all azimuth angles (sigh)
    r_all = np.linspace(R_root, R_tip, 200)       # dense discretization in radius
    sigh_all = np.linspace(0, 2*np.pi, 360)       # dense discretization in azimuth

    for r in r_all:
        for sigh_val in sigh_all:
            alpha = theta_fn(r, sigh_val) - phi_fn(r, sigh_val)
            if alpha > rotor_aero["alpha_stall"]:
                print(f"Stall detected in tail rotor at r = {r:.3f} m, sigh = {sigh_val:.3f} rad, alpha = {alpha:.3f} deg")
                return {"stall_status": 1, "r": r, "sigh": sigh_val, "alpha": alpha}

    B_fn,B0_final=beta_fn(rho, alpha_tpp, Omega, V_inf, v_fn, c_fn, Cl_fn, R_root, R_tip, I)
    print("Converged B0 (coning angle):", B0_final)
        
    res,res_tail=balance_main_and_tail( 
        coll, theta_1c, theta_1s,
        TOGW, Omega, alpha_tpp, rotor, tail_rotor, flight_condition,
        b, rho, B_fn, t_horizon_s, tol2, delE, delR, max_iter = 10,
    )
    
    if res["stall_status"]==1:
        print("Mission not possible due to insufficient thrust.")
        return
    if res_tail["stall_status"]==1:
        print("Mission not possible due to tail rotor stall.")
        return
    


    # Tabulate sample values at selected radial and azimuth positions
    import itertools
    sample_r = np.linspace(R_root, R_tip, 10)
    sample_sigh = np.linspace(0, 2*np.pi, 10)
    table = []

    for r, sigh in itertools.product(sample_r, sample_sigh):
        table.append({
            "r (m)": round(r, 3),
            "theta (deg)": round(theta_fn(r, sigh), 3),
            "lambda": round(Lambda_induced_forward(r, sigh), 5),
            "v_induced (m/s)": round(v_fn(r, sigh), 4),
            "phi (deg)": round(phi_fn(r, sigh), 3),
            "chord (m)": round(c_fn(r), 4),
            "Cl": round(Cl_fn(r, sigh), 4),
            "Cd": round(Cd_fn(r, sigh), 5)
        })

    # --- Print converged scalar outputs (including new items) ---
    print("\nMain Rotor Results:")
    print(f"Tip Path Plane Angle = {alpha_tpp} rad")
    print(f"T (Thrust)        = {res['T']:.4f} N")
    print(f"L (Lift)          = {res['T']* np.cos(alpha_tpp)} N")
    print(f"D (Fuselgae)      = {res['T']* np.sin(alpha_tpp)} N")
    print(f"D (Rotor)         = {res['D']:.4f} N")
    print(f"Q (Torque)        = {res['Q']:.4f} Nm")
    print(f"P (Power)         = {res['P']:.4f} W")
    print(f"Rolling moment    = {res['Mr']:.4f} Nm")
    print(f"Pitching moment   = {res['Mp']:.4f} Nm")

    print(f"theta_c:          = {res["cyclic_c"]:.2f} deg")
    print(f"theta_s:          = {res["cyclic_s"]:.2f} deg")
    print(f"main_collective   = {res["collective"]:.2f} deg")
   


    
    # --- Print converged scalar outputs (including new items) ---
    print("\nConverged results for tail rotor:")
    print(f"T (Thrust)        = {res_tail['T']:.4f} N")
    print(f"L (Lift)          = {res_tail['T']* np.cos(alpha_tpp)} N")
    print(f"D (Drag)          = {res_tail['D']:.4f} N")
    print(f"Q (Torque)        = {res_tail['Q']:.4f} Nm")
    print(f"P (Power)         = {res_tail['P']:.4f} W")
    print(f"Rolling moment    = {res_tail['Mr']:.4f} Nm")
    print(f"Pitching moment   = {res_tail['Mp']:.4f} Nm")
    print(f"tail_collective    = {res_tail["tail_collective"]:.2f} deg")

    # --- Warning if not converged ---
    if "warning" in res:
        print("\nWarning:", res["warning"])



    return (res,res_tail)