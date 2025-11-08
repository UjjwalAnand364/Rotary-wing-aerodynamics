import numpy as np
from Tail_rotor import Simulate_tail_rotor
from Trim_conditions import trim_cyclic
# from Balance_thrust import balance_thrust
from Params import engine


def balance_main_and_tail(coll, theta_1c, theta_1s,
                        TOGW, Omega, alpha_tpp, rotor, tail_rotor, flight_condition, 
                        b, rho, B_fn, t_horizon_s, tol2, delE, delR, max_iter):
    
    theta_c_init=theta_1c
    theta_s_init=theta_1s
    coll_init=coll
    V_inf=flight_condition["velocity"][0]
    V_climb=flight_condition['velocity'][2]


        
    # Find cyclic values to trim
    res=trim_cyclic(rotor, "angle",
        theta_c_init, theta_s_init, coll_init, delE, delR,
        TOGW, alpha_tpp, V_inf, V_climb, b, rho, B_fn, t_horizon_s
    )
    print("Cyclic_c",res["cyclic_c"],"Cyclic_s:",res["cyclic_s"])

    theta_c_init, theta_s_init = res["cyclic_c"], res["cyclic_s"]
    # cyclic_s_prev=res["cyclic_s"]
    coll_init=res["collective"]
    
    # Find tail collective to balance torque
    tail_coll_init = tail_rotor["collective"]
    res_tail = Simulate_tail_rotor(tail_rotor,engine,flight_condition,tail_coll_init,res["Q"],tol2,delE)
   

    return res, res_tail
    
