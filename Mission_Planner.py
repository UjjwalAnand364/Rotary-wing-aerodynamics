import numpy as np
import matplotlib.pyplot as plt


from Solver import *
from helper_functions import *
from Simulator import *
from Params import *


def Mission_Planner_Enhanced_3(initial_fuel_weight, mission_segments, initial_payload_weight):
    """
    Enhanced mission planner that handles multiple flight segments in any order.
    
    Parameters:
    -----------
    initial_fuel_weight : float
        Initial fuel weight in kg
    mission_segments : list of dict
        List of mission segments, each dict contains segment-specific parameters:
        - {"type": "hover", "duration": float (minutes)}
        - {"type": "vertical_climb", "rate": float (m/s), "target_altitude": float (m)}
        - {"type": "forward_flight", "speed": float (m/s), "distance": float (m), "wind": float (m/s, optional)}
        - {"type": "climb_flight", "h_speed": float (m/s), "v_rate": float (m/s), "target_altitude": float (m), "wind": float (m/s, optional)}
        - {"type": "descent_flight", "h_speed": float (m/s), "v_rate": float (m/s), "target_altitude": float (m), "wind": float (m/s, optional)}
        - {"type": "payload_change", "weight_change": float (kg)}
        - {"type": "vertical_descent", "rate": float (m/s), "target_altitude": float (m)}
    initial_payload_weight : float
        Initial payload weight in kg
    """
    
    # Check fuel weight limit
    if initial_fuel_weight > fuselage["max_fuel_weight"]:
        print(f"Mission Failed: Fuel weight exceeds max of {fuselage['max_fuel_weight']} kg")
        return
    
    # Initialize mission state
    TOGW = initial_fuel_weight + fuselage["Empty_Weight"] + initial_payload_weight
    print(f"Takeoff Gross Weight: {TOGW:.2f} kg")
    print(f"Initial Fuel: {initial_fuel_weight:.2f} kg")
    print(f"Initial Payload: {initial_payload_weight:.2f} kg")
    print(f"Empty Weight: {fuselage['Empty_Weight']:.2f} kg")
    print("="*80)
    
    # Mission tracking variables
    times = []
    weights = []
    fuels = []
    fuel_rates = []
    power_required = []
    phases = []
    altitudes = []
    speeds = []
    climb_rates = []
    distances_covered = []
    segment_labels = []
    segment_times = []
    
    current_fuel = float(initial_fuel_weight)
    current_payload = float(initial_payload_weight)
    current_weight = TOGW
    current_altitude = flight_condition["altitude"]
    t = 0.0
    total_distance_m = 0.0
    step = 10  # 10 minute intervals
    
    # Process each mission segment
    for seg_idx, segment in enumerate(mission_segments):
        seg_type = segment["type"]
        print(f"\n{'='*80}")
        print(f"SEGMENT {seg_idx + 1}: {seg_type.upper()}")
        print(f"{'='*80}")
        
        # Handle payload change (instantaneous)
        if seg_type == "payload_change":
            weight_change = segment["weight_change"]
            current_payload += weight_change
            current_weight = fuselage["Empty_Weight"] + current_payload + current_fuel
            print(f"Payload changed by {weight_change:+.2f} kg")
            print(f"New payload: {current_payload:.2f} kg")
            print(f"New gross weight: {current_weight:.2f} kg")
            
            # Record the instantaneous change
            if len(times) > 0:
                times.append(t)
                weights.append(current_weight)
                fuels.append(current_fuel)
                fuel_rates.append(6)  # Hover fuel consumption during instantaneous change
                power_required.append(power_required[-1] if power_required else 0.0)
                phases.append(seg_type)
                altitudes.append(current_altitude)
                speeds.append(0)
                climb_rates.append(0.0)
                distances_covered.append(total_distance_m / 1000.0)
                segment_labels.append(f"Seg {seg_idx + 1}: {seg_type}")
                segment_times.append(t)
            continue
        
        # Determine segment parameters
        segment_duration = 0.0
        velocity_vector = [0.0, 0.0, 0.0]
        wind_vector = [0.0, 0.0, 0.0]
        target_altitude = current_altitude
        horizontal_distance = 0.0
        segment_label = f"Seg {seg_idx + 1}: {seg_type}"
        
        if seg_type == "hover":
            segment_duration = segment["duration"]
            velocity_vector = [0.0, 0.0, 0.0]
            
        elif seg_type == "vertical_climb":
            climb_rate = segment["rate"]
            target_altitude = segment["target_altitude"]
            altitude_change = target_altitude - current_altitude
            if altitude_change <= 0:
                print(f"Warning: Already at or above target altitude. Skipping segment.")
                continue
            segment_duration = altitude_change / climb_rate / 60.0  # convert to minutes
            velocity_vector = [0.0, 0.0, climb_rate]
            
        elif seg_type == "vertical_descent":
            descent_rate = segment["rate"]
            target_altitude = segment["target_altitude"]
            altitude_change = current_altitude - target_altitude
            if altitude_change <= 0:
                print(f"Warning: Already at or below target altitude. Skipping segment.")
                continue
            segment_duration = altitude_change / abs(descent_rate) / 60.0  # convert to minutes
            velocity_vector = [0.0, 0.0, descent_rate]
            
        elif seg_type == "forward_flight":
            speed = segment["speed"]
            distance = segment["distance"]
            wind = segment.get("wind", 0.0)
            segment_duration = distance / speed / 60.0  # convert to minutes
            velocity_vector = [speed, 0.0, 0.0]
            wind_vector = [wind, 0.0, 0.0]
            horizontal_distance = distance
            
        elif seg_type == "climb_flight":
            h_speed = segment["h_speed"]
            v_rate = segment["v_rate"]
            target_altitude = segment["target_altitude"]
            wind = segment.get("wind", 0.0)
            altitude_change = target_altitude - current_altitude
            if altitude_change <= 0:
                print(f"Warning: Already at or above target altitude. Skipping segment.")
                continue
            segment_duration = altitude_change / v_rate / 60.0  # convert to minutes
            velocity_vector = [h_speed, 0.0, v_rate]
            wind_vector = [wind, 0.0, 0.0]
            horizontal_distance = h_speed * segment_duration * 60.0
            
        elif seg_type == "descent_flight":
            h_speed = segment["h_speed"]
            v_rate = segment["v_rate"]
            target_altitude = segment["target_altitude"]
            wind = segment.get("wind", 0.0)
            altitude_change = current_altitude - target_altitude
            if altitude_change <= 0:
                print(f"Warning: Already at or below target altitude. Skipping segment.")
                continue
            segment_duration = altitude_change / abs(v_rate) / 60.0  # convert to minutes
            velocity_vector = [h_speed, 0.0, v_rate]
            wind_vector = [wind, 0.0, 0.0]
            horizontal_distance = h_speed * segment_duration * 60.0
        
        print(f"Segment Duration: {segment_duration:.2f} minutes")
        print(f"Velocity: {velocity_vector}")
        print(f"Wind: {wind_vector}")
        print(f"Starting Altitude: {current_altitude:.1f} m")
        if target_altitude != current_altitude:
            print(f"Target Altitude: {target_altitude:.1f} m")
        
        # Execute segment with time steps
        elapsed = 0.0
        while elapsed < segment_duration - 1e-9:
            interval = min(step, segment_duration - elapsed)
            
            # Calculate thrust requirement
            thrust_req = current_weight * 9.81
            
            # Update flight condition for this step
            flight_condition_now = flight_condition.copy()
            flight_condition_now["velocity"] = velocity_vector.copy()
            flight_condition_now["wind"] = wind_vector.copy()
            flight_condition_now["altitude"] = current_altitude
            flight_condition_now["fuel_weight"] = current_fuel
            
            # Update payload
            payload_now = payload.copy()
            payload_now["weight"] = current_payload
            
            # Update fuselage with current fuel weight
            fuselage_now = fuselage.copy()
            fuselage_now["fuel_weight"] = current_fuel
            
            # Calculate available power at current altitude
            corrected_engine_power = engine["max_power_avail"] * density_ratio(flight_condition_now["altitude"])
            
            # Call Sim_Start_Forward - it will internally find the required collective
            res, res_tail = Sim_Start_Forward(
                rotor=rotor,
                tail_rotor=tail_rotor,
                engine=engine,
                flight_condition=flight_condition_now,
                tol2=5e-3,
                t_horizon_s=60
            )
            
            # Get the collective pitch determined by Sim_Start_Forward
            collective_pitch = res.get("collective", 0.0)
            
            # Check for stall
            if res["stall_status"] == 1 or res_tail["stall_status"] == 1:
                print(f"\nMission Failed: Rotor stall at t={t:.2f} min in {seg_type}")
                
                # Plot even for failed missions
                if len(times) > 0:
                    plot_mission_results(times, weights, fuels, fuel_rates, power_required, 
                                       altitudes, speeds, climb_rates, distances_covered, 
                                       segment_labels, engine, mission_success=False)
                return
            
            # Check thrust adequacy
            if res["T"] < thrust_req - 200:   ## Softbound for actual thrust
                print(f"\nMission Failed: Insufficient thrust at t={t:.2f} min in {seg_type}")
                print(f"Required: {thrust_req:.2f} N, Available: {res['T']:.2f} N")
                print(f"Collective: {collective_pitch:.2f} deg")
                
                # Plot even for failed missions
                if len(times) > 0:
                    plot_mission_results(times, weights, fuels, fuel_rates, power_required, 
                                       altitudes, speeds, climb_rates, distances_covered, 
                                       segment_labels, engine, mission_success=False)
                return
            
            # Calculate total power
            total_power = res["P"] + res_tail["P"] / (1 - engine["engines_loss"])
            
            # Check power adequacy
            if total_power > corrected_engine_power:
                print(f"\nMission Failed: Insufficient power at t={t:.2f} min in {seg_type}")
                print(f"Required: {total_power:.2f} W, Available: {corrected_engine_power:.2f} W")
                
                # Plot even for failed missions
                if len(times) > 0:
                    plot_mission_results(times, weights, fuels, fuel_rates, power_required, 
                                       altitudes, speeds, climb_rates, distances_covered, 
                                       segment_labels, engine, mission_success=False)
                return
            
            # Calculate fuel consumption
            bsfc = engine["bsfc"]  # kg/kWh
            phase_fuel_rate = (bsfc * (total_power / 1000.0)) / 60.0  # kg/min
            fuel_needed = phase_fuel_rate * interval
            
            # Check fuel exhaustion
            if fuel_needed >= current_fuel - 1e-12:
                time_to_empty = current_fuel / phase_fuel_rate if phase_fuel_rate > 0 else 0
                t_exhaust = t + time_to_empty
                distance_covered = 0
                if seg_type in ["forward_flight", "climb_flight", "descent_flight"]:
                    distance_covered = velocity_vector[0] * 60.0 * time_to_empty
                    total_distance_m += distance_covered
                
                print(f"\n{'*'*80}")
                print(f"FUEL EXHAUSTED at t={t_exhaust:.2f} min")
                print(f"Phase: {seg_type}")
                print(f"Gross Weight: {current_weight:.2f} kg")
                print(f"Fuel Rate: {phase_fuel_rate:.4f} kg/min")
                print(f"Power Required: {total_power:.2f} W")
                print(f"Partial Interval: {time_to_empty:.2f} min")
                if distance_covered > 0:
                    print(f"Distance covered in partial interval: {distance_covered:.2f} m")
                print(f"Total Distance: {total_distance_m/1000.0:.2f} km ({total_distance_m/1852.0:.2f} NM)")
                print(f"{'*'*80}")
                
                # Plot even for failed missions
                if len(times) > 0:
                    plot_mission_results(times, weights, fuels, fuel_rates, power_required, 
                                       altitudes, speeds, climb_rates, distances_covered, 
                                       segment_labels, engine, mission_success=False)
                return
            
            # Consume fuel
            current_fuel -= fuel_needed
            current_fuel = max(current_fuel, 0.0)
            current_weight = fuselage["Empty_Weight"] + current_payload + current_fuel
            
            # Update distance for forward motion
            if seg_type in ["forward_flight", "climb_flight", "descent_flight"]:
                total_distance_m += velocity_vector[0] * 60.0 * interval
            
            # Calculate current speed and climb rate
            current_speed = np.sqrt(velocity_vector[0]**2 + velocity_vector[1]**2)  # m/s
            current_climb_rate = velocity_vector[2]  # m/s
            
            # Record data
            t_end = t + interval
            times.append(t_end)
            weights.append(current_weight)
            fuels.append(current_fuel)
            fuel_rates.append(phase_fuel_rate)
            power_required.append(total_power)
            phases.append(seg_type)
            altitudes.append(current_altitude)
            speeds.append(current_speed)
            climb_rates.append(current_climb_rate)
            distances_covered.append(total_distance_m / 1000.0)  # in km
            segment_labels.append(segment_label)
            segment_times.append(t_end)
            
            # Print progress
            print(f"\nt = {t_end:.2f} min | {seg_type}")
            print(f"  Weight: {current_weight:.2f} kg | Fuel: {current_fuel:.2f} kg")
            print(f"  Fuel Rate: {phase_fuel_rate:.4f} kg/min | Power: {total_power:.2f} W")
            print(f"  Collective: {collective_pitch:.2f} deg | Altitude: {current_altitude:.1f} m")
            
            t = t_end
            elapsed += interval
        
        # Update altitude at end of segment
        current_altitude = target_altitude
    
    # Mission complete
    total_distance_km = total_distance_m / 1000.0
    total_distance_nm = total_distance_m / 1852.0
    
    print(f"\n{'='*80}")
    print(f"MISSION SUCCESS ✅")
    print(f"{'='*80}")
    print(f"Initial TOGW: {TOGW:.2f} kg")
    print(f"Final Weight: {current_weight:.2f} kg")
    print(f"Total Fuel Used: {initial_fuel_weight - current_fuel:.2f} kg")
    print(f"Fuel Remaining: {current_fuel:.2f} kg")
    print(f"Total Mission Time: {t:.2f} minutes ({t/60.0:.2f} hours)")
    print(f"Total Distance: {total_distance_km:.2f} km ({total_distance_nm:.2f} NM)")
    print(f"Final Altitude: {current_altitude:.1f} m")
    
    # Plotting
    plot_mission_results(times, weights, fuels, fuel_rates, power_required, 
                        altitudes, speeds, climb_rates, distances_covered, 
                        segment_labels, engine, mission_success=True)
    
    return {
        "success": True,
        "fuel_remaining": current_fuel,
        "total_time": t,
        "total_distance_km": total_distance_km,
        "total_distance_nm": total_distance_nm,
        "final_weight": current_weight,
        "final_altitude": current_altitude
    }


def plot_mission_results(times, weights, fuels, fuel_rates, power_required, 
                        altitudes, speeds, climb_rates, distances_covered, 
                        segment_labels, engine, mission_success=True):
    """
    Plot mission results with segment-wise labels
    """
    if len(times) == 0:
        print("No data to plot")
        return
    
    # Create figure with 4x2 subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Add mission status to title
    status = "MISSION SUCCESS ✅" if mission_success else "MISSION FAILED ❌"
    fig.suptitle(status, fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Gross Weight vs Time
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(times, weights, linewidth=2, color='blue')
    ax1.set_xlabel("Time (min)", fontsize=10)
    ax1.set_ylabel("Gross Weight (kg)", fontsize=10)
    ax1.set_title("Gross Weight vs Time", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle="--", alpha=0.7)
    
    # 2. Fuel Remaining vs Time
    ax2 = plt.subplot(4, 2, 2)
    ax2.plot(times, fuels, linewidth=2, color='orange')
    ax2.set_xlabel("Time (min)", fontsize=10)
    ax2.set_ylabel("Fuel Weight (kg)", fontsize=10)
    ax2.set_title("Fuel Remaining vs Time", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.7)
    
    # 3. Fuel Burn Rate vs Time
    ax3 = plt.subplot(4, 2, 3)
    ax3.plot(times, fuel_rates, linewidth=2, color='green')
    ax3.set_xlabel("Time (min)", fontsize=10)
    ax3.set_ylabel("Fuel Burn Rate (kg/min)", fontsize=10)
    ax3.set_title("Fuel Burn Rate vs Time", fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle="--", alpha=0.7)
    
    # 4. Altitude vs Time
    ax4 = plt.subplot(4, 2, 4)
    ax4.plot(times, altitudes, linewidth=2, color='purple')
    ax4.set_xlabel("Time (min)", fontsize=10)
    ax4.set_ylabel("Altitude (m AMSL)", fontsize=10)
    ax4.set_title("Altitude vs Time", fontsize=12, fontweight='bold')
    ax4.grid(True, linestyle="--", alpha=0.7)
    
    # 5. Power Required vs Available
    ax5 = plt.subplot(4, 2, 5)
    power_required_kw = [p/1000 for p in power_required]
    ax5.plot(times, power_required_kw, linewidth=2, color='red', label='Required')
    available_power_kw = [engine["max_power_avail"] * density_ratio(alt) / 1000 for alt in altitudes]
    ax5.plot(times, available_power_kw, linewidth=2, color='darkred', linestyle='--', label='Available', alpha=0.8)
    ax5.set_xlabel("Time (min)", fontsize=10)
    ax5.set_ylabel("Power (kW)", fontsize=10)
    ax5.set_title("Power Required vs Available", fontsize=12, fontweight='bold')
    ax5.grid(True, linestyle="--", alpha=0.7)
    ax5.legend(fontsize=9, loc='best')
    
    # 6. Horizontal Speed vs Time
    ax6 = plt.subplot(4, 2, 6)
    speeds_kmh = [s * 3.6 for s in speeds]  # Convert m/s to km/h
    ax6.plot(times, speeds_kmh, linewidth=2, color='cyan')
    ax6.set_xlabel("Time (min)", fontsize=10)
    ax6.set_ylabel("Speed (km/h)", fontsize=10)
    ax6.set_title("Horizontal Speed vs Time", fontsize=12, fontweight='bold')
    ax6.grid(True, linestyle="--", alpha=0.7)
    
    # 7. Climb Rate vs Time
    ax7 = plt.subplot(4, 2, 7)
    climb_rates_mpm = [c * 60 for c in climb_rates]  # Convert m/s to m/min
    ax7.plot(times, climb_rates_mpm, linewidth=2, color='brown')
    ax7.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    ax7.set_xlabel("Time (min)", fontsize=10)
    ax7.set_ylabel("Climb Rate (m/min)", fontsize=10)
    ax7.set_title("Climb Rate vs Time", fontsize=12, fontweight='bold')
    ax7.grid(True, linestyle="--", alpha=0.7)
    
    # 8. Cumulative Distance vs Time
    ax8 = plt.subplot(4, 2, 8)
    ax8.plot(times, distances_covered, linewidth=2, color='magenta')
    ax8.set_xlabel("Time (min)", fontsize=10)
    ax8.set_ylabel("Distance Covered (km)", fontsize=10)
    ax8.set_title("Cumulative Distance vs Time", fontsize=12, fontweight='bold')
    ax8.grid(True, linestyle="--", alpha=0.7)
    
# ----- Segment-wise vertical markers, colored regions, and labels -----
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
    if len(segment_labels) == len(times):
        segment_colors = plt.cm.tab10(np.linspace(0, 1, len(set(segment_labels))))
        color_map = {label: segment_colors[i % len(segment_colors)] for i, label in enumerate(sorted(set(segment_labels)))}

        for ax in axes:
            prev_idx = 0
            for i in range(1, len(segment_labels)):
                if segment_labels[i] != segment_labels[i - 1] or i == len(segment_labels) - 1:
                    start_t = times[prev_idx]
                    end_t = times[i]
                    seg_label = segment_labels[prev_idx]
                    ax.axvspan(start_t, end_t, color=color_map[seg_label], alpha=0.08)
                    ax.axvline(x=start_t, color=color_map[seg_label], linestyle='--', linewidth=1)
                    ax.text(start_t, ax.get_ylim()[1]*0.95, seg_label,
                            rotation=90, verticalalignment='top', fontsize=8, color=color_map[seg_label])
                    prev_idx = i

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()



# Example usage:
"""
mission = [
    {"type": "hover", "duration": 5},
    {"type": "vertical_climb", "rate": 5, "target_altitude": 3000},
    {"type": "forward_flight", "speed": 55.56, "distance": 50000, "wind": 10},
    {"type": "climb_flight", "h_speed": 41.67, "v_rate": 3, "target_altitude": 4000, "wind": -5},
    {"type": "payload_change", "weight_change": -200},
    {"type": "descent_flight", "h_speed": 41.67, "v_rate": -3, "target_altitude": 2000, "wind": 0},
    {"type": "vertical_descent", "rate": -5, "target_altitude": 0},
]

result = Mission_Planner_Enhanced(
    initial_fuel_weight=800,
    mission_segments=mission,
    initial_payload_weight=700
)
"""