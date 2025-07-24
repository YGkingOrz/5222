
from lib_piglet.utils.tools import eprint
from typing import List, Tuple
import glob, os, sys,time,json
import heapq
from collections import defaultdict

#import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.envs.agent_utils import EnvAgent
    from flatland.utils.controller import get_action, Train_Actions, Directions, check_conflict, path_controller, evaluator, remote_evaluator
except Exception as e:
    eprint("Cannot load flatland modules!")
    eprint(e)
    exit(1)

#########################
# Debugger and visualizer options
#########################

# Set these debug option to True if you want more information printed
debug = False
visualizer = False

# If you want to test on specific instance, turn test_single_instance to True and specify the level and test number
test_single_instance = False
level = 0
test = 0

#########################
# Global variables for maintaining state across function calls
#########################
global_reservation_table = {}
global_agent_priorities = {}

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_neighbors(pos, direction, rail):
    """Get valid neighboring positions from current position and direction"""
    neighbors = []
    valid_transitions = rail.get_transitions(pos[0], pos[1], direction)
    
    for i, valid in enumerate(valid_transitions):
        if valid:
            new_x, new_y = pos[0], pos[1]
            if i == Directions.NORTH:
                new_x -= 1
            elif i == Directions.EAST:
                new_y += 1
            elif i == Directions.SOUTH:
                new_x += 1
            elif i == Directions.WEST:
                new_y -= 1
            
            if 0 <= new_x < rail.height and 0 <= new_y < rail.width:
                neighbors.append(((new_x, new_y), i))
    
    return neighbors

def a_star_with_reservations(start_pos, start_dir, target_pos, rail, agent_id, max_timestep, reservation_table, start_time=0):
    """A* search with spatio-temporal reservations"""
    # Priority queue: (f_score, g_score, timestep, position, direction)
    open_set = [(manhattan_distance(start_pos, target_pos), 0, start_time, start_pos, start_dir, [])]
    closed_set = set()
    
    while open_set:
        f_score, g_score, timestep, pos, direction, path = heapq.heappop(open_set)
        
        # Check if we've reached the goal
        if pos == target_pos:
            return path + [pos]
        
        # Skip if already processed this state
        state = (timestep, pos, direction)
        if state in closed_set:
            continue
        closed_set.add(state)
        
        # Check time limit
        if timestep >= max_timestep:
            continue
        
        # Try staying in place (wait action)
        next_timestep = timestep + 1
        if not is_reserved(pos, next_timestep, agent_id, reservation_table):
            new_path = path + [pos]
            h_score = manhattan_distance(pos, target_pos)
            new_f = g_score + 1 + h_score
            heapq.heappush(open_set, (new_f, g_score + 1, next_timestep, pos, direction, new_path))
        
        # Try moving to neighboring positions
        for (new_pos, new_direction) in get_neighbors(pos, direction, rail):
            if not is_reserved(new_pos, next_timestep, agent_id, reservation_table):
                # Check for edge conflicts (swapping)
                if not is_edge_conflict(pos, new_pos, next_timestep, agent_id, reservation_table):
                    new_path = path + [pos]
                    h_score = manhattan_distance(new_pos, target_pos)
                    new_f = g_score + 1 + h_score
                    heapq.heappush(open_set, (new_f, g_score + 1, next_timestep, new_pos, new_direction, new_path))
    
    return []  # No path found

def is_reserved(pos, timestep, agent_id, reservation_table):
    """Check if a position is reserved at a given timestep"""
    if (pos, timestep) in reservation_table:
        return reservation_table[(pos, timestep)] != agent_id
    return False

def is_edge_conflict(pos1, pos2, timestep, agent_id, reservation_table):
    """Check for edge conflicts (two agents swapping positions)"""
    # Check if another agent is moving from pos2 to pos1 at the same time
    if (pos2, timestep - 1) in reservation_table and (pos1, timestep) in reservation_table:
        other_agent = reservation_table[(pos2, timestep - 1)]
        if other_agent != agent_id and reservation_table[(pos1, timestep)] == other_agent:
            return True
    return False

def reserve_path(path, agent_id, reservation_table, start_time=0):
    """Reserve a path in the reservation table"""
    for i, pos in enumerate(path):
        timestep = start_time + i
        reservation_table[(pos, timestep)] = agent_id

def clear_agent_reservations(agent_id, reservation_table):
    """Clear all reservations for a specific agent"""
    keys_to_remove = [key for key, value in reservation_table.items() if value == agent_id]
    for key in keys_to_remove:
        del reservation_table[key]

def calculate_agent_priority(agent, current_timestep=0):
    """Calculate agent priority based on deadline and distance to goal"""
    distance_to_goal = manhattan_distance(agent.initial_position, agent.target)
    deadline_urgency = max(1, agent.deadline - current_timestep - distance_to_goal)
    return -deadline_urgency  # Negative for min-heap (more urgent = higher priority)

def get_path(agents: List[EnvAgent], rail: GridTransitionMap, max_timestep: int):
    """Main pathfinding function using prioritized planning with conflict resolution"""
    global global_reservation_table, global_agent_priorities
    
    # Initialize global state
    global_reservation_table = {}
    global_agent_priorities = {}
    
    # Calculate priorities for all agents
    for i, agent in enumerate(agents):
        global_agent_priorities[i] = calculate_agent_priority(agent)
    
    # Sort agents by priority (most urgent first)
    agent_order = sorted(range(len(agents)), key=lambda i: global_agent_priorities[i])
    
    path_all = [[] for _ in range(len(agents))]
    
    # Plan paths for agents in priority order
    for agent_id in agent_order:
        agent = agents[agent_id]
        
        # Find path using A* with reservations
        path = a_star_with_reservations(
            agent.initial_position,
            agent.initial_direction,
            agent.target,
            rail,
            agent_id,
            max_timestep,
            global_reservation_table
        )
        
        if path:
            path_all[agent_id] = path
            reserve_path(path, agent_id, global_reservation_table)
        else:
            # If no path found, try with relaxed constraints
            path_all[agent_id] = [agent.initial_position]  # Stay at start
    
    return path_all

def replan(agents: List[EnvAgent], rail: GridTransitionMap, current_timestep: int, 
           existing_paths: List[Tuple], max_timestep: int, 
           new_malfunction_agents: List[int], failed_agents: List[int]):
    """Replan paths when malfunctions or conflicts occur"""
    global global_reservation_table, global_agent_priorities
    
    if debug:
        print(f"Replanning at timestep {current_timestep}", file=sys.stderr)
        print(f"New malfunction agents: {new_malfunction_agents}", file=sys.stderr)
        print(f"Failed agents: {failed_agents}", file=sys.stderr)
    
    # Identify all agents that need replanning
    agents_to_replan = set(new_malfunction_agents + failed_agents)
    
    # Add agents that might be affected by the failed agents
    for failed_agent in failed_agents:
        # Find agents whose paths might conflict with the failed agent's new situation
        for agent_id in range(len(agents)):
            if agent_id not in agents_to_replan:
                # Check if this agent's future path conflicts with failed agents
                if current_timestep < len(existing_paths[agent_id]):
                    agents_to_replan.add(agent_id)
    
    # Clear reservations for agents that need replanning
    for agent_id in agents_to_replan:
        clear_agent_reservations(agent_id, global_reservation_table)
    
    # Update current positions and handle malfunctions
    updated_paths = list(existing_paths)
    
    for agent_id in agents_to_replan:
        agent = agents[agent_id]
        
        # Determine current position
        if current_timestep < len(existing_paths[agent_id]):
            current_pos = existing_paths[agent_id][current_timestep]
        else:
            current_pos = agent.target if existing_paths[agent_id] else agent.initial_position
        
        # Check if agent has malfunction
        malfunction_duration = 0
        if hasattr(agent, 'malfunction_data') and agent.malfunction_data:
            malfunction_duration = agent.malfunction_data.get('malfunction', 0)
        
        # Plan new path from current position
        if malfunction_duration > 0:
            # Agent is malfunctioning, stay in place
            new_path = [current_pos] * (malfunction_duration + 1)
            # After malfunction, try to reach target
            remaining_path = a_star_with_reservations(
                current_pos,
                agent.direction if hasattr(agent, 'direction') else agent.initial_direction,
                agent.target,
                rail,
                agent_id,
                max_timestep,
                global_reservation_table,
                current_timestep + malfunction_duration
            )
            if remaining_path:
                new_path.extend(remaining_path[1:])  # Skip first position to avoid duplication
        else:
            # Normal replanning
            new_path = a_star_with_reservations(
                current_pos,
                agent.direction if hasattr(agent, 'direction') else agent.initial_direction,
                agent.target,
                rail,
                agent_id,
                max_timestep,
                global_reservation_table,
                current_timestep
            )
        
        if new_path:
            # Update the path from current timestep onwards
            updated_path = existing_paths[agent_id][:current_timestep] + new_path
            updated_paths[agent_id] = updated_path
            # Reserve the new path
            reserve_path(new_path, agent_id, global_reservation_table, current_timestep)
        else:
            # If no path found, extend current path with staying in place
            if current_timestep < len(existing_paths[agent_id]):
                stay_path = [existing_paths[agent_id][current_timestep]] * (max_timestep - current_timestep)
                updated_paths[agent_id] = existing_paths[agent_id][:current_timestep] + stay_path
    
    return updated_paths

#####################################################################
# Instantiate a Remote Client
# You should not modify codes below, unless you want to modify test_cases to test specific instance.
#####################################################################
if __name__ == "__main__":

    if len(sys.argv) > 1:
        remote_evaluator(get_path,sys.argv, replan = replan)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path, "multi_test_case/level*_test_*.pkl"))

        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path,"multi_test_case/level{}_test_{}.pkl".format(level, test)))
        test_cases.sort()
        deadline_files =  [test.replace(".pkl",".ddl") for test in test_cases]
        evaluator(get_path, test_cases, debug, visualizer, 3, deadline_files, replan = replan)




