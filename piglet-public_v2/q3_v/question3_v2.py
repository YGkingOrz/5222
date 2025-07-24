
from lib_piglet.utils.tools import eprint
from typing import List, Tuple
import glob, os, sys, time, json
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

# Store reservation table for conflict avoidance
reservation_table = defaultdict(set)  # {timestep: {(x,y), ...}}
edge_reservation_table = defaultdict(set)  # {timestep: {((x1,y1), (x2,y2)), ...}}

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_valid_moves(rail: GridTransitionMap, pos, direction):
    """Get valid moves from current position and direction"""
    valid_moves = []
    transitions = rail.get_transitions(pos[0], pos[1], direction)
    
    for new_direction, is_valid in enumerate(transitions):
        if is_valid:
            new_pos = pos
            if new_direction == Directions.NORTH:
                new_pos = (pos[0] - 1, pos[1])
            elif new_direction == Directions.EAST:
                new_pos = (pos[0], pos[1] + 1)
            elif new_direction == Directions.SOUTH:
                new_pos = (pos[0] + 1, pos[1])
            elif new_direction == Directions.WEST:
                new_pos = (pos[0], pos[1] - 1)
            
            # Check bounds
            if 0 <= new_pos[0] < rail.height and 0 <= new_pos[1] < rail.width:
                valid_moves.append((new_pos, new_direction))
    
    # Add wait action (stay in same position)
    valid_moves.append((pos, direction))
    return valid_moves

def is_conflict(pos1, pos2, time, prev_pos1=None, prev_pos2=None):
    """Check if there's a conflict between two positions at given time"""
    # Vertex conflict: same position at same time
    if pos1 == pos2:
        return True
    
    # Edge conflict: agents swap positions
    if prev_pos1 is not None and prev_pos2 is not None:
        if pos1 == prev_pos2 and pos2 == prev_pos1:
            return True
    
    return False

def space_time_astar(rail: GridTransitionMap, start_pos, start_dir, goal_pos, max_timestep, agent_id, deadline=None):
    """A* search in space-time graph avoiding reserved positions"""
    # Priority queue: (f_score, g_score, timestep, pos, direction, path)
    open_set = [(manhattan_distance(start_pos, goal_pos), 0, 0, start_pos, start_dir, [start_pos])]
    closed_set = set()
    
    while open_set:
        f_score, g_score, timestep, pos, direction, path = heapq.heappop(open_set)
        
        # Goal test
        if pos == goal_pos:
            # Extend path to max_timestep by staying at goal
            extended_path = path[:]
            while len(extended_path) < max_timestep:
                extended_path.append(goal_pos)
            return extended_path
        
        # Skip if already processed
        state = (timestep, pos, direction)
        if state in closed_set:
            continue
        closed_set.add(state)
        
        # Time limit check
        if timestep >= max_timestep - 1:
            continue
        
        # Get valid moves
        valid_moves = get_valid_moves(rail, pos, direction)
        
        for new_pos, new_direction in valid_moves:
            new_timestep = timestep + 1
            new_state = (new_timestep, new_pos, new_direction)
            
            if new_state in closed_set:
                continue
            
            # Check reservation conflicts
            if new_pos in reservation_table[new_timestep]:
                continue
            
            # Check edge conflicts
            edge = (pos, new_pos) if pos != new_pos else None
            if edge and edge in edge_reservation_table[new_timestep]:
                continue
            
            new_g_score = g_score + 1
            new_h_score = manhattan_distance(new_pos, goal_pos)
            
            # Add penalty for being late
            penalty = 0
            if deadline and new_timestep > deadline:
                penalty = 2 * (new_timestep - deadline)
            
            new_f_score = new_g_score + new_h_score + penalty
            new_path = path + [new_pos]
            
            heapq.heappush(open_set, (new_f_score, new_g_score, new_timestep, new_pos, new_direction, new_path))
    
    # If no path found, return a simple path that stays at start position
    fallback_path = [start_pos] * max_timestep
    return fallback_path

def reserve_path(path, agent_id):
    """Reserve positions and edges for a given path"""
    for t, pos in enumerate(path):
        if pos is not None:  # Safety check
            reservation_table[t].add(pos)
            
            # Reserve edge if moving
            if t > 0 and path[t-1] is not None:
                prev_pos = path[t-1]
                if prev_pos != pos:
                    edge_reservation_table[t].add((prev_pos, pos))
                    edge_reservation_table[t].add((pos, prev_pos))  # Bidirectional

def clear_agent_reservations(path, agent_id):
    """Clear reservations for a specific agent's path"""
    for t, pos in enumerate(path):
        if pos is not None and pos in reservation_table[t]:
            reservation_table[t].remove(pos)
        
        if t > 0 and path[t-1] is not None and pos is not None:
            prev_pos = path[t-1]
            if prev_pos != pos:
                edge = (prev_pos, pos)
                if edge in edge_reservation_table[t]:
                    edge_reservation_table[t].remove(edge)
                reverse_edge = (pos, prev_pos)
                if reverse_edge in edge_reservation_table[t]:
                    edge_reservation_table[t].remove(reverse_edge)

def get_path(agents: List[EnvAgent], rail: GridTransitionMap, max_timestep: int):
    """Main function to compute paths for all agents"""
    global reservation_table, edge_reservation_table
    
    # Clear previous reservations
    reservation_table.clear()
    edge_reservation_table.clear()
    
    # Calculate priorities based on deadline and distance
    agent_priorities = []
    for i, agent in enumerate(agents):
        distance = manhattan_distance(agent.initial_position, agent.target)
        deadline = getattr(agent, 'deadline', max_timestep)
        # Higher priority for agents with tighter deadlines and longer distances
        priority = deadline - distance
        agent_priorities.append((priority, i))
    
    # Sort by priority (lower value = higher priority)
    agent_priorities.sort()
    
    paths = [[] for _ in range(len(agents))]
    
    # Plan paths in priority order
    for priority, agent_id in agent_priorities:
        agent = agents[agent_id]
        deadline = getattr(agent, 'deadline', max_timestep)
        
        # Find path using space-time A*
        path = space_time_astar(
            rail, 
            agent.initial_position, 
            agent.initial_direction, 
            agent.target, 
            max_timestep, 
            agent_id,
            deadline
        )
        
        # Ensure path is not empty and has correct length
        if not path:
            path = [agent.initial_position] * max_timestep
        elif len(path) < max_timestep:
            # Extend path by staying at the last position
            last_pos = path[-1] if path else agent.initial_position
            while len(path) < max_timestep:
                path.append(last_pos)
        
        paths[agent_id] = path
        reserve_path(path, agent_id)
        
        if debug:
            print(f"Agent {agent_id} path length: {len(path)}, first few positions: {path[:5]}", file=sys.stderr)
    
    # Final validation - ensure all paths are valid
    for agent_id, path in enumerate(paths):
        if not path or len(path) < max_timestep:
            # Create fallback path
            agent = agents[agent_id]
            fallback_path = [agent.initial_position] * max_timestep
            paths[agent_id] = fallback_path
            if debug:
                print(f"Created fallback path for agent {agent_id}", file=sys.stderr)
    
    return paths

def replan(agents: List[EnvAgent], rail: GridTransitionMap, current_timestep: int, 
           existing_paths: List[Tuple], max_timestep: int, 
           new_malfunction_agents: List[int], failed_agents: List[int]):
    """Replan paths when malfunctions or failures occur"""
    global reservation_table, edge_reservation_table
    
    if debug:
        print(f"Replanning at timestep {current_timestep}", file=sys.stderr)
        print(f"New malfunction agents: {new_malfunction_agents}", file=sys.stderr)
        print(f"Failed agents: {failed_agents}", file=sys.stderr)
    
    # Create a copy of existing paths
    new_paths = [list(path) if path else [] for path in existing_paths]
    
    # Ensure all paths have minimum length
    for agent_id, path in enumerate(new_paths):
        if len(path) < max_timestep:
            agent = agents[agent_id]
            last_pos = path[-1] if path else agent.initial_position
            while len(path) < max_timestep:
                path.append(last_pos)
            new_paths[agent_id] = path
    
    # Clear future reservations from current timestep onwards
    for t in range(current_timestep, max_timestep):
        reservation_table[t].clear()
        edge_reservation_table[t].clear()
    
    # Re-reserve paths up to current timestep
    for agent_id, path in enumerate(new_paths):
        for t in range(min(current_timestep, len(path))):
            if t < len(path) and path[t] is not None:
                reservation_table[t].add(path[t])
                if t > 0 and path[t-1] is not None and path[t] != path[t-1]:
                    edge_reservation_table[t].add((path[t-1], path[t]))
                    edge_reservation_table[t].add((path[t], path[t-1]))
    
    # Identify agents that need replanning
    agents_to_replan = set(new_malfunction_agents + failed_agents)
    
    # Add agents whose future paths might be affected
    for agent_id in range(len(agents)):
        if agent_id not in agents_to_replan:
            agent = agents[agent_id]
            # Check if agent has reached goal
            if (hasattr(agent, 'status') and agent.status == 2) or \
               (current_timestep < len(new_paths[agent_id]) and 
                new_paths[agent_id][current_timestep] == agent.target):
                continue
            
            # If agent's path is too short, needs replanning
            if len(new_paths[agent_id]) <= current_timestep:
                agents_to_replan.add(agent_id)
    
    # Replan for affected agents
    for agent_id in agents_to_replan:
        agent = agents[agent_id]
        
        # Determine current position
        if hasattr(agent, 'position') and agent.position is not None:
            current_pos = agent.position
            current_dir = agent.direction
        elif current_timestep < len(new_paths[agent_id]) and new_paths[agent_id][current_timestep] is not None:
            current_pos = new_paths[agent_id][current_timestep]
            current_dir = agent.initial_direction  # Simplified
        else:
            current_pos = agent.initial_position
            current_dir = agent.initial_direction
        
        # Check for malfunction
        malfunction_duration = 0
        if hasattr(agent, 'malfunction_data') and agent.malfunction_data:
            malfunction_duration = agent.malfunction_data.get('malfunction', 0)
        
        # Clear old reservations for this agent
        if agent_id < len(new_paths) and len(new_paths[agent_id]) > current_timestep:
            old_path = new_paths[agent_id][current_timestep:]
            for t, pos in enumerate(old_path, current_timestep):
                if pos is not None and pos in reservation_table[t]:
                    reservation_table[t].remove(pos)
        
        # Plan new path from current position
        remaining_timesteps = max_timestep - current_timestep - malfunction_duration
        if remaining_timesteps > 0:
            deadline = getattr(agent, 'deadline', max_timestep)
            new_path_segment = space_time_astar(
                rail, current_pos, current_dir, agent.target, 
                remaining_timesteps, agent_id, deadline
            )
            
            if new_path_segment:
                # Update path from current timestep
                new_paths[agent_id] = new_paths[agent_id][:current_timestep + malfunction_duration]
                
                # Add wait actions for malfunction duration
                for _ in range(malfunction_duration):
                    new_paths[agent_id].append(current_pos)
                
                # Add new path segment
                if len(new_path_segment) > 1:
                    new_paths[agent_id].extend(new_path_segment[1:])  # Skip first position as it's current
                
                # Ensure path reaches max_timestep
                while len(new_paths[agent_id]) < max_timestep:
                    last_pos = new_paths[agent_id][-1] if new_paths[agent_id] else current_pos
                    new_paths[agent_id].append(last_pos)
                
                # Reserve new path
                for t in range(current_timestep + malfunction_duration, len(new_paths[agent_id])):
                    if t < len(new_paths[agent_id]) and new_paths[agent_id][t] is not None:
                        pos = new_paths[agent_id][t]
                        reservation_table[t].add(pos)
                        if t > 0 and new_paths[agent_id][t-1] is not None and new_paths[agent_id][t] != new_paths[agent_id][t-1]:
                            edge_reservation_table[t].add((new_paths[agent_id][t-1], pos))
                            edge_reservation_table[t].add((pos, new_paths[agent_id][t-1]))
    
    # Final validation for replanned paths
    for agent_id, path in enumerate(new_paths):
        if not path or len(path) < max_timestep:
            agent = agents[agent_id]
            fallback_path = [agent.initial_position] * max_timestep
            new_paths[agent_id] = fallback_path
    
    return new_paths

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




