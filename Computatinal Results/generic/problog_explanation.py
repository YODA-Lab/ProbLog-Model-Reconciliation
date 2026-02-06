from collections import defaultdict
from itertools import combinations
from model_process import *
import heapq

# A framework for defining and solving a search problem based on DNF formulas and facts.
class SearchProblem:

	def __init__(self, dnf_h, final_facts_h, dnf_a, final_facts_a, mpe_true, algorithm, cost):
		self.facts_h = final_facts_h
		self.facts_a = final_facts_a

		self.dnf_h = dnf_h
		self.dnf_a = dnf_a

		self.mpe_true = mpe_true
		self.algorithm = algorithm
		self.cost_order = cost

		self.added_rules = []



	def actions(self):
		possible_actions = []
		# Iterate through facts_a to find discrepancies or missing facts in facts_h
		# Step 1: Handle discrepancies in facts_h and facts_a
		for fact, prob_a in self.facts_a.items():
			if fact in self.facts_h:
				prob_h = self.facts_h[fact]
				# If probabilities differ, generate an Update action
				if prob_h != prob_a:
					if self.algorithm == 'all':
						possible_actions.append({
							'type': 'UpdateFact',
							'fact': fact,
							'new_probability': prob_a
						})
					elif self.algorithm == 'clever':
						if (prob_h > 0.5 and prob_a < 0.5) or (prob_h < 0.5 and prob_a > 0.5):
							possible_actions.append({
								'type': 'UpdateFact',
								'fact': fact,
								'new_probability': prob_a
							})
			# If fact is missing in facts_h, generate an Add action
			else:
				if self.algorithm == 'all':
					possible_actions.append({
						'type': 'AddFact',
						'fact': fact,
						'new_probability': prob_a
					})
				elif self.algorithm == 'clever':
					if self.mpe_true:
						possible_actions.append({
							'type': 'AddFact',
							'fact': fact,
							'new_probability': prob_a
							})
					else:
						if len(self.dnf_h) == 0:
							possible_actions.append({
								'type': 'AddFact',
								'fact': fact,
								'new_probability': prob_a
								})




		# Step 2: Handle discrepancies in dnf_h and dnf_a
		# (1) Add clauses from dnf_a that are not in dnf_h
		for clause in self.dnf_a:
			if clause not in self.dnf_h:
				# Check if all facts in the clause are available in facts_h
				if all(fact.lstrip('\\+') in self.facts_h for fact in clause):
					if self.algorithm == 'all':
						possible_actions.append({
							'type': 'AddRule',
							'rule': clause
						})
					elif self.algorithm == 'clever':
						if self.mpe_true:
							# Case 1: mpe_true == True
							A = {fact if prob > 0.5 else f'\\+{fact}' for fact, prob in self.facts_h.items()}
							# Check if all facts in the clause are in set A
							if all(fact in A for fact in clause):
								possible_actions.append({
									'type': 'AddRule',
									'rule': clause
								})
						else:
							if len(self.dnf_h) == 0:
								A = {f'\\+{fact}' if prob > 0.5 else fact for fact, prob in self.facts_h.items()}
								# Check if at least one fact in the clause is in set A
								if any(fact in A for fact in clause):
									possible_actions.append({
										'type': 'AddRule',
										'rule': clause
									})

		# Step 3: Handle rules in dnf_h that are not in dnf_a
		for clause in self.dnf_h:
			# if clause not in self.dnf_a:
			if self.algorithm == 'all':
				if clause not in self.added_rules:
					possible_actions.append({
						'type': 'RemoveRule',
						'rule': clause
					})
			elif self.algorithm == 'clever':
				if self.mpe_true:
					# If self.mpe_true == True, skip this rule removal step
					pass
				else:
					# Construct not_A based on self.facts_h
					not_A = {fact if prob > 0.5 else f'\\+{fact}' for fact, prob in self.facts_h.items()}
					
					# Check if all facts in the clause are in not_A
					if all(fact in not_A for fact in clause):
						possible_actions.append({
							'type': 'RemoveRule',
							'rule': clause
						})
		return possible_actions

	def transition_model(self, action):
		# Update the probability of an existing fact in facts_h
		if action['type'] == 'UpdateFact':
			self.facts_h[action['fact']] = action['new_probability']

		# Add a new fact with the specified probability to facts_h
		elif action['type'] == 'AddFact':
			self.facts_h[action['fact']] = action['new_probability']

		# Add a new rule to dnf_h
		elif action['type'] == 'AddRule':
			self.dnf_h.append(action['rule'])
			self.added_rules.append(action['rule'])

		# Remove an existing clause from dnf_h
		elif action['type'] == 'RemoveRule':
			self.dnf_h.remove(action['rule'])

	def cost(self, action):
		if action['type'] == 'UpdateFact':
			return self.cost_order[0]
		elif action['type'] == 'AddFact':
			return self.cost_order[1]
		elif action['type'] == 'AddRule':
			return self.cost_order[2]
		elif action['type'] == 'RemoveRule':
			return self.cost_order[3]


	def is_goal(self, action):
		if len(self.dnf_h) == 0:
			return False
		if self.algorithm == 'all':
			relevant_facts_h = get_relevant_facts(self.dnf_h, list(self.facts_h.items()))
			mpe_h_true_prob, mpe_h_true_assignment, mpe_h_false_prob, mpe_h_false_assignment = calculate_mpe(self.dnf_h, relevant_facts_h)

			if self.mpe_true:
				if mpe_h_true_prob > mpe_h_false_prob:
					return True
				else:
					return False
			else:
				if mpe_h_true_prob < mpe_h_false_prob:
					return True
				else:
					return False

			# if self.mpe_true:
			# 	for clause in self.dnf_h:
			# 		if all(
			# 			(self.facts_h[fact] > 0.5 if not fact.startswith('\\+') else self.facts_h[fact[2:]] < 0.5)
			# 			for fact in clause
			# 		):
			# 			return True
					
			# 	return False
			# else:
			# 	for clause in self.dnf_h:
			# 		if not any(
			# 			(self.facts_h[fact] < 0.5 if not fact.startswith('\\+') else self.facts_h[fact[2:]] > 0.5)
			# 			for fact in clause
			# 		):
			# 			return False
			# 	return True



		elif self.algorithm == 'clever':
			if action['type'] == 'UpdateFact':
				if self.mpe_true:
					for clause in self.dnf_h:
						if all(
							(self.facts_h[fact] > 0.5 if not fact.startswith('\\+') else self.facts_h[fact[2:]] < 0.5)
							for fact in clause
						):
							return True
					return False
				else:
					for clause in self.dnf_h:
						if not any(
							(self.facts_h[fact] < 0.5 if not fact.startswith('\\+') else self.facts_h[fact[2:]] > 0.5)
							for fact in clause
						):
							return False
					return True
			elif action['type'] == 'AddFact':
				return False
			elif action['type'] == 'AddRule':
				return True
				# if self.mpe_true:
				# 	return True
				# else:
				# 	for clause in self.dnf_h:
				# 		if not any(
				# 			(self.facts_h[fact] < 0.5 if not fact.startswith('\\+') else self.facts_h[fact[2:]] > 0.5)
				# 			for fact in clause
				# 		):
				# 			return False
				# 	return True

			elif action['type'] == 'RemoveRule':
				if self.mpe_true:
					return True
				else:
					for clause in self.dnf_h:
						if not any(
							(self.facts_h[fact] < 0.5 if not fact.startswith('\\+') else self.facts_h[fact[2:]] > 0.5)
							for fact in clause
						):
							return False
					return True

# Search all paths that satisfy the goal in the search problem.
def search_BFS(problem):
	# Stack for BFS: Each element is (facts_h, dnf_h, path)
	stack = [(problem.facts_h.copy(), problem.dnf_h.copy(), [])]
	goal_paths = []  # To store all paths that satisfy the goal

	while stack:
		# Pop the last state for DFS
		current_facts_h, current_dnf_h, path = stack.pop()

		# Temporarily set the current state in the problem
		problem.facts_h = current_facts_h
		problem.dnf_h = current_dnf_h

		# Generate all possible actions using the `actions` method
		actions = problem.actions()

		for action in actions:
			# Temporarily update the problem's state for this action
			problem.facts_h = current_facts_h.copy()
			problem.dnf_h = current_dnf_h.copy()

			# Apply the action using the `transition_model`
			problem.transition_model(action)

			# Record the new path
			new_path = path + [action]

			# Check if the new state satisfies the goal

			if problem.is_goal(action):
				goal_paths.append(new_path)

			else:
				# Push the new state and path onto the stack for further exploration

				stack.append((problem.facts_h.copy(), problem.dnf_h.copy(), new_path))

	return goal_paths

def search_dijkstra(problem):
    """
    Returns (path, cost) of the solution with minimal total cost, 
    or (None, None) if no solution found.
    
    We do NOT create is_goal_state(...). Instead, we still use problem.is_goal(action),
    but we only check it AFTER we pop from the priority queue. 
    That ensures we don't prematurely return a suboptimal path.
    """
    # 1) Build the initial "frozen" state
    start_state = (
        frozenset(problem.facts_h.items()),
        frozenset(frozenset(clause) for clause in problem.dnf_h)
    )
    start_g = 0
    start_path = []
    start_action = None  # There's no "last action" to reach the initial state

    # 2) Priority queue: each item is (g, state, path, last_action)
    open_list = []
    heapq.heappush(open_list, (start_g, start_state, start_path, start_action))

    # 3) Dictionary to record the best known cost to reach a state
    best_g = {start_state: 0}

    while open_list:
        # Pop the entry with the smallest g
        cur_g, cur_state, cur_path, last_action = heapq.heappop(open_list)

        # Reconstruct (facts_h, dnf_h) in mutable form, so we can call actions()
        facts_h_dict = dict(cur_state[0])  
        dnf_h_list   = [list(clause) for clause in cur_state[1]]
        problem.facts_h = facts_h_dict
        problem.dnf_h   = dnf_h_list

        # If we have found a cheaper way to cur_state before, skip
        if best_g[cur_state] < cur_g:
            continue

        # ========== KEY: Check if the last action leads to goal ==========
        # If 'last_action' is valid (not None), see if it satisfies the goal
        if last_action is not None:
            # We are "officially" expanding the state reached by 'last_action'
            # Now, if it is goal => it's guaranteed to be minimal cost
            # because Dijkstra expands states in ascending order of g.
            if problem.is_goal(last_action):
                return cur_path, cur_g

       

        # Generate next possible actions from here
        actions = problem.actions()

        # For each action => produce a successor state
        for action in actions:
            # Copy current state
            next_facts = facts_h_dict.copy()
            next_dnf   = [c[:] for c in dnf_h_list]

            # Apply the transition
            problem.facts_h = next_facts
            problem.dnf_h   = next_dnf
            problem.transition_model(action)

            step_cost = problem.cost(action)
            new_g = cur_g + step_cost

            # Freeze the successor
            next_state = (
                frozenset(problem.facts_h.items()),
                frozenset(frozenset(cl) for cl in problem.dnf_h)
            )
            # Path is extended by this action
            new_path = cur_path + [action]

            # If it's a new or better route to next_state
            if (next_state not in best_g) or (new_g < best_g[next_state]):
                best_g[next_state] = new_g
                # Push with 'action' as the last_action that led to next_state
                heapq.heappush(open_list, (new_g, next_state, new_path, action))

    # If we exhaust all reachable states without returning, no solution found.
    return None, None


def heuristic(problem):
    """
    Estimate how many corrections are needed (and their costs) to reach the goal state.
    This is a simple difference-based approach:
      1) Check if facts in the current state match the goal facts (facts_a).
         - If a fact is missing, we assume we need at least one AddFact.
         - If a fact is present but with a different probability, we assume one UpdateFact.
      2) Check if rules (dnf_h) match the goal rules (dnf_a).
         - If a rule is missing, we need one AddRule.
         - If an extra rule exists, we need one RemoveRule.

    Returns an integer or float that does NOT exceed the real cost from this state to a goal.
    """

    # Unpack the frozen sets back into normal structures
    # facts_h_frozen, dnf_h_frozen = state

    # # Convert them to mutable structures for easier comparison
    # current_facts = dict(facts_h_frozen)  # from frozenset of (fact, probability)
    # current_dnf   = [set(clause) for clause in dnf_h_frozen]

    # Extract costs from problem.cost_order
    cost_update = problem.cost_order[0]   # UpdateFact
    cost_add_f  = problem.cost_order[1]   # AddFact
    cost_add_r  = problem.cost_order[2]   # AddRule
    cost_rem_r  = problem.cost_order[3]   # RemoveRule

    return min(cost_update, cost_add_r, cost_rem_r)






def search_astar(problem):
    """
    Perform A* search. Return (path, cost) for the goal with minimal (real) cost, 
    or (None, None) if no solution is found.

    - 'heuristic' is a function: heuristic(problem, state) -> non-negative float
    """
    # 1) Build initial state
    start_state = (
        frozenset(problem.facts_h.items()),
        frozenset(frozenset(clause) for clause in problem.dnf_h)
    )


    start_g = 0
    start_path = []
    start_action = None

    # We compute f = g + h
    start_h = heuristic(problem)
    start_f = start_g + start_h

    # 2) Priority queue now stores (f, g, state, path, last_action)
    open_list = []
    heapq.heappush(open_list, (start_f, start_g, start_state, start_path, start_action))

    best_g = {start_state: 0}

    while open_list:
        cur_f, cur_g, cur_state, cur_path, last_action = heapq.heappop(open_list)

        # If there's a better g known, skip
        if cur_state in best_g and cur_g > best_g[cur_state]:
            continue

        # Expand
        facts_h_dict = dict(cur_state[0])
        dnf_h_list   = [list(clause) for clause in cur_state[1]]
        problem.facts_h = facts_h_dict
        problem.dnf_h   = dnf_h_list

        # Reconstruct state if needed, or check last_action if needed
        # ...
        # If last_action leads to goal => return
        if last_action is not None:
            if problem.is_goal(last_action):
                return cur_path, cur_g

      

        actions = problem.actions()

        # print(actions)

        for action in actions:
            problem.facts_h = {**facts_h_dict}
            problem.dnf_h = [list(clause) for clause in dnf_h_list]

            problem.transition_model(action)
            step_cost = problem.cost(action)
            new_g = cur_g + step_cost

            next_state = (
                frozenset(problem.facts_h.items()),
                frozenset(frozenset(cl) for cl in problem.dnf_h)
            )

           

            # If new or improved
            if (next_state not in best_g) or (new_g < best_g[next_state]):
                best_g[next_state] = new_g
                # compute new f
                new_h = heuristic(problem)
                new_f = new_g + new_h
                new_path = cur_path + [action]
                heapq.heappush(open_list, (new_f, new_g, next_state, new_path, action))

    return None, None




def search_greedy_all_solutions(problem, heuristic):
    """
    Performs a pure heuristic-based (Greedy Best-First) search that 
    explores all possible solutions, instead of stopping at the first one.
    
    It only uses the heuristic function h(n) to prioritize which state to expand,
    without considering the actual cost so far (g(n)). Consequently, it does not
    guarantee to find minimal-cost solutions; it merely attempts to find every 
    path that leads to a goal.
    
    Args:
        problem: An instance of a SearchProblem-like class, which includes:
            - facts_h, dnf_h to represent the current state
            - actions(): returns all feasible actions in the current state
            - transition_model(action): applies the action to generate a new state
            - is_goal(action): checks if executing that action achieves the goal
        heuristic: A function heuristic(problem, state) -> float
            that estimates how "close" state is to the goal. A smaller value 
            means the state is considered nearer to the goal.
            
    Returns:
        A list of solutions, where each solution is a list of actions that 
        yields a goal upon completion. If there are no solutions, returns an 
        empty list. 
    """

    # Convert the initial state's facts_h and dnf_h into immutable structures 
    # (so they can be used as keys in sets/dicts).
    start_state = (
        frozenset(problem.facts_h.items()),
        frozenset(frozenset(clause) for clause in problem.dnf_h)
    )

    # Start with an empty action path. There's no 'last_action' in the initial state.
    start_path = []
    start_action = None

    # Compute the initial heuristic value.
    start_h_val = heuristic(problem, start_state)

    # Priority queue (min-heap) containing (h_val, state, path, last_action).
    # We use h_val solely to choose which state to pop next.
    open_list = []
    heapq.heappush(open_list, (start_h_val, start_state, start_path, start_action))

    # 'visited' set to avoid re-expanding the same state repeatedly.
    # If your problem demands exploring a state multiple times (e.g., different 
    # paths with the same state), you can remove or modify this logic. 
    visited = set()
    visited.add(start_state)

    # 'solutions' will store all the goal-reaching paths we discover.
    solutions = []

    while open_list:
        # Pop the state with the smallest heuristic value
        cur_h_val, cur_state, cur_path, last_action = heapq.heappop(open_list)

        # Reconstruct the mutable version of the current state for 
        # problem.actions() and problem.transition_model().
        facts_h_dict = dict(cur_state[0])
        dnf_h_list   = [list(cl) for cl in cur_state[1]]
        problem.facts_h = facts_h_dict
        problem.dnf_h   = dnf_h_list

        # If this state was reached by some action (i.e., last_action is not None),
        # check whether that action satisfied the goal.
        if last_action is not None:
            if problem.is_goal(last_action):
                # Record the path that led to this goal. 
                # Unlike a regular "stop at first solution", we continue to find more solutions.
                solutions.append(cur_path)

        # Generate all successor actions from the current state
        actions = problem.actions()
        for action in actions:
            # Copy the current state's data to apply the transition
            next_facts = facts_h_dict.copy()
            next_dnf   = [clause[:] for clause in dnf_h_list]

            problem.facts_h = next_facts
            problem.dnf_h   = next_dnf
            problem.transition_model(action)

            # Convert the new state back into an immutable representation
            next_state = (
                frozenset(problem.facts_h.items()),
                frozenset(frozenset(cl) for cl in problem.dnf_h)
            )

            # Build the updated path by appending this action
            new_path = cur_path + [action]

            # If this state hasn't been visited yet, calculate its heuristic and push it onto the queue
            if next_state not in visited:
                visited.add(next_state)
                h_val = heuristic(problem, next_state)  # Greedy uses only h(n)
                heapq.heappush(open_list, (h_val, next_state, new_path, action))

    # When the queue is exhausted, we have collected every path that achieves the goal.
    return solutions





# Remove duplicate paths by sorting paths based on length and normalizing actions within each path.
def remove_duplicate_paths(goal_paths):
	unique_paths = set()  # To store normalized paths as frozensets
	final_paths = []  # To store sorted paths corresponding to unique sets

	for path in goal_paths:
		# Sort actions within each path
		sorted_path = sorted(
			path,
			key=lambda action: (
				action['type'],  # Primary sort by action type
				action.get('fact', ''),  # Secondary sort by fact
				action.get('new_probability', 0),  # Tertiary sort by probability
				tuple(action.get('rule', []))  # Lastly sort by rule
			)
		)

		# Normalize the sorted path
		normalized_path = frozenset(
			frozenset((k, tuple(v) if isinstance(v, list) else v) for k, v in action.items())
			for action in sorted_path
		)

		# Add to unique paths if not already present
		if normalized_path not in unique_paths:
			unique_paths.add(normalized_path)
			final_paths.append(sorted_path)

	# Sort final paths based on length, then on internal action order
	final_paths.sort(key=lambda path: (
		len(path),  # Primary sort by length
		[(
			action['type'],  # Primary sort by action type
			action.get('fact', ''),  # Secondary sort by fact
			action.get('new_probability', 0),  # Tertiary sort by probability
			tuple(action.get('rule', []))  # Lastly sort by rule
		) for action in path]
	))

	return final_paths





