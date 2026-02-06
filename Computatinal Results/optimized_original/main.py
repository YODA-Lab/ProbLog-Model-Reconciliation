from model_process import *
import argparse
import ast
from problog_explanation import *
import time
import os

# Argument input
parser = argparse.ArgumentParser()

'''
query: the event the human query
KBa: model of agent
KBh: model of human
algorithm: all, clever
tree_search: different search ways: BFS, A*, heuristic
cost_order: which one would be acceptable most, the order of ['UpdateFact', 'AddFact', 'AddRule', 'RemoveRule']
'''
parser.add_argument('--query') 
parser.add_argument('--KBa')
parser.add_argument('--KBh')
parser.add_argument('--tree_search')
parser.add_argument('--algorithm')
parser.add_argument('--cost_order')
args = parser.parse_args()

query = args.query
KBa = args.KBa
KBh = args.KBh
search = args.tree_search
algorithm = args.algorithm
cost_order = ast.literal_eval(args.cost_order)



if __name__ == '__main__':

	# Initial Step: Parse ProbLog files

	facts_h, rules_h = parse_problog_file(KBh)
	facts_a, rules_a = parse_problog_file(KBa)

	filename = os.path.basename(KBh)
	last_character = filename.split("_")[-2]

	# print("Human Facts:", facts_h)
	# print("Human Rules:", rules_h)

	# print("Agent Facts:", facts_a)
	# print("Agent Rules:", rules_a)

	# Step 1: Generate DNF formulas
	dnf_h = get_dnf(query, rules_h)
	dnf_a = get_dnf(query, rules_a)

	# print("Human DNF:", dnf_h)
	# print("Agent DNF:", dnf_a)

	# Step 2: Extract relevant facts
	relevant_facts_h = get_relevant_facts(dnf_h, facts_h)
	relevant_facts_a = get_relevant_facts(dnf_a, facts_a)

	# print("Relevant Facts (Human):", relevant_facts_h)
	# print("Relevant Facts (Agent):", relevant_facts_a)

	# Step 3: Generate final facts
	final_facts_h, final_facts_a = get_final_facts(relevant_facts_h, relevant_facts_a, facts_h, facts_a)

	# print("Final Facts (Human):", final_facts_h)
	# print("Final Facts (Agent):", final_facts_a)

	# print(dnf_h)
	# print(relevant_facts_h)

	# Calculate MPE for Human and Agent models
	# mpe_h_true_prob, mpe_h_true_assignment, mpe_h_false_prob, mpe_h_false_assignment = calculate_mpe(dnf_h, relevant_facts_h)
	# mpe_a_true_prob, mpe_a_true_assignment, mpe_a_false_prob, mpe_a_false_assignment = calculate_mpe(dnf_a, relevant_facts_a)

	# # Output MPE results
	# print(f"Human Model - MPE (True): Probability = {mpe_h_true_prob}, Assignment = {mpe_h_true_assignment}")
	# print(f"Human Model - MPE (False): Probability = {mpe_h_false_prob}, Assignment = {mpe_h_false_assignment}")
	# print(f"Agent Model - MPE (True): Probability = {mpe_a_true_prob}, Assignment = {mpe_a_true_assignment}")
	# print(f"Agent Model - MPE (False): Probability = {mpe_a_false_prob}, Assignment = {mpe_a_false_assignment}")

	if last_character == "A":
		mpe_true_agent = True
	else:
		mpe_true_agent = False

	# consistent = (mpe_a_true_prob > mpe_a_false_prob and mpe_h_true_prob > mpe_h_false_prob) or \
	# 		 (mpe_a_true_prob < mpe_a_false_prob and mpe_h_true_prob < mpe_h_false_prob)

	consistent = False
	

	if consistent:
		print("Human and Agent Models are Consistent.")
	else:
		# start = time.time()
		problem = SearchProblem(dnf_h, final_facts_h, dnf_a, final_facts_a, mpe_true_agent, algorithm, cost_order)

		if search == 'BFS':
			goal_paths = search_BFS(problem)
			# goal_paths = remove_duplicate_paths(goal_paths)
			
			# 定义每种动作的成本
			action_costs = {'UpdateFact': 0.9801, 'AddFact': 0.8688, 'AddRule': 1.0202, 'RemoveRule': 1.1511
			}

			# 计算路径的总成本函数
			def calculate_path_cost(path):
				return sum(action_costs.get(action['type'], float('inf')) for action in path)

			# 找到最低成本的路径
			shortest_path = min(goal_paths, key=calculate_path_cost)
			
			print("\nShortest Path (Based on Cost):")
			print(shortest_path)
			print(f"Total Cost: {calculate_path_cost(shortest_path):.4f}")

		elif search == 'Dijkstra':
			goal_paths, cost = search_dijkstra(problem)
			print(goal_paths)
			print(cost)

		elif search == "A":
			goal_paths, cost = search_astar(problem)
			# print(goal_paths)
			print(cost)

		elif search == 'Heuristic':
			goal_paths = search_greedy_all_solutions(problem)

			goal_paths = remove_duplicate_paths(goal_paths)
			print("All Goal Paths:")
			for i, path in enumerate(goal_paths, 1):
				print(f"Path {i}: {path}")

			shortest_path = min(goal_paths, key=len)
			print("\nShortest Path:")
			print(shortest_path)



