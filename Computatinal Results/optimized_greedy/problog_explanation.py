from collections import defaultdict
from itertools import combinations
from model_process import *
import heapq
import random

from dataclasses import dataclass

import itertools
@dataclass(frozen=True)
class ProblemState:
    facts: tuple
    dnf: tuple

def transition_model(state: ProblemState, action) -> ProblemState:
    facts_h = dict(state.facts)
    dnf_h = [list(clause) for clause in state.dnf]

    if action['type'] == 'UpdateFact':
        facts_h[action['fact']] = action['new_probability']

    elif action['type'] == 'AddFact':
        facts_h[action['fact']] = action['new_probability']

    elif action['type'] == 'AddRule':
        rule = action['rule']
        dnf_h.append(rule)

    elif action['type'] == 'RemoveRule':
        dnf_h.remove(action['rule'])

    new_facts = tuple(sorted(facts_h.items()))
    new_dnf = tuple(sorted(tuple(sorted(cl)) for cl in dnf_h))

    return ProblemState(
        facts=new_facts,
        dnf=new_dnf,
    )


# A framework for defining and solving a search problem based on DNF formulas and facts.
class SearchProblem:

    def __init__(self, dnf_h, final_facts_h, dnf_a, final_facts_a, mpe_true, algorithm, cost):
        self.facts_h = final_facts_h  # dict
        self.facts_a = dict(final_facts_a)  # convert list of tuples to dict if needed

        self.dnf_h = [tuple(sorted(clause)) for clause in dnf_h]  # normalize
        self.dnf_a = [tuple(sorted(clause)) for clause in dnf_a]  # normalize

        self.mpe_true = mpe_true
        self.algorithm = algorithm
        self.cost_order = cost
        self.heuristic_cache = {}

    def actions(self, state: ProblemState):
        facts_h = dict(state.facts)
        dnf_h = [list(clause) for clause in state.dnf]
        possible_actions = []

        for fact, prob_a in self.facts_a.items():
            if fact in facts_h:
                prob_h = facts_h[fact]
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
                        if len(dnf_h) == 0:
                            possible_actions.append({
                                'type': 'AddFact',
                                'fact': fact,
                                'new_probability': prob_a
                            })

        for clause in self.dnf_a:
            if clause not in dnf_h:
                if all(fact.lstrip('\\+') in facts_h for fact in clause):
                    if self.algorithm == 'all':
                        possible_actions.append({
                            'type': 'AddRule',
                            'rule': clause
                        })
                    elif self.algorithm == 'clever':
                        if self.mpe_true:
                            A = {fact if prob > 0.5 else f'\\+{fact}' for fact, prob in facts_h.items()}
                            if all(fact in A for fact in clause):
                                possible_actions.append({
                                    'type': 'AddRule',
                                    'rule': clause
                                })
                        else:
                            if len(dnf_h) == 0:
                                A = {f'\\+{fact}' if prob > 0.5 else fact for fact, prob in facts_h.items()}
                                if any(fact in A for fact in clause):
                                    possible_actions.append({
                                        'type': 'AddRule',
                                        'rule': clause
                                    })

        for clause in dnf_h:
            if self.algorithm == 'all':
                possible_actions.append({
                    'type': 'RemoveRule',
                    'rule': clause
                })
            elif self.algorithm == 'clever':
                if not self.mpe_true:
                    not_A = {fact if prob > 0.5 else f'\\+{fact}' for fact, prob in facts_h.items()}
                    if all(fact in not_A for fact in clause):
                        possible_actions.append({
                            'type': 'RemoveRule',
                            'rule': clause
                        })
        return possible_actions

    def cost(self, action):
        if action['type'] == 'UpdateFact':
            return self.cost_order[0]
        elif action['type'] == 'AddFact':
            return self.cost_order[1]
        elif action['type'] == 'AddRule':
            return self.cost_order[2]
        elif action['type'] == 'RemoveRule':
            return self.cost_order[3]

    def is_goal(self, state: ProblemState, action):
        facts_h = dict(state.facts)
        dnf_h = [list(clause) for clause in state.dnf]
        if len(dnf_h) == 0:
            return False
        if self.algorithm == 'all':
            relevant_facts_h = get_relevant_facts(dnf_h, list(facts_h.items()))
            mpe_h_true_prob, _, mpe_h_false_prob, _ = calculate_mpe(dnf_h, relevant_facts_h)
            return (mpe_h_true_prob > mpe_h_false_prob) if self.mpe_true else (mpe_h_true_prob < mpe_h_false_prob)

        elif self.algorithm == 'clever':
            if action['type'] == 'UpdateFact':
                if self.mpe_true:
                    for clause in dnf_h:
                        if all((facts_h[f] > 0.5 if not f.startswith('\\+') else facts_h[f[2:]] < 0.5) for f in clause):
                            return True
                    return False
                else:
                    for clause in dnf_h:
                        if not any((facts_h[f] < 0.5 if not f.startswith('\\+') else facts_h[f[2:]] > 0.5) for f in clause):
                            return False
                    return True
            elif action['type'] == 'AddFact':
                return False
            elif action['type'] == 'AddRule':
                return True
            elif action['type'] == 'RemoveRule':
                if self.mpe_true:
                    return True
                else:
                    for clause in dnf_h:
                        # print("Goal:", clause)
                        if not any((facts_h[f] < 0.5 if not f.startswith('\\+') else facts_h[f[2:]] > 0.5) for f in clause):
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


def fact_modification_available(clauses, facts_h, facts_a):
    modification_available = {}
    for fact, prob_a in facts_a.items():
        if fact in facts_h:
            prob_h = facts_h[fact]
            if (prob_h > 0.5 and prob_a < 0.5) or (prob_h < 0.5 and prob_a > 0.5):
                modification_available[fact] = prob_h
    return modification_available


from itertools import combinations
from collections import defaultdict

def find_min_facts_to_cover_clauses(clauses, facts_h):
    # print(clauses)
    # print(facts_h)
    fact_dir_to_clauses = defaultdict(set)

    for i, clause in enumerate(clauses):
        for lit in clause:
            is_neg = lit.startswith('\\+')
            fact = lit.lstrip('\\+')

            if fact not in facts_h:
                continue  # 只能使用可修改的 fact 来满足子句

            current_val = facts_h[fact]

            # ✅ 情况一：当前值已经满足 literal → 可选择保持不变
            if (not is_neg and current_val < 0.5) or (is_neg and current_val >= 0.5):
                fact_dir_to_clauses[(fact, 'stay')].add(i)

            # ✅ 情况二：当前值不能满足 → 考虑方向修改
            if not is_neg and current_val >= 0.5:
                fact_dir_to_clauses[(fact, 'decrease')].add(i)
            elif is_neg and current_val < 0.5:
                fact_dir_to_clauses[(fact, 'increase')].add(i)

    all_clause_ids = set(range(len(clauses)))
    candidates = list(fact_dir_to_clauses.keys())  # 所有 (fact, direction) 对

    # 穷举组合，找最小集合覆盖所有子句，且不能选同一个 fact 多个方向
    for r in range(1, len(candidates) + 1):
        for subset in combinations(candidates, r):
            facts_to_action = defaultdict(set)
            for f, act in subset:
                facts_to_action[f].add(act)

            # ❌ 非法：同一个 fact 出现多个方向
            if any(len(v) > 1 for v in facts_to_action.values()):
                continue

            # ✅ 计算这组动作能覆盖的所有子句
            covered = set()
            for key in subset:
                covered |= fact_dir_to_clauses[key]

            if covered == all_clause_ids:
                # ✅ 只保留非 'stay' 的修改动作
                modifications_only = {f: d for f, d in subset if d != 'stay'}
                return modifications_only, len(modifications_only)

    return {}, float('inf')  # 无法覆盖所有子句

    # print(clauses)
    # print(facts_h)
    # if not clauses:
    #     return set(), 0

    # # 1. 构建方向明确的映射
    # fact_to_clauses_increase = defaultdict(set)
    # fact_to_clauses_decrease = defaultdict(set)

    # for i, clause in enumerate(clauses):
    #     for lit in clause:
    #         fact = lit.lstrip('\\+')
    #         is_neg = lit.startswith('\\+')

    #         if fact not in facts_h:
    #             continue

    #         prob = facts_h[fact]

    #         if not is_neg:
    #             # 正 literal：需要 prob >= 0.5 才满足
    #             if prob < 0.5:
    #                 fact_to_clauses_increase[fact].add(i)
    #         else:
    #             # 负 literal：需要 prob < 0.5 才满足
    #             if prob >= 0.5:
    #                 fact_to_clauses_decrease[fact].add(i)

    # # 2. 整合所有 (fact, direction) → clause ID 映射
    # fact_direction_to_clauses = {}
    # all_clause_ids = set(range(len(clauses)))

    # for fact in set(fact_to_clauses_increase) | set(fact_to_clauses_decrease):
    #     if fact in fact_to_clauses_increase:
    #         fact_direction_to_clauses[(fact, 'increase')] = fact_to_clauses_increase[fact]
    #     if fact in fact_to_clauses_decrease:
    #         fact_direction_to_clauses[(fact, 'decrease')] = fact_to_clauses_decrease[fact]

    # candidates = list(fact_direction_to_clauses.keys())

    # # 3. 穷举所有可能组合（最小覆盖子集）
    # for r in range(1, len(candidates) + 1):
    #     for subset in combinations(candidates, r):
    #         covered = set()
    #         for (fact, direction) in subset:
    #             covered |= fact_direction_to_clauses[(fact, direction)]
    #         if covered == all_clause_ids:
    #             return set(subset), len(subset)

    # return set(), float('inf')  # 无法覆盖所有子句

def find_min_modifications(clauses, facts_h):
    fact_dir_to_clauses = defaultdict(set)  # (fact, direction) -> 覆盖哪些 clause
    fact_dir_to_cost = dict()               # (fact, direction) -> cost

    for i, clause in enumerate(clauses):
        for lit in clause:
            fact = lit.lstrip('\\+')
            is_neg = lit.startswith('\\+')

            if fact not in facts_h:
                continue  # human 模型中不存在这个 fact，不能修改，跳过

            prob = facts_h[fact]

            if not is_neg:
                target_dir = '<'   # 想让 prob < 0.5 来满足正 literal
                cost = 0 if prob < 0.5 else 1
            else:
                target_dir = '>'   # 想让 prob ≥ 0.5 来满足反 literal
                cost = 0 if prob >= 0.5 else 1

            fact_dir_to_clauses[(fact, target_dir)].add(i)
            fact_dir_to_cost[(fact, target_dir)] = cost

    uncovered = set(range(len(clauses)))  # 记录还没被满足的 clause 的编号
    selected_facts = {}
    total_cost = 0

    while uncovered:
        best = None
        best_cover = set()
        best_cost = float('inf')

        for (fact, dir_), clause_ids in fact_dir_to_clauses.items():
            if fact in selected_facts:
                continue  # 已经选过这个 fact
            coverage = clause_ids & uncovered
            if not coverage:
                continue  # 没有新的子句能覆盖
            cost = fact_dir_to_cost[(fact, dir_)]
            # 优先选 cost 小但覆盖多的（greedy）
            if (len(coverage), -cost) > (len(best_cover), -best_cost):
                best = (fact, dir_)
                best_cover = coverage
                best_cost = cost

        if best is None:
            break  # 没有更多可覆盖的了

        fact, dir_ = best
        if best_cost > 0:
            selected_facts[fact] = dir_
            total_cost += best_cost
        uncovered -= best_cover
    uncovered_clauses = [clauses[i] for i in sorted(uncovered)]

        # print("Total clauses:", len(clauses))
        # print("Available modifications:", len(facts_h))
        # print("Final selected facts:", selected_facts)
        # print("Remaining uncovered clauses:", uncovered)

    return selected_facts, total_cost, uncovered_clauses


        # print("Total clauses:", len(clauses))
        # print("Available modifications:", len(facts_h))
        # print("Final selected facts:", selected_facts)
        # print("Remaining uncovered clauses:", uncovered)

    # return selected_facts, total_cost


import time

def greedy_maximum_coverage(clauses, facts_h, min_coverage_ratio=1.0, verbose=False):
    """
    贪心地选择最少的 (fact, direction) 组合，覆盖尽可能多的子句。
    - min_coverage_ratio: 1.0 表示全覆盖；<1 表示只覆盖一定比例即可
    - verbose: 若为 True，会输出覆盖率和运行时间
    """
    start_time = time.time()

    fact_dir_to_clauses = defaultdict(set)
    fact_dir_to_cost = {}

    for i, clause in enumerate(clauses):
        for lit in clause:
            fact = lit.lstrip('\\+')
            is_neg = lit.startswith('\\+')
            if fact not in facts_h:
                continue
            prob = facts_h[fact]
            if not is_neg:
                target_dir = '<'
                cost = 0 if prob < 0.5 else 1
            else:
                target_dir = '>'
                cost = 0 if prob >= 0.5 else 1
            fact_dir_to_clauses[(fact, target_dir)].add(i)
            fact_dir_to_cost[(fact, target_dir)] = cost

    uncovered = set(range(len(clauses)))
    selected_facts = {}
    total_cost = 0

    total_clauses = len(uncovered)
    target_coverage = int(min_coverage_ratio * total_clauses + 0.999)

    while uncovered and (total_clauses - len(uncovered)) < target_coverage:
        best = None
        best_cover = set()
        best_cost = float('inf')

        for (fact, dir_), clause_ids in fact_dir_to_clauses.items():
            if (fact, dir_) in selected_facts:
                continue
            coverage = clause_ids & uncovered
            if not coverage:
                continue
            cost = fact_dir_to_cost[(fact, dir_)]
            if (len(coverage), -cost) > (len(best_cover), -best_cost):
                best = (fact, dir_)
                best_cover = coverage
                best_cost = cost

        if best is None:
            break

        selected_facts[best] = True
        total_cost += best_cost
        uncovered -= best_cover

    covered_count = total_clauses - len(uncovered)
    coverage_ratio = covered_count / total_clauses

    if verbose:
        elapsed = time.time() - start_time
        print(f"[Greedy] Selected {len(selected_facts)} fact-directions, "
              f"Cost = {total_cost}, "
              f"Covered {covered_count}/{total_clauses} ({coverage_ratio:.1%}), "
              f"Time: {elapsed:.3f}s")

    return selected_facts, total_cost, covered_count




def find_min_modifications1(clauses, facts_h):
    fact_dir_to_clauses = defaultdict(set)
    fact_dir_to_cost = dict()

    for i, clause in enumerate(clauses):
        for lit in clause:
            fact = lit.lstrip('\\+')
            is_neg = lit.startswith('\\+')
            prob = facts_h.get(fact, 0)

            if not is_neg:
                target_dir = '<'   # want prob < 0.5
                cost = 0 if prob < 0.5 else 1
            else:
                target_dir = '>'  # want prob ≥ 0.5
                cost = 0 if prob >= 0.5 else 1

            fact_dir_to_clauses[(fact, target_dir)].add(i)
            fact_dir_to_cost[(fact, target_dir)] = cost

    uncovered = set(range(len(clauses)))
    selected_facts = {}
    total_cost = 0

    while uncovered:
        best = None
        best_cover = set()
        best_cost = float('inf')

        for (fact, dir_), clause_ids in fact_dir_to_clauses.items():
            if fact in selected_facts:
                continue
            coverage = clause_ids & uncovered
            if not coverage:
                continue
            cost = fact_dir_to_cost[(fact, dir_)]
            # 优先选 cost 小但覆盖多的
            if (len(coverage), -cost) > (len(best_cover), -best_cost):
                best = (fact, dir_)
                best_cover = coverage
                best_cost = cost

        if best is None:
            break  # some clauses can't be covered

        fact, dir_ = best
        if best_cost > 0:
            selected_facts[fact] = dir_
            total_cost += best_cost
        uncovered -= best_cover


    return selected_facts, total_cost


def greedy_min_cost_cover(to_delete_clauses, selected_deleted_facts, cost_update=1, cost_remove=2):
    from collections import defaultdict

    # 1. 建立 fact -> 覆盖子句映射
    fact_dir_to_clause_ids = defaultdict(set)
    for idx, clause in enumerate(to_delete_clauses):
        for lit in clause:
            fact = lit.lstrip('\\+')
            is_neg = lit.startswith('\\+')
            required_dir = '<' if not is_neg else '>'
            if (fact, required_dir) in selected_deleted_facts.items():
                fact_dir_to_clause_ids[(fact, required_dir)].add(idx)

    uncovered = set(range(len(to_delete_clauses)))
    selected_facts = set()

    while uncovered:
        best_fact = None
        best_coverage = set()

        for (fact, dir_), covered in fact_dir_to_clause_ids.items():
            if fact in selected_facts:
                continue
            effective_cover = covered & uncovered
            if len(effective_cover) > len(best_coverage):
                best_fact = (fact, dir_)
                best_coverage = effective_cover

        if not best_fact:
            break  # No more coverage possible

        selected_facts.add(best_fact[0])
        uncovered -= best_coverage

    total_cost = len(selected_facts) * cost_update + len(uncovered) * cost_remove
    return total_cost, selected_facts


def count_all_literals_above_half1(clauses, facts_h):
    count = 0
    to_delete_clauses = []

    # print(facts_h)

    for clause in clauses:
        # print(clause)
        all_high = True
        satisfy = False
        for lit in clause:
            fact = lit.lstrip('\\+')
            prob = facts_h.get(fact, 0)
            if not lit.startswith('\\+'):
                # 正 literal，prob >= 0.5 才算“满足”
                if prob < 0.5:
                    all_high = False 
                    satisfy = True
                    break
            else:
                # 负 literal，prob < 0.5 才算“满足”
                if prob > 0.5:
                    all_high = False
                    satisfy = True
                    break

        if all_high:
            count += 1
            to_delete_clauses.append(clause)


    # print(list(to_delete_clauses))

    return count, to_delete_clauses




def count_all_literals_above_half(clauses, facts_h):
    count = 0
    to_delete_clauses = set()

    # print(facts_h)

    for clause in clauses:
        all_high = True
        satisfy = False
        for lit in clause:
            fact = lit.lstrip('\\+')
            if fact in facts_h.keys():
                prob = facts_h.get(fact, 0)

                if not lit.startswith('\\+'):
                    # 正 literal，prob >= 0.5 才算“满足”
                    if prob < 0.5:
                        all_high = False 
                        satisfy = True
                        break
                else:
                    # 负 literal，prob < 0.5 才算“满足”
                    if prob > 0.5:
                        all_high = False
                        satisfy = True
                        break
            else:
                all_high = False

        if all_high:
            count += 1
            to_delete_clauses.add(tuple(clause))

        if satisfy:
            to_delete_clauses.add(tuple(clause))

    # print(list(to_delete_clauses))

    return count, list(to_delete_clauses)


import itertools

def count_clause_intersections(common_facts, facts_h, to_delete_clauses):
    new_literals = [
        f if facts_h[f] >= 0.5 else '\\+' + f
        for f in common_facts
    ]
    new_literals_set = set(new_literals)

    count = 0
    for clause in to_delete_clauses:
        if any(lit in new_literals_set for lit in clause):
            count += 1
    return count








def max_clauses_satisfiable_by_modification(final_dnf_h, avaliable_modification):
    """
    Given a DNF and available facts we can modify, return the maximum number of clauses
    that can be made satisfiable by modifying the probabilities of facts in the dict.
    """
    satisfiable_count = 0

    for clause in final_dnf_h:
        for lit in clause:
            fact = lit.lstrip('\\+')
            is_neg = lit.startswith('\\+')
            if fact in avaliable_modification:
                # Check if we can flip this literal to be satisfied
                target_prob = avaliable_modification[fact]
                if (not is_neg and target_prob >= 0.5) or (is_neg and target_prob < 0.5):
                    satisfiable_count += 1
                    break  # This clause is satisfied, move to next clause

    return satisfiable_count



def heuristic(problem, state, action):
    # print(problem.facts_a)

    facts_h = dict(state.facts)
    dnf_h = [list(clause) for clause in state.dnf]

    # print("Current:", facts_h)
    # print("Current:", dnf_h)

    change_probability_counts = []
    add_rule = False

    if problem.mpe_true:
        A = {fact if prob > 0.5 else f'\\+{fact}' for fact, prob in facts_h.items()}
        add_rule = any(all(fact in A for fact in clause) for clause in problem.dnf_a)

        change_probability_counts = []
        for clause in dnf_h:
            count = 0
            for fact in clause:
                clean_fact = fact.lstrip('\\+')
                prob = facts_h.get(clean_fact)
                is_negated = fact.startswith('\\+')
                if (not is_negated and prob < 0.5) or (is_negated and prob >= 0.5):
                    count += 1
            change_probability_counts.append(count)

        change_cost = min(change_probability_counts) * problem.cost_order[0]
        if add_rule:
            return min(change_cost, problem.cost_order[2])
        else:
            return min(change_cost, problem.cost_order[1] + problem.cost_order[2])
    else:
        if len(dnf_h)!= 0:
            # print(facts_h)
            avaliable_modification = fact_modification_available(dnf_h, facts_h, problem.facts_a)

            unavaliable_modification = {k: v for k, v in facts_h.items() if k not in avaliable_modification or avaliable_modification[k] != v}

            # print(len(unavaliable_modification))
            count_never_can_change, alway_delete_clauses = count_all_literals_above_half(dnf_h, unavaliable_modification)

            # print('Always:', alway_delete_clauses)

            final_dnf_h = list(set(map(tuple, dnf_h)) - set(map(tuple, alway_delete_clauses)))
            final_dnf_h = list(map(list, final_dnf_h))



            # print("Available Modification:", avaliable_modification)
            # print("Unavailable Modification:",unavaliable_modification)

            # print("Final DNF:", final_dnf_h)
            # print("Count Always Delete:", count_never_can_change)

            # print(len(final_dnf_h))
            # print(len(avaliable_modification))

            # start_ratio, min_ratio, step = auto_tune_coverage_params(final_dnf_h, avaliable_modification)


            # selected_facts, cost, hit, final_ratio = adaptive_greedy_coverage(
            #     final_dnf_h, avaliable_modification,
            #     start_ratio=start_ratio,
            #     min_ratio=min_ratio,
            #     step=step,
            #     verbose=True
            # )

            # # selected_facts, total_cost, _, _ = adaptive_greedy_coverage(final_dnf_h, avaliable_modification)
            # selected_facts, total_cost, _ = greedy_maximum_coverage(final_dnf_h, avaliable_modification, min_coverage_ratio=0.2, verbose=False)
            # # print(selected_facts)

            # selected_facts, total_cost = find_min_modifications(final_dnf_h, avaliable_modification)





            # selected_facts, _ = find_min_facts_to_cover_clauses(final_dnf_h, avaliable_modification)




            final_count, to_delete = count_all_literals_above_half1(final_dnf_h, facts_h)

            # print("Selected Facts:", selected_facts)




            cost = 0

            if final_count == 0:
                cost = 0
            else:
                selected_facts, greedy_cost, uncovered = find_min_modifications(final_dnf_h, avaliable_modification)
                # min_len = min(len(clause) for clause in final_dnf_h)
                # min_clauses = [clause for clause in final_dnf_h if len(clause) == min_len]

                # if len(min_clauses) >= 6:
                #     min_clauses = random.sample(min_clauses, 6) 

                # print(len(min_clauses))
                
                # selected_facts, _ = find_min_facts_to_cover_clauses(min_clauses, avaliable_modification)

                # selected_facts, _ = find_min_facts_to_cover_clauses(min_clauses, facts_h)

                # cost_min = 0
                # cost_max = 0
                # cost_middle = 0



                # selected_facts, a_cost = find_min_modifications(final_dnf_h, avaliable_modification)
                
                if greedy_cost == 0:
                    cost = problem.cost_order[3]

                # if len(selected_facts) == 0:
                #     cost = problem.cost_order[3]
                #     min_len = min(len(clause) for clause in to_delete)
                #     min_delete = [clause for clause in to_delete if len(clause) == min_len]
                #     min_modification, _ = find_min_facts_to_cover_clauses(min_delete, avaliable_modification)

                #     cost_min = min(problem.cost_order[3] * len(min_delete), problem.cost_order[3] + problem.cost_order[0] * len(min_modification))

                #     max_len = max(len(clause) for clause in to_delete)
                #     if max_len != min_len:
                #         max_delete = [clause for clause in to_delete if len(clause) == max_len]
                #         max_modification, _ = find_min_facts_to_cover_clauses(max_delete, avaliable_modification)
                #         cost_max = min(problem.cost_order[3] * len(max_delete), problem.cost_order[3] + problem.cost_order[0] * len(max_modification))

                #     if max_len - min_len >1:
                #         middle_len = max_len - 1
                #         middle_delete = [clause for clause in to_delete if len(clause) == middle_len]
                #         middle_modification, _ = find_min_facts_to_cover_clauses(middle_delete, avaliable_modification)
                #         cost_middle = min(problem.cost_order[3] * len(middle_delete), problem.cost_order[3] + problem.cost_order[0] * len(middle_modification))



                    # cost = max(max(cost_min, cost_max, cost_middle), problem.cost_order[3])
                    # print(cost)
                else:
                    # cost = min(greedy_cost * problem.cost_order[0], final_count * problem.cost_order[3])
                    cost = min(greedy_cost * min(problem.cost_order[0], problem.cost_order[3]), final_count * problem.cost_order[3])





            # cost = 0

            # if len(selected_facts) == 0:
            #     cost = problem.cost_order[2]
            # else:
            #     if final_count != 0:
            #         cost = min(len(selected_facts) * problem.cost_order[0], final_count * problem.cost_order[3])
            # # print(cost)



            # # # print(selected_facts)
            # # # print(facts_h)

            # # # print(dnf_h)

            # print(count_never_can_change * problem.cost_order[3] + cost * 10)

            return 2 * (count_never_can_change * problem.cost_order[3] + cost)


            # selected_facts, total_cost = find_min_modifications1(dnf_h, facts_h)
            # count, to_delete_clauses = count_all_literals_above_half(dnf_h, facts_h)
            # selected_deleted_facts, deleted_cost = find_min_modifications1(to_delete_clauses,facts_h)


            # return min(len(selected_facts) * problem.cost_order[0], count *  problem.cost_order[3])


            # print(no_need_remove)
            # print(no_need_to_remove2)

            # print(count * problem.cost_order[3])

            # print(len(common_facts)*problem.cost_order[0] + (count - no_need_remove) * problem.cost_order[3])
            # print(len(selected_facts)*problem.cost_order[0])


            # return max(min(len(common_facts)*problem.cost_order[0] + (count - no_need_remove) * problem.cost_order[3], len(selected_facts)*problem.cost_order[0]), problem.cost_order[0])
            
            # return min(count * problem.cost_order[3], len(selected_facts)*problem.cost_order[0])

            # print(problem.cost_order[0] * len(selected_deleted_facts) - problem.cost_order[3] * count)

            # if (problem.cost_order[0] * len(selected_deleted_facts) - problem.cost_order[3] * count) <= 0:
            #     return min(len(selected_facts)*problem.cost_order[0], problem.cost_order[0] * len(selected_deleted_facts))
            # else:
            #     best_cost, best_combo = min_total_cost_to_break_clauses(to_delete_clauses, selected_deleted_facts, cost_update=problem.cost_order[0], cost_remove=problem.cost_order[3])
            #     return min(len(selected_facts)*problem.cost_order[0], best_cost)

            # cost = [len(selected_facts)*problem.cost_order[0], count * problem.cost_order[3]]
            # print(cost)
            # return min(cost)
            # return problem.cost_order[0]
        else:
            return 2 * problem.cost_order[2]










# def search_astar(problem):
#     """
#     Perform A* search. Return (path, cost) for the goal with minimal (real) cost, 
#     or (None, None) if no solution is found.

#     - 'heuristic' is a function: heuristic(problem, state) -> non-negative float
#     """
#     # 1) Build initial state
#     start_state = (
#         frozenset(problem.facts_h.items()),
#         frozenset(frozenset(clause) for clause in problem.dnf_h)
#     )


#     start_g = 0
#     start_path = []
#     start_action = None

#     # We compute f = g + h
#     start_h = heuristic(problem)
#     start_f = start_g + start_h

#     # 2) Priority queue now stores (f, g, state, path, last_action)
#     open_list = []
#     heapq.heappush(open_list, (start_f, start_g, start_state, start_path, start_action))

#     # print("Intial open_list =", open_list)


#     best_g = {start_state: 0}

#     while open_list:
#         cur_f, cur_g, cur_state, cur_path, last_action = heapq.heappop(open_list)

#         # print("open_list =", open_list)

#         # If there's a better g known, skip
#         if cur_state in best_g and cur_g > best_g[cur_state]:
#             continue

#         # Expand
#         facts_h_dict = dict(cur_state[0])
#         dnf_h_list   = [list(clause) for clause in cur_state[1]]
#         problem.facts_h = facts_h_dict
#         problem.dnf_h   = dnf_h_list

#         # Reconstruct state if needed, or check last_action if needed
#         # ...
#         # If last_action leads to goal => return
#         if last_action is not None:
#             if problem.is_goal(last_action):
#                 return cur_path, cur_g

      

#         actions = problem.actions()

#         # print(actions)


#         # print(actions)

#         for action in actions:
#             problem.facts_h = {**facts_h_dict}
#             problem.dnf_h = [list(clause) for clause in dnf_h_list]

#             problem.transition_model(action)
#             step_cost = problem.cost(action)
#             new_g = cur_g + step_cost

#             next_state = (
#                 frozenset(problem.facts_h.items()),
#                 frozenset(frozenset(cl) for cl in problem.dnf_h)
#             )

           

#             # If new or improved
#             if (next_state not in best_g) or (new_g < best_g[next_state]):
#                 best_g[next_state] = new_g
#                 # compute new f
#                 new_h = heuristic(problem)
#                 new_f = new_g + new_h
#                 new_path = cur_path + [action]
#                 heapq.heappush(open_list, (new_f, new_g, next_state, new_path, action))


#     return None, None

def count_all_literals_above_half2(clauses, facts_h):
    count = 0
    to_delete_clauses = set()

    # print(facts_h)

    for clause in clauses:
        all_high = True
        satisfy = False
        for lit in clause:
            fact = lit.lstrip('\\+')
            if fact in facts_h.keys():
                prob = facts_h.get(fact, 0)

                if not lit.startswith('\\+'):
                    # 正 literal，prob >= 0.5 才算“满足”
                    if prob < 0.5:
                        all_high = False 
                        satisfy = True
                        break
                else:
                    # 负 literal，prob < 0.5 才算“满足”
                    if prob > 0.5:
                        all_high = False
                        satisfy = True
                        break
            else:
                all_high = False

        if all_high:
            count += 1
            to_delete_clauses.add(tuple(clause))

    # print(list(to_delete_clauses))

    return count, list(to_delete_clauses)
    

def search_astar(problem):
    # print(problem.facts_a)
    # print(problem.facts_h)
    # print(problem.dnf_h)

    if problem.mpe_true:
        start_state = ProblemState(
        facts=tuple(sorted(problem.facts_h.items())),
        dnf=tuple(sorted(tuple(sorted(clause)) for clause in problem.dnf_h))
        )



        start_g = 0
        start_path = []
        start_action = None
        start_h = heuristic(problem, start_state, start_action)
        # print("Start:", start_h)
        start_f = start_g + start_h

    else:
        avaliable_modification = fact_modification_available(problem.dnf_h, problem.facts_h, problem.facts_a)

        unavaliable_modification = {
            k: v for k, v in problem.facts_h.items()
            if k not in avaliable_modification or avaliable_modification[k] != v
        }


        # count_never_can_change, always_delete_clauses = count_all_literals_above_half(problem.dnf_h, unavaliable_modification)
        count, to_delete_clauses = count_all_literals_above_half2(problem.dnf_h, unavaliable_modification)

        removal_actions = []
        for clause in to_delete_clauses:
            removal_actions.append({
                "type": "RemoveRule",
                "rule": tuple(clause)
            })

        # print(to_delete_clauses)


        final_dnf_h = list(set(map(tuple, problem.dnf_h)) - set(map(tuple, to_delete_clauses)))
        final_dnf_h = list(map(list, final_dnf_h))

        # print(final_dnf_h)

        # === 3. 应用这些动作更新初始状态 ===

        # === 4. 生成初始状态 ===
        start_state = ProblemState(
            facts=tuple(sorted(problem.facts_h.items())),
            dnf=tuple(sorted(tuple(sorted(clause)) for clause in final_dnf_h))
        )

        start_g = count * problem.cost_order[3]
        start_path = list(removal_actions)
        # print(start_path)
        # print(start_path)
        if len(removal_actions) == 0:
            start_action = None
        else:
            start_action = removal_actions[-1]

            if problem.is_goal(start_state, start_action):
                return start_path, start_g
        start_h = heuristic(problem, start_state, start_action)
        start_f = start_g + start_h
        # print(start_f)

    open_list = []
    counter = itertools.count()  # 🔥 Unique counter for tie-break

    # print(problem.facts_h)
    # print(problem.facts_a)

    heapq.heappush(open_list, (start_f, start_g, next(counter), start_state, start_path, start_action))
    best_g = {start_state: start_g}

    # print(start_g)


    while open_list:
        cur_f, cur_g, _, cur_state, cur_path, last_action = heapq.heappop(open_list)

        # print("=== Expanding Node ===")
        # print("Cur_g:", cur_g)
        # print(len(open_list))


        if cur_state in best_g and cur_g > best_g[cur_state]:
            continue

        # print(last_action)

        if last_action is not None and problem.is_goal(cur_state, last_action):
            # print(f"[DEBUG] Reached goal with cost: {cur_g}")
            # print(f"[DEBUG] Path taken: {cur_path}")
            return cur_path, cur_g

        actions = problem.actions(cur_state)
        # print(actions)

        for action in actions:
            next_state = transition_model(cur_state, action)
            # print("Trans:", transition_model)
            step_cost = problem.cost(action)
            new_g = cur_g + step_cost

            if (next_state not in best_g) or (new_g < best_g[next_state]):
                best_g[next_state] = new_g
                new_h = heuristic(problem, next_state, action)
                new_f = new_g + new_h
                new_path = cur_path + [action]
                # print(f"[DEBUG] Action: {action}")
                # print(f"[DEBUG] g: {new_g:.4f}, h: {new_h:.4f}, f: {new_f:.4f}")
                heapq.heappush(open_list, (new_f, new_g, next(counter), next_state, new_path, action))

    return None, None

# def search_astar(problem):
#     """
#     Perform A* search. Return (path, cost, trace_log) for the goal with minimal cost.
#     """
#     import heapq

#     trace_log = []  # 用于记录每个 step 的 trace 信息
#     step_counter = 0

#     start_state = (
#         frozenset(problem.facts_h.items()),
#         frozenset(frozenset(clause) for clause in problem.dnf_h)
#     )

#     start_g = 0
#     start_path = []
#     start_action = None
#     start_h = heuristic(problem)
#     start_f = start_g + start_h

#     open_list = []
#     heapq.heappush(open_list, (start_f, start_g, start_state, start_path, start_action))
#     best_g = {start_state: 0}

#     while open_list:
#         cur_f, cur_g, cur_state, cur_path, last_action = heapq.heappop(open_list)

#         if cur_state in best_g and cur_g > best_g[cur_state]:
#             continue

#         # 还原当前状态
#         facts_h_dict = dict(cur_state[0])
#         dnf_h_list   = [list(clause) for clause in cur_state[1]]
#         problem.facts_h = facts_h_dict
#         problem.dnf_h   = dnf_h_list

#         # 记录当前 step 展开状态
#         step_record = {
#             "step": step_counter,
#             "g": cur_g,
#             "h": heuristic(problem),
#             "f": cur_f,
#             "facts_h": dict(problem.facts_h),
#             "dnf_h": [list(clause) for clause in problem.dnf_h],
#             "action_taken": str(last_action),
#             "is_goal": False,
#             "expanded_actions": [],
#         }

#         # 终止判断
#         if last_action is not None and problem.is_goal(last_action):
#             step_record["is_goal"] = True
#             trace_log.append(step_record)
#             return cur_path, cur_g, trace_log

#         actions = problem.actions()

#         for action in actions:
#             problem.facts_h = {**facts_h_dict}
#             problem.dnf_h = [list(clause) for clause in dnf_h_list]

#             problem.transition_model(action)
#             step_cost = problem.cost(action)
#             new_g = cur_g + step_cost

#             next_state = (
#                 frozenset(problem.facts_h.items()),
#                 frozenset(frozenset(cl) for cl in problem.dnf_h)
#             )

#             if (next_state not in best_g) or (new_g < best_g[next_state]):
#                 best_g[next_state] = new_g
#                 new_h = heuristic(problem)
#                 new_f = new_g + new_h
#                 new_path = cur_path + [action]

#                 heapq.heappush(open_list, (new_f, new_g, next_state, new_path, action))

#         #         # 记录扩展动作
#         #         step_record["expanded_actions"].append({
#         #             "action": str(action),
#         #             "step_cost": step_cost,
#         #             "new_g": new_g,
#         #             "new_h": new_h,
#         #             "new_f": new_f,
#         #         })

#         # trace_log.append(step_record)
#         # step_counter += 1
#         # print(trace_log)

#     return None, None



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





