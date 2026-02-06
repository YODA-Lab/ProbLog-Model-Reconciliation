from itertools import product


# --------------------------------------------------
# Parse a ProbLog file to extract facts and rules.
def parse_problog_file(file_path):
    facts, rules = [], []
    # file_path = f"data/{file_path}"
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if '::' in line:  # Fact line
                prob, fact = line.split('::')
                facts.append((fact.strip('.'), float(prob)))
            elif ':-' in line:  # Rule line
                head, body = line.split(':-')
                head = head.strip()
                body = [b.strip() for b in body.strip('.').split(',')]
                rules.append((head, body))
    return facts, rules

# --------------------------------------------------
# Find dependencies based on the existing facts
def find_dependencies(head, rules):
    dnf = []
    for h, body in rules:
        if h == head:
            clause_dnf = []
            for b in body:
                if b.startswith('\\+'):  # Negation case
                    clause_dnf.append([[b]])
                else:
                    sub_dnf = find_dependencies(b, rules)
                    if sub_dnf:
                        clause_dnf.append(sub_dnf)
                    else:
                        clause_dnf.append([[b]])
            expanded_clause = [
                [item if isinstance(item, list) else [item] for item in combination]
                for combination in product(*clause_dnf)
            ]
            expanded_clause = [sum(items, []) for items in expanded_clause]
            dnf.extend(expanded_clause)
    return dnf

# Generate the DNF formula related to the query based on the rules.
def get_dnf(query, rules):

    return find_dependencies(query, rules)



# --------------------------------------------------
# Extract facts and probabilities directly related to the DNF formula.
def get_relevant_facts(dnf, facts):
    # Collect all facts (positive or from negation) from the DNF
    relevant_facts_set = {
        fact[2:] if fact.startswith('\\+') else fact  # Remove \\+ for negated facts
        for clause in dnf for fact in clause
    }
    # Return the related facts and their probabilities
    return {f: p for f, p in facts if f in relevant_facts_set}

# --------------------------------------------------
# Generate final used facts for Human and Agent models.
def get_final_facts(relevant_facts_h, relevant_facts_a, facts_h, facts_a):
    # Combine relevant facts from Human and Agent models
    all_relevant_facts = set(relevant_facts_h.keys()).union(set(relevant_facts_a.keys()))
    # Final facts for the Human model
    final_facts_h = {f: p for f, p in facts_h if f in all_relevant_facts}
    # Final facts for the Agent model
    final_facts_a = {f: p for f, p in facts_a if f in all_relevant_facts}

    return final_facts_h, final_facts_a

# --------------------------------------------------
# Calculate MPE
# Check if a single clause is satisfied by the current assignment.
def satisfies_clause(clause, assignment):
    for literal in clause:
        if literal.startswith('\\+'):  # Negation case
            fact = literal[2:].strip()
            if assignment.get(fact, False):  # If the fact is True, the clause is not satisfied
                return False
        else:  # Positive literal
            if not assignment.get(literal, False):  # If the fact is False, the clause is not satisfied
                return False
    return True

# Check if the entire DNF formula is satisfied by the current assignment.
def satisfies_dnf(dnf, assignment):
    for clause in dnf:
        if satisfies_clause(clause, assignment):  # If any clause is satisfied, the DNF is satisfied
            return True
    return False

# Calculate the Maximum Probability Explanation (MPE) for both True and False cases.
def calculate_mpe(dnf, relevant_facts):
    prob_map = dict(relevant_facts)  # relevant_facts is a dictionary {fact: probability}
    facts_list = list(prob_map.keys())  # Get all relevant fact names
    max_true_prob = 0
    max_false_prob = 0
    best_true_assignment = None
    best_false_assignment = None

    # Enumerate all possible fact assignments
    for values in product([True, False], repeat=len(facts_list)):
        assignment = dict(zip(facts_list, values))  # Generate the current assignment

        # Calculate the probability of the current assignment
        prob = 1.0
        for fact, value in assignment.items():
            if value:  # If the fact is True, multiply by its probability
                prob *= prob_map[fact]
            else:  # If the fact is False, multiply by (1 - probability)
                prob *= (1 - prob_map[fact])

        # Check if the current assignment satisfies the DNF
        if satisfies_dnf(dnf, assignment):  # If Query is True
            if prob > max_true_prob:
                max_true_prob = prob
                best_true_assignment = assignment
        else:  # If Query is False
            if prob > max_false_prob:
                max_false_prob = prob
                best_false_assignment = assignment

    return max_true_prob, best_true_assignment, max_false_prob, best_false_assignment


