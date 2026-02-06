import random
import os

def generate_facts(num_facts):
    facts = {}
    index = 0
    suffix = 0

    while len(facts) < num_facts:
        letter = chr(97 + index % 26)
        if letter == 'd':
            index += 1
            continue

        if index >= 26:
            fact_name = f"{letter}{suffix}"
        else:
            fact_name = letter

        prob = round(random.uniform(0.1, 0.9), 2)
        while prob == 0.5:
            prob = round(random.uniform(0.1, 0.9), 2)

        facts[fact_name] = prob
        index += 1

        if index >= 26:
            suffix += 1

    return facts

def generate_agent_rules(facts, num_rules, condition):
    rules = []
    fact_names = list(facts.keys())
    seen_rule_signatures = set()

    if condition == "A":
        high_prob_conditions = [
            f if facts[f] > 0.5 else f"\\+{f}"
            for f in fact_names if facts[f] > 0.5
        ]

        selected_conditions = random.sample(high_prob_conditions, min(4, len(high_prob_conditions)))
        signature = frozenset(sorted(selected_conditions))
        seen_rule_signatures.add(signature)
        rules.append(f"d :- {', '.join(selected_conditions)}.")

        while len(rules) < num_rules:
            conditions = []
            selected_facts = set()

            while len(conditions) < random.randint(2, 4):
                fact = random.choice(fact_names)
                if fact not in selected_facts and f"\\+{fact}" not in selected_facts:
                    literal = random.choice([fact, f"\\+{fact}"])
                    conditions.append(literal)
                    selected_facts.add(fact)

            signature = frozenset(sorted(conditions))
            if signature in seen_rule_signatures:
                continue 

            seen_rule_signatures.add(signature)
            rules.append(f"d :- {', '.join(conditions)}.")

    elif condition == "B":
        while len(rules) < num_rules:
            conditions = []
            selected_facts = set()

            low_prob_facts = [f for f in fact_names if facts[f] < 0.5] \
                            + [f"\\+{f}" for f in fact_names if facts[f] > 0.5]
            selected_condition = random.choice(low_prob_facts)
            conditions.append(selected_condition)
            selected_facts.add(selected_condition.lstrip("\\+"))

            while len(conditions) < random.randint(2, 4):
                fact = random.choice(fact_names)
                if fact not in selected_facts and f"\\+{fact}" not in selected_facts:
                    literal = random.choice([fact, f"\\+{fact}"])
                    conditions.append(literal)
                    selected_facts.add(fact)

            signature = frozenset(sorted(conditions))
            if signature in seen_rule_signatures:
                continue

            seen_rule_signatures.add(signature)
            rules.append(f"d :- {', '.join(conditions)}.")

    return rules



def generate_human_facts(agent_facts, complexity):
    human_facts = agent_facts.copy()
    num_adjust = int(len(agent_facts) * complexity)
    
    adjusted_facts = random.sample(list(human_facts.keys()), num_adjust)
    
    for fact in adjusted_facts:
        rand_choice = random.random()
        if rand_choice < 1 / 3:
            human_facts[fact] = round(1 - human_facts[fact], 2)
        elif rand_choice < 2 / 3:
            del human_facts[fact]
        else:
            del human_facts[fact]
            new_fact_name = generate_new_fact_name(human_facts)
            new_prob = generate_non_0_5_probability()
            human_facts[new_fact_name] = new_prob
    
    return human_facts

def generate_new_fact_name(existing_facts):
    index = 0
    while True:
        new_fact = f"new_fact_{index}"
        if new_fact not in existing_facts:
            return new_fact
        index += 1

def generate_non_0_5_probability():
    while True:
        prob = round(random.uniform(0.1, 0.9), 2)
        if prob != 0.5:
            return prob

def generate_human_rules(human_facts, agent_num_rules, complexity, condition):
    num_rules = max(1, int(agent_num_rules * (1 - complexity + 2/3 * complexity)))
    rules = []
    fact_names = list(human_facts.keys())
    seen_rule_signatures = set() 

    if condition == "A":
        while len(rules) < num_rules:
            conditions = []
            selected_facts = set()

            low_prob_facts = [f for f in fact_names if human_facts[f] < 0.5] \
                             + [f"\\+{f}" for f in fact_names if human_facts[f] > 0.5]

            selected_condition = random.choice(low_prob_facts)
            conditions.append(selected_condition)
            selected_facts.add(selected_condition.lstrip("\\+"))

            while len(conditions) < random.randint(2, 4):
                fact = random.choice(fact_names)
                if fact not in selected_facts and f"\\+{fact}" not in selected_facts:
                    literal = random.choice([fact, f"\\+{fact}"])
                    conditions.append(literal)
                    selected_facts.add(fact)

            signature = frozenset(sorted(conditions))
            if signature in seen_rule_signatures:
                continue

            seen_rule_signatures.add(signature)
            rules.append(f"d :- {', '.join(conditions)}.")

    elif condition == "B":
        high_prob_conditions = [
            f if human_facts[f] > 0.5 else f"\\+{f}"
            for f in fact_names if human_facts[f] > 0.5
        ]

        selected_conditions = random.sample(high_prob_conditions, min(4, len(high_prob_conditions)))
        signature = frozenset(sorted(selected_conditions))
        seen_rule_signatures.add(signature)
        rules.append(f"d :- {', '.join(selected_conditions)}.")

        while len(rules) < num_rules:
            conditions = []
            selected_facts = set()

            while len(conditions) < random.randint(2, 4):
                fact = random.choice(fact_names)
                if fact not in selected_facts and f"\\+{fact}" not in selected_facts:
                    literal = random.choice([fact, f"\\+{fact}"])
                    conditions.append(literal)
                    selected_facts.add(fact)

            signature = frozenset(sorted(conditions))
            if signature in seen_rule_signatures:
                continue

            seen_rule_signatures.add(signature)
            rules.append(f"d :- {', '.join(conditions)}.")

    return rules


def save_problog_model(filename, facts, rules):
    with open(filename, "w") as file:
        for fact, prob in facts.items():
            file.write(f"{prob}::{fact}.\n")
        file.write("\n")
        for rule in rules:
            file.write(rule + "\n")




def generate_experiments():
    settings = [(10, 5), (20, 10), (50, 25), (100, 50), (500, 250), (1000, 500)]
    complexities = [0.2, 0.4, 0.6, 0.8]  # 添加 0.8 复杂度
    conditions = ["A", "B"]

    if not os.path.exists("generated_models"):
        os.makedirs("generated_models")
    
    os.chdir("generated_models")
    
    for num_facts, num_rules in settings:
        for condition in conditions:
            for i in range(1, 1001):
                agent_filename = f"KBa_{num_facts}_{num_rules}_{condition}_{i}.pl"
                
                if not os.path.exists(agent_filename):
                    agent_facts = generate_facts(num_facts)
                    agent_rules = generate_agent_rules(agent_facts, num_rules, condition)
                    save_problog_model(agent_filename, agent_facts, agent_rules)
                    print(f"Generated Agent Model: {agent_filename}")
                else:
                    print(f"Agent Model already exists: {agent_filename}")
                
                for complexity in complexities:
                    human_filename = f"KBh_{num_facts}_{num_rules}_{complexity}_{condition}_{i}.pl"
                    
                    if not os.path.exists(human_filename):
                        human_facts = generate_human_facts(agent_facts, complexity)
                        human_rules = generate_human_rules(human_facts, num_rules, complexity, condition)
                        save_problog_model(human_filename, human_facts, human_rules)
                        print(f"Generated Human Model: {human_filename}")
                    else:
                        print(f"Human Model already exists: {human_filename}")

generate_experiments()

