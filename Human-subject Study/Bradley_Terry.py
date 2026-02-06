import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import pandas as pd

features = ["Add Fact", "Change Probability", "Add Rule", "Remove Rule"]




# Explanation 1: "Change Probability"
# Explanation 2: "Add Rule"
# Explanation 3: "Add Rule", "Add Fact"
# Explanation 4: "Add Rule", "Change Probability"
# Explanation 5: "Remove Rule", "Add Rule"


comparison_data = [
    (["Change Probability"], ["Add Rule"], 51, 49),
    (["Add Rule", "Add Fact"], ["Add Rule", "Change Probability"], 58, 42),
    (["Add Rule", "Add Fact"], ["Remove Rule", "Add Rule"], 52, 48),
    (["Add Rule", "Change Probability"], ["Remove Rule", "Add Rule"], 59, 41),
]




initial_betas = np.zeros(len(features))

# Bradley-Terry
def bradley_terry_probability(beta_f, beta_g):
    return np.exp(beta_f) / (np.exp(beta_f) + np.exp(beta_g))


def negative_log_likelihood(betas):
    loss = 0
    for R1, R2, count_R1, count_R2 in comparison_data:

        common_features = set(R1).intersection(set(R2))
        

        R1_unique = [f for f in R1 if f not in common_features]
        R2_unique = [f for f in R2 if f not in common_features]
        

        p_R1_gt_R2 = 0
        if len(R1_unique) > 0 and len(R2_unique) > 0:
            for f in R1_unique:
                for g in R2_unique:
                    idx_f = features.index(f)
                    idx_g = features.index(g)
                    p_R1_gt_R2 += bradley_terry_probability(betas[idx_f], betas[idx_g])
            p_R1_gt_R2 /= (len(R1_unique) * len(R2_unique))

        p_R2_gt_R1 = 1 - p_R1_gt_R2
        

        loss -= count_R1 * np.log(p_R1_gt_R2) + count_R2 * np.log(p_R2_gt_R1)
    return loss


result = minimize(negative_log_likelihood, initial_betas, method='L-BFGS-B')


optimized_betas = result.x


feature_costs = np.exp(-optimized_betas)


for feature, cost in zip(features, feature_costs):
    print(f"Feature: {feature}, Cost: {cost:.4f}")



