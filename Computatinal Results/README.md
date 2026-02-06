# Computational Results

This folder contains the code and scripts used to generate and evaluate the computational results reported in **Section 6.2** of the paper.

## Files

- `generate_models.py`: Python script for generating models for different experimental cases.
- `generic/`: Contains code for running the **Generic Search** method.
- `optimized_original/`: Contains code for running the **Optimized Search** method.
- `optimized_greedy/`: Contains code for running the greedy version of the **Optimized Search** method.

---

## How to Use

### Step 1: Generate Models

Run the following command to generate all models required for the experiments:

```bash
python generate_models.py
```

---

### Step 2: Run Generic Search (`generic/`)

Run the baseline **Generic Search** method:

```bash
bash generic.sh
```

Note: Please update the directory path of the generated models in `generic.sh` before running.

---

### Step 3: Run Optimized Search (`optimized_original/`)

Run the **Optimized Search** method:

```bash
bash optimized.sh
```

Note: Please update the directory path of the generated models in `optimized.sh` before running.

---

### Step 4: Run Greedy Optimized Search (`optimized_greedy/`)

Run the greedy version of the **Optimized Search** method:

```bash
bash optimized_greedy.sh
```
Note: Please update the directory path of the generated models in `optimized_greedy.sh` before running.