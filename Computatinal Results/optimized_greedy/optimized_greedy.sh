
#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
MODEL_DIR="${SCRIPT_DIR}/../generated_models"
RESULT_LOG="${SCRIPT_DIR}/optimized_greedy2.log"

> "$RESULT_LOG"

FACTS=(10 20 50 100 1000)
RULES=(5 10 25 50 500)
COMPLEXITIES=(0.2 0.4 0.6 0.8)
TIME_LIMIT=600

echo "Running Experiments..." | tee -a "$RESULT_LOG"

for index in ${!FACTS[@]}; do
    fact=${FACTS[$index]}
    rule=${RULES[$index]}
    
    for complexity in "${COMPLEXITIES[@]}"; do
        total_time_A=0
        total_time_B=0

        echo "Testing: ${fact} facts, ${rule} rules, ${complexity} complexity, Condition A" | tee -a "$RESULT_LOG"
        
        for i in $(seq 1 100); do
            if [[ $fact -eq "100" && $rule -eq "50" && $complexity == "0.8" && $i -eq 15 ]]; then
                echo "Skipping Condition A/B - Model $i for ${fact} facts, ${rule} rules, ${complexity} complexity" | tee -a "$RESULT_LOG"
                continue
            fi

            KBa_A="${MODEL_DIR}/KBa_${fact}_${rule}_A_${i}.pl"
            KBh_A="${MODEL_DIR}/KBh_${fact}_${rule}_${complexity}_A_${i}.pl"
            KBa_B="${MODEL_DIR}/KBa_${fact}_${rule}_B_${i}.pl"
            KBh_B="${MODEL_DIR}/KBh_${fact}_${rule}_${complexity}_B_${i}.pl"

            
            # Condition A
            if [[ -f $KBa_A && -f $KBh_A ]]; then
                output=$(python3 -c "
import time
import subprocess
import timeout_decorator

@timeout_decorator.timeout($TIME_LIMIT)
def run_model():
    start = time.time()
    result = subprocess.run(
        ['python3', 'main1.py', '--query', 'd', '--KBa', '$KBa_A', '--KBh', '$KBh_A', '--tree_search', 'A', '--algorithm', 'clever', '--cost_order', '[0.9801,0.8688,1.0202,1.1511]'],
        capture_output=True, text=True
    )
    end = time.time()
    print(int((end - start) * 1000))
    print(result.stdout.strip())

try:
    run_model()
except timeout_decorator.timeout_decorator.TimeoutError:
    print('TIMEOUT')
")
                if [[ "$output" == "TIMEOUT" ]]; then
                    echo "Condition A - Model $i: Timeout after $TIME_LIMIT seconds" | tee -a "$RESULT_LOG"
                else
                    runtime_ms=$(echo "$output" | head -n 1)
                    cost_output=$(echo "$output" | tail -n +2)
                    total_time_A=$(echo "$total_time_A + $runtime_ms" | bc)
                    echo "Condition A - Model $i: ${runtime_ms} ms | Cost: ${cost_output}" | tee -a "$RESULT_LOG"
                fi
            fi

            # Condition B
            if [[ -f $KBa_B && -f $KBh_B ]]; then
                output=$(python3 -c "
import time
import subprocess
import timeout_decorator

@timeout_decorator.timeout($TIME_LIMIT)
def run_model():
    start = time.time()
    result = subprocess.run(
        ['python3', 'main1.py', '--query', 'd', '--KBa', '$KBa_B', '--KBh', '$KBh_B', '--tree_search', 'A', '--algorithm', 'clever', '--cost_order', '[0.9801,0.8688,1.0202,1.1511]'],
        capture_output=True, text=True
    )
    end = time.time()
    print(int((end - start) * 1000))
    print(result.stdout.strip())

try:
    run_model()
except timeout_decorator.timeout_decorator.TimeoutError:
    print('TIMEOUT')
")
                if [[ "$output" == "TIMEOUT" ]]; then
                    echo "Condition B - Model $i: Timeout after $TIME_LIMIT seconds" | tee -a "$RESULT_LOG"
                else
                    runtime_ms=$(echo "$output" | head -n 1)
                    cost_output=$(echo "$output" | tail -n +2)
                    total_time_B=$(echo "$total_time_B + $runtime_ms" | bc)
                    echo "Condition B - Model $i: ${runtime_ms} ms | Cost: ${cost_output}" | tee -a "$RESULT_LOG"
                fi
            fi
        done
        
        average_time_A=$(echo "scale=3; $total_time_A / 100" | bc)
        average_time_B=$(echo "scale=3; $total_time_B / 100" | bc)

         echo "----------------------------------------" | tee -a "$RESULT_LOG"

        {
            echo "Testing: ${fact} facts, ${rule} rules, ${complexity} complexity - Total Time for Condition A: ${total_time_A} ms"
            echo "Testing: ${fact} facts, ${rule} rules, ${complexity} complexity - Average Time for Condition A: ${average_time_A} ms"
            echo "Testing: ${fact} facts, ${rule} rules, ${complexity} complexity - Total Time for Condition B: ${total_time_B} ms"
            echo "Testing: ${fact} facts, ${rule} rules, ${complexity} complexity - Average Time for Condition B: ${average_time_B} ms"
            echo "----------------------------------------"
        } >> "$RESULT_LOG"
    done
done

echo "All Experiments Completed." | tee -a "$RESULT_LOG"

# SCRIPT_DIR=$(dirname "$(realpath "$0")")
# MODEL_DIR="${SCRIPT_DIR}/../generated_models"
# RESULT_LOG="${SCRIPT_DIR}/greedy.log"
# TMP_DIR="${SCRIPT_DIR}/tmp_output"
# mkdir -p "$TMP_DIR"
# > "$RESULT_LOG"

# FACTS=(10 20 50 100 1000)
# RULES=(5 10 25 50 500)
# COMPLEXITIES=(0.2 0.4 0.6 0.8)
# TIME_LIMIT=600

# echo "Running Experiments..." | tee -a "$RESULT_LOG"

# for index in ${!FACTS[@]}; do
#     fact=${FACTS[$index]}
#     rule=${RULES[$index]}

#     for complexity in "${COMPLEXITIES[@]}"; do
#         total_time_A=0
#         total_time_B=0

#         echo "Testing: ${fact} facts, ${rule} rules, ${complexity} complexity" | tee -a "$RESULT_LOG"

#         for i in $(seq 1 100); do
#             KBa_A="${MODEL_DIR}/KBa_${fact}_${rule}_A_${i}.pl"
#             KBh_A="${MODEL_DIR}/KBh_${fact}_${rule}_${complexity}_A_${i}.pl"
#             KBa_B="${MODEL_DIR}/KBa_${fact}_${rule}_B_${i}.pl"
#             KBh_B="${MODEL_DIR}/KBh_${fact}_${rule}_${complexity}_B_${i}.pl"

#             ### ---------- Condition A ----------
#             if [[ -f $KBa_A && -f $KBh_A ]]; then
#                 tmpfile="${TMP_DIR}/output_a_${fact}_${rule}_${complexity}_${i}.txt"

#                 python3 main.py --query d \
#                     --KBa "$KBa_A" --KBh "$KBh_A" \
#                     --tree_search A --algorithm clever \
#                     --cost_order "[0.9801,0.8688,1.0202,1.1511]" \
#                     > "$tmpfile" 2>&1
#                 exit_code=$?

#                 if [[ $exit_code -eq 124 ]]; then
#                     echo "Condition A - Model $i: Timeout after ${TIME_LIMIT}s" | tee -a "$RESULT_LOG"
#                 else
#                     runtime_val=$(grep "Runtime:" "$tmpfile" | awk '{print $2}')
#                     cost_output=$(grep "Cost:" "$tmpfile" | head -n 1 | awk '{print $2}')

#                     if [[ -z "$runtime_val" || -z "$cost_output" ]]; then
#                         echo "Condition A - Model $i: Output missing or parse error" | tee -a "$RESULT_LOG"
#                         echo "--- Raw Output A $i ---" >> "$RESULT_LOG"
#                         cat "$tmpfile" >> "$RESULT_LOG"
#                         echo "----------------------------------------" >> "$RESULT_LOG"
#                         continue
#                     fi

#                     echo "Condition A - Model $i: ${runtime_val} s | Cost: ${cost_output}" | tee -a "$RESULT_LOG"
#                     total_time_A=$(echo "$total_time_A + $runtime_val" | bc)
#                 fi
#             fi

#             ### ---------- Condition B ----------
#             if [[ -f $KBa_B && -f $KBh_B ]]; then
#                 tmpfile="${TMP_DIR}/output_b_${fact}_${rule}_${complexity}_${i}.txt"

#                 python3 main.py --query d \
#                     --KBa "$KBa_B" --KBh "$KBh_B" \
#                     --tree_search A --algorithm clever \
#                     --cost_order "[0.9801,0.8688,1.0202,1.1511]" \
#                     > "$tmpfile" 2>&1
#                 exit_code=$?

#                 if [[ $exit_code -eq 124 ]]; then
#                     echo "Condition B - Model $i: Timeout after ${TIME_LIMIT}s" | tee -a "$RESULT_LOG"
#                 else
#                     runtime_val=$(grep "Runtime:" "$tmpfile" | awk '{print $2}')
#                     cost_output=$(grep "Cost:" "$tmpfile" | head -n 1 | awk '{print $2}')

#                     if [[ -z "$runtime_val" || -z "$cost_output" ]]; then
#                         echo "Condition B - Model $i: Output missing or parse error" | tee -a "$RESULT_LOG"
#                         echo "--- Raw Output B $i ---" >> "$RESULT_LOG"
#                         cat "$tmpfile" >> "$RESULT_LOG"
#                         echo "----------------------------------------" >> "$RESULT_LOG"
#                         continue
#                     fi

#                     echo "Condition B - Model $i: ${runtime_val} s | Cost: ${cost_output}" | tee -a "$RESULT_LOG"
#                     total_time_B=$(echo "$total_time_B + $runtime_val" | bc)
#                 fi
#             fi
#         done

#         average_time_A=$(echo "scale=4; $total_time_A / 100" | bc)
#         average_time_B=$(echo "scale=4; $total_time_B / 100" | bc)

#         echo "----------------------------------------" | tee -a "$RESULT_LOG"
#         {
#             echo "Testing: ${fact} facts, ${rule} rules, ${complexity} complexity - Total Time A: ${total_time_A} s"
#             echo "Testing: ${fact} facts, ${rule} rules, ${complexity} complexity - Average Time A: ${average_time_A} s"
#             echo "Testing: ${fact} facts, ${rule} rules, ${complexity} complexity - Total Time B: ${total_time_B} s"
#             echo "Testing: ${fact} facts, ${rule} rules, ${complexity} complexity - Average Time B: ${average_time_B} s"
#             echo "----------------------------------------"
#         } >> "$RESULT_LOG"
#     done
# done

# echo "All Experiments Completed." | tee -a "$RESULT_LOG"
