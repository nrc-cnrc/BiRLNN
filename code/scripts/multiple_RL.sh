#!/usr/bin/env bash
set -euo pipefail

REPO="/home/jameshko/Documents/birlnn_latest/BIMODAL"
cd "$REPO"

# Python executable (can override with environment variable)
PYTHON="${PYTHON:-python3}"
SCRIPT="model/main_fine_tuner.py"

# Default experiments (space-separated array). Change or override via EXPERIMENTS env var (comma-separated).
experiments=("FBRNN_SELFIES_fixed_1024")

# Default reward pairs (semicolon-separated list of comma pairs). Change or override via REWARDS env var.
# Default pairs: 1,0 and 1,1
rewards=("1,0" "1,1")

# EXTRA_ARGS allows passing additional flags to main_fine_tuner.py, e.g. "--steps 100 --algo ppo"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# If EXPERIMENTS env var set, parse comma-separated list
if [ -n "${EXPERIMENTS:-}" ]; then
	IFS=',' read -r -a experiments <<< "$EXPERIMENTS"
fi

# If REWARDS env var set, parse semicolon-separated list
if [ -n "${REWARDS:-}" ]; then
	IFS=';' read -r -a rewards <<< "$REWARDS"
fi

echo "Using PYTHON=$PYTHON"
echo "Experiments: ${experiments[*]}"
echo "Reward pairs: ${rewards[*]}"
if [ -n "$EXTRA_ARGS" ]; then
	echo "Extra args passed to script: $EXTRA_ARGS"
fi

for exp in "${experiments[@]}"; do
	for rw in "${rewards[@]}"; do
		echo "------------------------------------------------------------"
		echo "Running experiment=$exp  reward_weights=$rw"
		# Run the Python script sequentially for each combo. EXTRA_ARGS is appended as-is.
		$PYTHON "$SCRIPT" --experiment "$exp" --reward-weights "$rw" $EXTRA_ARGS
		rc=$?
		if [ $rc -ne 0 ]; then
			echo "Run failed for experiment=$exp reward_weights=$rw (exit code $rc)" >&2
			exit $rc
		fi
	done
done

echo "All runs completed."

