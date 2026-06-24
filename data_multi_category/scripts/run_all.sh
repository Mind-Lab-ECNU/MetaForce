#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/run_all_$(date +%Y%m%d_%H%M%S).log"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Redirect output to both console and log file
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================="
echo "Running all data preparation scripts"
echo "Working directory: ${SCRIPT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "=========================================="

# Function to run a script and report status
run_script() {
    local script_name=$1
    echo ""
    echo "=========================================="
    echo "Running: ${script_name}"
    echo "=========================================="
    bash "${SCRIPT_DIR}/${script_name}"
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ ${script_name} completed successfully"
    else
        echo "✗ ${script_name} failed with exit code ${exit_code}"
        return $exit_code
    fi
}

# Array of scripts to run in order
scripts=(
    "run_all_scripts_add.sh"
    "run_all_scripts.sh"
    "run_all_test_scripts_add.sh"
    "run_all_test_scripts.sh"
)

# Track overall status
overall_success=true

# Run each script
for script in "${scripts[@]}"; do
    if ! run_script "$script"; then
        overall_success=false
        echo ""
        echo "Warning: $script failed, but continuing with next script..."
    fi
done

echo ""
echo "=========================================="
echo "All scripts execution completed"
echo "=========================================="

if [ "$overall_success" = true ]; then
    echo "✓ All scripts completed successfully!"
    exit 0
else
    echo "⚠ Some scripts failed. Please check the output above."
    exit 1
fi
