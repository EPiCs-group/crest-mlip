#!/bin/bash
# ============================================================================
# CREST-MLIP Job Generator and Submitter (Snellius)
# Automatically creates and submits CREST+UMA jobs for all folders with CONF.xyz
# Handles spin multiplicities: S1 (uhf=0), S3 (uhf=2), S5 (uhf=4), etc.
# ============================================================================

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

# SLURM settings (Snellius A100)
PARTITION="gpu_a100"
CPUS=18
GPUS=1
WALLTIME="48:00:00"
MEMORY=""  # Leave empty for default, or set like "64G"

# Paths
CREST_EXECUTABLE="$HOME/crest-mlip/build/crest"

# Conda environment
CONDA_ENV="crest-uma"
CONDA_INIT="source $HOME/miniforge3/etc/profile.d/conda.sh"

# Job settings
CHARGE=0

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

get_uhf_from_spin() {
    local spin_num=${1#S}
    echo $((spin_num - 1))
}

find_crest_folders() {
    local root_dir=${1:-.}
    local folders=()
    while IFS= read -r -d '' spin_folder; do
        if [ -f "$spin_folder/CONF.xyz" ]; then
            folders+=("$spin_folder")
        fi
    done < <(find "$root_dir" -type d -name 'S[13579]' -print0 | sort -z)
    echo "${folders[@]}"
}

# ============================================================================
# SLURM SCRIPT GENERATOR
# ============================================================================

generate_slurm_script() {
    local target_dir=$1
    local spin_folder_name=$2
    local uhf=$3
    local parent_name=$(basename "$(dirname "$target_dir")")
    local job_name="CREST_${parent_name}_${spin_folder_name}"
    local script_path="$target_dir/submit_crest.sh"

    local mem_flag=""
    if [ -n "$MEMORY" ]; then
        mem_flag="#SBATCH --mem=$MEMORY"
    fi

    cat > "$script_path" << SLURM_EOF
#!/bin/bash
#SBATCH --partition=$PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --gpus-per-task=$GPUS
#SBATCH --time=$WALLTIME
#SBATCH -J $job_name
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
${mem_flag}

# ============================================================================
# CREST-MLIP Conformer Search (UMA on GPU)
# Generated: $(date)
# Molecule: $parent_name
# Spin: $spin_folder_name (UHF=$uhf, charge=$CHARGE)
# ============================================================================

# Load CUDA runtime, then activate conda on top
module purge 2>/dev/null
module load 2024
module load CUDA/12.6.0
$CONDA_INIT
conda activate $CONDA_ENV

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK

echo "=========================================="
echo "CREST-MLIP Job: $parent_name / $spin_folder_name"
echo "=========================================="
echo "Directory: \$(pwd)"
echo "Hostname:  \$(hostname)"
echo "GPU:       \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:      \$(date)"
echo "UHF=$uhf  Charge=$CHARGE  Threads=$CPUS"
echo ""

# Check input
if [ ! -f "CONF.xyz" ]; then
    echo "ERROR: CONF.xyz not found!"
    exit 1
fi

# Write minimal TOML (only MLIP config that has no CLI equivalent)
cat > crest.toml << 'TOML_EOF'
runtype = "imtd-gc"

[[calculation.level]]
method = "uma"
device = "cuda"
TOML_EOF

# Verify GPU is visible
echo "CUDA check:"
python -c "import torch; print('  torch.cuda.available:', torch.cuda.is_available()); print('  devices:', torch.cuda.device_count()); print('  device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')" 2>&1
echo ""

echo "Running CREST conformer search with UMA..."
echo ""

$CREST_EXECUTABLE CONF.xyz \\
  --input crest.toml \\
  --noreftopo \\
  -T $CPUS \\
  -chrg $CHARGE \\
  -uhf $uhf \\
  > crest_output.log 2>&1
RC=\$?

echo ""
if [ \$RC -eq 0 ] && [ -f "crest_conformers.xyz" ]; then
    NAT=\$(head -1 crest_conformers.xyz)
    NCONF=\$(grep -c "^\$NAT\$" crest_conformers.xyz)
    echo "SUCCESS: \$NCONF conformers found"
    echo "Files: crest_conformers.xyz, crest_best.xyz"
    tail -20 crest_output.log | grep -E 'ensemble|conformers|E lowest|Wall Time|runtime'
else
    echo "FAILED (exit=\$RC)"
    echo "Last 20 lines of log:"
    tail -20 crest_output.log
    exit \$RC
fi

echo ""
echo "Finished: \$(date)"
SLURM_EOF

    chmod +x "$script_path"
    echo "$script_path"
}

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

generate_all_scripts() {
    local root_dir=${1:-.}
    log_info "Searching for S*/CONF.xyz in: $root_dir"

    local folders=($(find_crest_folders "$root_dir"))
    if [ ${#folders[@]} -eq 0 ]; then
        log_warn "No folders with CONF.xyz found!"
        return 1
    fi

    log_info "Found ${#folders[@]} folder(s)"
    echo ""

    local generated=0
    for folder in "${folders[@]}"; do
        local spin_name=$(basename "$folder")
        local parent_name=$(basename "$(dirname "$folder")")
        local uhf=$(get_uhf_from_spin "$spin_name")

        log_info "[$((generated + 1))/${#folders[@]}] $parent_name/$spin_name (UHF=$uhf)"

        script_path=$(generate_slurm_script "$folder" "$spin_name" "$uhf")
        log_info "  -> $script_path"
        generated=$((generated + 1))
    done

    echo ""
    log_info "Generated $generated submission script(s)"
    log_info "Review with: $0 dry-run $root_dir"
    log_info "Submit with: $0 submit $root_dir"
}

submit_all_jobs() {
    local root_dir=${1:-.}
    local dry_run=${2:-false}

    local scripts=($(find "$root_dir" -name "submit_crest.sh" -type f | sort))
    if [ ${#scripts[@]} -eq 0 ]; then
        log_warn "No submission scripts found! Run '$0 generate' first."
        return 1
    fi

    log_info "Found ${#scripts[@]} submission script(s)"
    [ "$dry_run" = true ] && log_warn "DRY RUN — not actually submitting"
    echo ""

    local submitted=0 skipped=0

    for script in "${scripts[@]}"; do
        local dir=$(dirname "$script")
        local spin_name=$(basename "$dir")
        local parent_name=$(basename "$(dirname "$dir")")

        if [ -f "$dir/crest_conformers.xyz" ]; then
            log_warn "  $parent_name/$spin_name — SKIP (already completed)"
            skipped=$((skipped + 1))
            continue
        fi

        if [ "$dry_run" = false ]; then
            cd "$dir"
            job_id=$(sbatch submit_crest.sh 2>&1)
            if [ $? -eq 0 ]; then
                log_info "  $parent_name/$spin_name — $job_id"
                submitted=$((submitted + 1))
            else
                log_error "  $parent_name/$spin_name — FAILED: $job_id"
            fi
            cd - > /dev/null
        else
            log_info "  $parent_name/$spin_name — would submit"
            submitted=$((submitted + 1))
        fi
    done

    echo ""
    log_info "Submitted: $submitted | Skipped: $skipped (completed)"
}

check_job_status() {
    local root_dir=${1:-.}
    local folders=($(find_crest_folders "$root_dir"))
    if [ ${#folders[@]} -eq 0 ]; then
        log_warn "No folders found!"
        return 1
    fi

    echo ""
    printf "%-55s %s\n" "JOB" "STATUS"
    printf "%-55s %s\n" "---" "------"

    local completed=0 running=0 pending=0 failed=0 notsubmitted=0

    for folder in "${folders[@]}"; do
        local spin_name=$(basename "$folder")
        local parent_name=$(basename "$(dirname "$folder")")
        local label="$parent_name/$spin_name"

        if [ -f "$folder/crest_conformers.xyz" ]; then
            local nat=$(head -1 "$folder/crest_conformers.xyz")
            local nconf=$(grep -c "^${nat}$" "$folder/crest_conformers.xyz" 2>/dev/null || echo "?")
            printf "%-55s ${GREEN}DONE${NC} (%s conformers)\n" "$label" "$nconf"
            completed=$((completed + 1))
        elif [ -f "$folder/crest_output.log" ]; then
            if grep -q "CREST terminated normally" "$folder/crest_output.log" 2>/dev/null; then
                printf "%-55s ${GREEN}DONE${NC}\n" "$label"
                completed=$((completed + 1))
            elif grep -q "CREST terminated with failures" "$folder/crest_output.log" 2>/dev/null; then
                printf "%-55s ${RED}FAILED${NC}\n" "$label"
                failed=$((failed + 1))
            else
                printf "%-55s ${YELLOW}RUNNING${NC}\n" "$label"
                running=$((running + 1))
            fi
        elif [ -f "$folder/submit_crest.sh" ]; then
            # Check SLURM queue
            local job_name="CREST_${parent_name}_${spin_name}"
            if squeue -u $USER -n "$job_name" --noheader 2>/dev/null | grep -q .; then
                printf "%-55s ${YELLOW}QUEUED${NC}\n" "$label"
                pending=$((pending + 1))
            else
                printf "%-55s READY\n" "$label"
                pending=$((pending + 1))
            fi
        else
            printf "%-55s NOT GENERATED\n" "$label"
            notsubmitted=$((notsubmitted + 1))
        fi
    done

    echo ""
    echo "Total: ${#folders[@]} | Done: $completed | Running: $running | Queued: $pending | Failed: $failed"
}

show_help() {
    cat << 'EOF'
CREST-MLIP Batch Job Manager (Snellius)

Usage:  ./crest_batch_submit.sh <command> [directory]

Commands:
  generate [dir]   Generate SLURM scripts for all S*/CONF.xyz folders
  submit [dir]     Submit all generated scripts to SLURM
  dry-run [dir]    Show what would be submitted without submitting
  status [dir]     Check status of all jobs
  list [dir]       List all folders that would be processed
  help             Show this help

Folder structure expected:
  molecule_name/
  ├── S1/CONF.xyz    (singlet, UHF=0)
  ├── S3/CONF.xyz    (triplet, UHF=2)
  └── S5/CONF.xyz    (quintet, UHF=4)

Configuration:
  Edit the CONFIGURATION section at the top of this script.
  Key settings: PARTITION, CPUS, WALLTIME, CREST_EXECUTABLE, CHARGE
EOF
}

# ============================================================================
# MAIN
# ============================================================================

case "${1:-help}" in
    generate)  generate_all_scripts "${2:-.}" ;;
    submit)    submit_all_jobs "${2:-.}" false ;;
    dry-run)   submit_all_jobs "${2:-.}" true ;;
    status)    check_job_status "${2:-.}" ;;
    list)      list_folders "${2:-.}" ;;
    help|--help|-h) show_help ;;
    *) log_error "Unknown command: $1"; show_help; exit 1 ;;
esac
