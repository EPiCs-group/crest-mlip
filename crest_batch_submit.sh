#!/bin/bash
# ============================================================================
# CREST-MLIP Job Generator and Submitter (Snellius)
# Two-stage: (1) UMA geometry optimization, (2) UMA conformer search
# Handles spin multiplicities: S1 (uhf=0), S3 (uhf=2), S5 (uhf=4), etc.
# ============================================================================

# No set -e: we handle errors per-job, don't abort on first failure


# ============================================================================
# CONFIGURATION
# ============================================================================

# SLURM settings (Snellius A100)
PARTITION="gpu_a100"
CPUS=18
GPUS=1
WALLTIME_OPT="02:00:00"    # Wall time for optimization step
WALLTIME_CONF="48:00:00"   # Wall time for conformer search
MEMORY=""

# Paths
CREST_EXECUTABLE="$HOME/crest-mlip/build/crest"

# Conda environment
CONDA_ENV="crest-uma"
CONDA_INIT="source $HOME/miniforge3/etc/profile.d/conda.sh"

# UMA model (registry name or local path)
UMA_MODEL="uma-s-1p2"

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

# Common SLURM + environment header
slurm_header() {
    local job_name=$1
    local walltime=$2
    local mem_flag=""
    if [ -n "$MEMORY" ]; then
        mem_flag="#SBATCH --mem=$MEMORY"
    fi
    cat << HEADER_EOF
#!/bin/bash
#SBATCH --partition=$PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --gpus-per-task=$GPUS
#SBATCH --time=$walltime
#SBATCH -J $job_name
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
${mem_flag}

module purge 2>/dev/null
module load 2024
module load CUDA/12.6.0
$CONDA_INIT
conda activate $CONDA_ENV

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
HEADER_EOF
}

# ============================================================================
# SLURM SCRIPT GENERATORS
# ============================================================================

generate_opt_script() {
    local target_dir=$1
    local spin_folder_name=$2
    local uhf=$3
    local parent_name=$(basename "$(dirname "$target_dir")")
    local script_path="$target_dir/submit_opt.sh"

    cat > "$script_path" << OPT_EOF
$(slurm_header "OPT_${parent_name}_${spin_folder_name}" "$WALLTIME_OPT")

# ============================================================================
# Stage 1: UMA Geometry Optimization
# Molecule: $parent_name | Spin: $spin_folder_name (UHF=$uhf, charge=$CHARGE)
# ============================================================================

echo "=========================================="
echo "Stage 1: UMA Geometry Optimization"
echo "Molecule: $parent_name / $spin_folder_name"
echo "Date:     \$(date)"
echo "GPU:      \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "=========================================="

if [ ! -f "CONF.xyz" ]; then
    echo "ERROR: CONF.xyz not found!"
    exit 1
fi

cat > opt.toml << TOML_EOF
runtype = "ancopt"

[[calculation.level]]
method = "uma"
device = "cuda"
model_path = "$UMA_MODEL"
maxdispl = 0.15
maxcycle = 500
TOML_EOF

$CREST_EXECUTABLE CONF.xyz \\
  --input opt.toml \\
  -T $CPUS \\
  -chrg $CHARGE \\
  -uhf $uhf \\
  > opt_output.log 2>&1
RC=\$?

if [ \$RC -eq 0 ] && [ -f "crestopt.xyz" ]; then
    cp crestopt.xyz OPT.xyz
    echo "SUCCESS: Optimization converged"
    grep -E 'total energy|gradient norm' opt_output.log | tail -2
else
    echo "FAILED (exit=\$RC)"
    tail -20 opt_output.log
    exit \$RC
fi

echo "Finished: \$(date)"
OPT_EOF

    chmod +x "$script_path"
    echo "$script_path"
}

generate_conf_script() {
    local target_dir=$1
    local spin_folder_name=$2
    local uhf=$3
    local parent_name=$(basename "$(dirname "$target_dir")")
    local script_path="$target_dir/submit_conf.sh"

    cat > "$script_path" << CONF_EOF
$(slurm_header "CONF_${parent_name}_${spin_folder_name}" "$WALLTIME_CONF")

# ============================================================================
# Stage 2: UMA Conformer Search (iMTD-GC)
# Molecule: $parent_name | Spin: $spin_folder_name (UHF=$uhf, charge=$CHARGE)
# ============================================================================

echo "=========================================="
echo "Stage 2: UMA Conformer Search"
echo "Molecule: $parent_name / $spin_folder_name"
echo "Date:     \$(date)"
echo "GPU:      \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "=========================================="

# Use pre-optimized geometry if available, else fall back to CONF.xyz
if [ -f "OPT.xyz" ]; then
    INPUT_XYZ="OPT.xyz"
    echo "Using pre-optimized geometry: OPT.xyz"
else
    INPUT_XYZ="CONF.xyz"
    echo "WARNING: OPT.xyz not found, using raw CONF.xyz"
fi

cat > crest.toml << 'TOML_EOF'
runtype = "imtd-gc"

[[calculation.level]]
method = "uma"
device = "cuda"
model_path = "$UMA_MODEL"
TOML_EOF

echo ""
echo "Running CREST conformer search with UMA..."
echo ""

$CREST_EXECUTABLE \$INPUT_XYZ \\
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
    tail -20 crest_output.log | grep -E 'ensemble|conformers|E lowest|Wall Time|runtime'
else
    echo "FAILED (exit=\$RC)"
    tail -20 crest_output.log
    exit \$RC
fi

echo "Finished: \$(date)"
CONF_EOF

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

        opt_path=$(generate_opt_script "$folder" "$spin_name" "$uhf")
        conf_path=$(generate_conf_script "$folder" "$spin_name" "$uhf")
        log_info "  -> $opt_path"
        log_info "  -> $conf_path"
        generated=$((generated + 1))
    done

    echo ""
    log_info "Generated $generated x 2 submission scripts (opt + conf)"
    log_info "Submit optimization:     $0 submit-opt $root_dir"
    log_info "Submit conformer search: $0 submit-conf $root_dir"
    log_info "Submit both (chained):   $0 submit-all $root_dir"
}

submit_opt_jobs() {
    local root_dir=${1:-.}
    local dry_run=${2:-false}

    local scripts=($(find "$root_dir" -name "submit_opt.sh" -type f | sort))
    if [ ${#scripts[@]} -eq 0 ]; then
        log_warn "No opt scripts found! Run '$0 generate' first."
        return 1
    fi

    log_info "Submitting ${#scripts[@]} optimization job(s)"
    [ "$dry_run" = true ] && log_warn "DRY RUN"
    echo ""

    local submitted=0 skipped=0
    for script in "${scripts[@]}"; do
        local dir=$(dirname "$script")
        local spin_name=$(basename "$dir")
        local parent_name=$(basename "$(dirname "$dir")")

        if [ -f "$dir/OPT.xyz" ]; then
            log_warn "  $parent_name/$spin_name — SKIP (OPT.xyz exists)"
            skipped=$((skipped + 1))
            continue
        fi

        if [ "$dry_run" = false ]; then
            cd "$dir"
            job_id=$(sbatch submit_opt.sh 2>&1)
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
    log_info "Submitted: $submitted | Skipped: $skipped"
}

submit_conf_jobs() {
    local root_dir=${1:-.}
    local dry_run=${2:-false}

    local scripts=($(find "$root_dir" -name "submit_conf.sh" -type f | sort))
    if [ ${#scripts[@]} -eq 0 ]; then
        log_warn "No conf scripts found! Run '$0 generate' first."
        return 1
    fi

    log_info "Submitting ${#scripts[@]} conformer search job(s)"
    [ "$dry_run" = true ] && log_warn "DRY RUN"
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
            job_id=$(sbatch submit_conf.sh 2>&1)
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
    log_info "Submitted: $submitted | Skipped: $skipped"
}

submit_all_chained() {
    local root_dir=${1:-.}
    local dry_run=${2:-false}

    local opt_scripts=($(find "$root_dir" -name "submit_opt.sh" -type f | sort))
    if [ ${#opt_scripts[@]} -eq 0 ]; then
        log_warn "No scripts found! Run '$0 generate' first."
        return 1
    fi

    log_info "Submitting ${#opt_scripts[@]} chained jobs (opt → conf)"
    [ "$dry_run" = true ] && log_warn "DRY RUN"
    echo ""

    local submitted=0 skipped=0
    for opt_script in "${opt_scripts[@]}"; do
        local dir=$(dirname "$opt_script")
        local spin_name=$(basename "$dir")
        local parent_name=$(basename "$(dirname "$dir")")

        if [ -f "$dir/crest_conformers.xyz" ]; then
            log_warn "  $parent_name/$spin_name — SKIP (already completed)"
            skipped=$((skipped + 1))
            continue
        fi

        if [ "$dry_run" = false ]; then
            cd "$dir"
            # Submit opt job
            opt_jobid=$(sbatch --parsable submit_opt.sh 2>&1)
            if [ $? -ne 0 ]; then
                log_error "  $parent_name/$spin_name — OPT submit failed: $opt_jobid"
                cd - > /dev/null
                continue
            fi
            # Submit conf job with dependency on opt
            conf_jobid=$(sbatch --parsable --dependency=afterok:$opt_jobid submit_conf.sh 2>&1)
            if [ $? -eq 0 ]; then
                log_info "  $parent_name/$spin_name — OPT:$opt_jobid → CONF:$conf_jobid"
                submitted=$((submitted + 1))
            else
                log_error "  $parent_name/$spin_name — CONF submit failed: $conf_jobid"
            fi
            cd - > /dev/null
        else
            log_info "  $parent_name/$spin_name — would submit OPT → CONF"
            submitted=$((submitted + 1))
        fi
    done

    echo ""
    log_info "Submitted: $submitted chained pairs | Skipped: $skipped"
}

check_job_status() {
    local root_dir=${1:-.}
    local folders=($(find_crest_folders "$root_dir"))
    if [ ${#folders[@]} -eq 0 ]; then
        log_warn "No folders found!"
        return 1
    fi

    echo ""
    printf "%-50s %-12s %s\n" "JOB" "OPT" "CONF"
    printf "%-50s %-12s %s\n" "---" "---" "----"

    local opt_done=0 conf_done=0 total=0

    for folder in "${folders[@]}"; do
        local spin_name=$(basename "$folder")
        local parent_name=$(basename "$(dirname "$folder")")
        local label="$parent_name/$spin_name"
        total=$((total + 1))

        # Opt status
        local opt_status="--"
        if [ -f "$folder/OPT.xyz" ]; then
            opt_status="${GREEN}DONE${NC}"
            opt_done=$((opt_done + 1))
        elif [ -f "$folder/opt_output.log" ]; then
            if grep -q "Geometry successfully optimized" "$folder/opt_output.log" 2>/dev/null; then
                opt_status="${GREEN}DONE${NC}"
                opt_done=$((opt_done + 1))
            elif grep -q "terminated with failures" "$folder/opt_output.log" 2>/dev/null; then
                opt_status="${RED}FAIL${NC}"
            else
                opt_status="${YELLOW}RUN${NC}"
            fi
        fi

        # Conf status
        local conf_status="--"
        if [ -f "$folder/crest_conformers.xyz" ]; then
            local nat=$(head -1 "$folder/crest_conformers.xyz")
            local nconf=$(grep -c "^${nat}$" "$folder/crest_conformers.xyz" 2>/dev/null || echo "?")
            conf_status="${GREEN}DONE${NC}($nconf)"
            conf_done=$((conf_done + 1))
        elif [ -f "$folder/crest_output.log" ]; then
            if grep -q "CREST terminated normally" "$folder/crest_output.log" 2>/dev/null; then
                conf_status="${GREEN}DONE${NC}"
                conf_done=$((conf_done + 1))
            elif grep -q "terminated with failures" "$folder/crest_output.log" 2>/dev/null; then
                conf_status="${RED}FAIL${NC}"
            else
                conf_status="${YELLOW}RUN${NC}"
            fi
        fi

        printf "%-50s %-12b %b\n" "$label" "$opt_status" "$conf_status"
    done

    echo ""
    echo "Total: $total | Opt done: $opt_done | Conf done: $conf_done"
}

show_help() {
    cat << 'EOF'
CREST-MLIP Batch Job Manager (Snellius) — Two-Stage Pipeline

Usage:  ./crest_batch_submit.sh <command> [directory]

Commands:
  generate [dir]     Generate SLURM scripts (opt + conf) for all folders
  submit-opt [dir]   Submit optimization jobs only
  submit-conf [dir]  Submit conformer search jobs only
  submit-all [dir]   Submit both chained: opt → conf (SLURM dependency)
  dry-run [dir]      Show what submit-all would do
  status [dir]       Check status of all jobs
  list [dir]         List all folders that would be processed
  help               Show this help

Pipeline:
  Stage 1 (submit_opt.sh):  CREST ancopt on CONF.xyz → OPT.xyz
  Stage 2 (submit_conf.sh): CREST iMTD-GC on OPT.xyz → crest_conformers.xyz

  submit-all chains them with --dependency=afterok so Stage 2
  only starts after Stage 1 succeeds.

Folder structure expected:
  molecule_name/
  ├── S1/CONF.xyz    (singlet, UHF=0)
  ├── S3/CONF.xyz    (triplet, UHF=2)
  └── S5/CONF.xyz    (quintet, UHF=4)
EOF
}

# ============================================================================
# MAIN
# ============================================================================

case "${1:-help}" in
    generate)    generate_all_scripts "${2:-.}" ;;
    submit-opt)  submit_opt_jobs "${2:-.}" false ;;
    submit-conf) submit_conf_jobs "${2:-.}" false ;;
    submit-all)  submit_all_chained "${2:-.}" false ;;
    dry-run)     submit_all_chained "${2:-.}" true ;;
    status)      check_job_status "${2:-.}" ;;
    list)        list_folders "${2:-.}" ;;
    help|--help|-h) show_help ;;
    *) log_error "Unknown command: $1"; show_help; exit 1 ;;
esac
