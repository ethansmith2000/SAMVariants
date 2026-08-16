#!/bin/bash
# =============================================================================
# GPT Pretraining Optimizer Comparison
#
# Clean comparison of:
#   - AdamW baseline
#   - Muon baseline
#   - Hybrid SAM perturbation optimizer
#
# Keep the same config-driven sweep format so each run is a small JSON override.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="/home/ethan/SAMVariants"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_gpt.py"
LOG_DIR="${SCRIPT_DIR}/slurm_logs"
CONFIG_DIR="${SCRIPT_DIR}/sweep_configs"
OUTPUT_ROOT="model-output/llm_sam_variants"
WANDB_PROJECT="sam-variants-llm"

mkdir -p "${LOG_DIR}" "${CONFIG_DIR}"

# Common settings
WD="0.01"
NUM_GPUS=1
NUM_WARMUP_STEPS="100"
LR_SCHEDULER_TYPE="constant"
ADAM_BETA1="0.9"
ADAM_BETA2="0.95"
ADAM_EPSILON="1.0e-8"
MUON_LR="6.0e-3"
MUON_BETA1="0.95"
MUON_NS_STEPS=6
MUON_MAX_DIM=16384
# "balanced": per-param unit directions scaled by sqrt(numel_p/total), total
# norm = rho. A raw "global" norm lets adam-fallback params (embeddings) absorb
# >99% of the budget when mixed with muon directions — don't use it for mixed
# ascent modes.
HYBRID_PERTURBATION_NORM="balanced"
HYBRID_MUON_FALLBACK_ASCENT="skip"
# rho > 0 = MSAM-style lookahead (perturb along the update direction);
# rho < 0 = classic SAM-style ascent. Sweep both signs — it's the cheapest,
# most decisive ablation for which mechanism is doing the work.
HYBRID_RHOS=("-1.0" "-0.3" "-0.1" "0.1" "0.3" "1.0")
# Matches MSAM's rho=0-during-warmup recommendation
HYBRID_PERTURBATION_START_STEP=100

# Model config: "hidden_size depth n_head lr batch_size max_train_steps"
MODEL_CONFIG="1024 12 8 4.0e-4 32 250000"
# MODEL_CONFIG="2048 24 16 2.0e-4 16 190000"

# Parse model config
read -r HIDDEN_SIZE DEPTH N_HEAD LR BATCH_SIZE MAX_TRAIN_STEPS <<< "${MODEL_CONFIG}"

submit_job() {
  local job_name="$1"
  local cfg_file="$2"

  sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=${NUM_GPUS}
#SBATCH --cpus-per-task=20
#SBATCH --job-name=${job_name}
#SBATCH --partition=queue1gpu
#SBATCH --time=6-09:59:59
#SBATCH --output=${LOG_DIR}/${job_name}-%j.out

cd ${SCRIPT_DIR}
source /home/ethan/leo-train-template/.venv/bin/activate

export TRITON_CACHE_DIR="/home/ethan/job_triton/triton_cache_\${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="/home/ethan/job_triton/inductor_cache_\${SLURM_JOB_ID}"
mkdir -p "\${TRITON_CACHE_DIR}" "\${TORCHINDUCTOR_CACHE_DIR}"

# NCCL settings for multi-GPU
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Use unique port per job to avoid conflicts when multiple jobs share a node
export MASTER_PORT=\$((29500 + (\${SLURM_JOB_ID} % 10000)))

if [[ ${NUM_GPUS} -gt 1 ]]; then
  accelerate launch \\
    --num_processes=${NUM_GPUS} \\
    --num_machines=1 \\
    --main_process_port=\${MASTER_PORT} \\
    --mixed_precision=bf16 \\
    --dynamo_backend=no \\
    ${TRAIN_SCRIPT} --override_json "${cfg_file}"
else
  python ${TRAIN_SCRIPT} --override_json "${cfg_file}"
fi
EOF
}

# Format:
#   "label mode rho ascent descent muon_nesterov normalize_perturbation"
#
# For AdamW and Muon, the hybrid-only fields are still written to JSON but ignored
# by train_gpt.py for those modes. Keeping every config structurally identical
# makes it easier to diff runs after launch.
optimizer_configs=(
  # Baselines. muon-nesterov is the honest control for rho>0 (MSAM-sign)
  # perturbations: those are lookahead-flavored, and nesterov is the cheap
  # built-in version of lookahead.
  # "adamw adamw 0.0 muon adam false true"
  "muon muon 0.0 muon muon false true"
  "muon-nesterov muon 0.0 muon muon true true"
)

for rho in "${HYBRID_RHOS[@]}"; do
  rho_label="${rho//./p}"
  rho_label="${rho_label//-/n}"
  # optimizer_configs+=("msam-adamw-rho${rho_label} hybrid_sam ${rho} momentum adam false true")
  # optimizer_configs+=("adam-sam-adam-rho${rho_label} hybrid_sam ${rho} adam adam false true")
  # optimizer_configs+=("hybrid-sam-muon-adam-rho${rho_label} hybrid_sam ${rho} muon adam false true")
  optimizer_configs+=("hybrid-sam-adam-muon-rho${rho_label} hybrid_sam ${rho} adam muon false true")
  optimizer_configs+=("hybrid-sam-muon-muon-rho${rho_label} hybrid_sam ${rho} muon muon false true")
done

for config in "${optimizer_configs[@]}"; do
  read -r label mode rho ascent descent muon_nesterov normalize_perturbation <<< "${config}"
  job_name="${label}-geglu-d${DEPTH}-adamlr${LR}-muonlr${MUON_LR}"
  cfg_file="${CONFIG_DIR}/${job_name}.json"

  cat > "${cfg_file}" <<JSON
{
  "mode": "${mode}",
  "learning_rate": ${LR},
  "weight_decay": ${WD},
  "adam_beta1": ${ADAM_BETA1},
  "adam_beta2": ${ADAM_BETA2},
  "adam_epsilon": ${ADAM_EPSILON},
  "muon_learning_rate": ${MUON_LR},
  "muon_beta1": ${MUON_BETA1},
  "muon_ns_steps": ${MUON_NS_STEPS},
  "muon_nesterov": ${muon_nesterov},
  "muon_max_dim": ${MUON_MAX_DIM},
  "hybrid_sam_rho": ${rho},
  "hybrid_sam_ascent": "${ascent}",
  "hybrid_sam_descent": "${descent}",
  "hybrid_sam_normalize_perturbation": ${normalize_perturbation},
  "hybrid_sam_perturbation_norm": "${HYBRID_PERTURBATION_NORM}",
  "hybrid_sam_muon_fallback_ascent": "${HYBRID_MUON_FALLBACK_ASCENT}",
  "hybrid_sam_perturbation_start_step": ${HYBRID_PERTURBATION_START_STEP},
  "hybrid_sam_track_stats": true,
  "eval_perturbed": true,
  "per_device_train_batch_size": ${BATCH_SIZE},
  "max_train_steps": ${MAX_TRAIN_STEPS},
  "lr_scheduler_type": "${LR_SCHEDULER_TYPE}",
  "base_output_dir": "${OUTPUT_ROOT}",
  "hidden_size": ${HIDDEN_SIZE},
  "depth": ${DEPTH},
  "n_head": ${N_HEAD},
  "ffn_type": "geglu",
  "num_warmup_steps": ${NUM_WARMUP_STEPS},
  "wandb_project": "${WANDB_PROJECT}"
}
JSON

  submit_job "${job_name}" "${cfg_file}"
  echo "Submitted: ${job_name}"
done
