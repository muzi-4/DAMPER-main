#!/usr/bin/env bash
set -Eeuo pipefail

#############################################
# Config
#############################################
TEMPERATURE=${TEMPERATURE:-0.1}
ALPHA=${ALPHA:-0.5}
BETA=${BETA:-0.1}
GAMMA1=${GAMMA1:-0.9}
GAMMA2=${GAMMA2:-0.9}
EPSILON=${EPSILON:-50}

GPU_ID=${GPU_ID:-0}
DEVICE="cuda:0"

# Conda envs
ENV_DAMPER=${ENV_DAMPER:-"damper"}
ENV_LF=${ENV_LF:-"llama-factory"}

# Datasets
DATASET1=${DATASET1:-"Pri_DDXPlus"}
DATASET2=${DATASET2:-"Pri_SLJA"}

# LLaMA-Factory settings
CONFIG_TRAIN=${CONFIG_TRAIN:-"qwen2.5_1.5b_lora_dpo.yaml"}
CONFIG_EXPORT=${CONFIG_EXPORT:-"qwen2.5_1.5b_lora_dpo_merge.yaml"}

# log
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR=${LOG_DIR:-"./logs/script/${TIMESTAMP}"}
mkdir -p "$LOG_DIR"

#############################################
# Helpers
#############################################
log() { printf "[%(%F %T)T] %s\n" -1 "$*"; }
die() { log "ERROR: $*"; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"; }

# params
FROM_STEP=1
END_STEP=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-step)
      FROM_STEP="$2"; shift 2;;
    --end-step)
      END_STEP="$2"; shift 2;;
    --dry-run)
      DRY_RUN=1; shift;;
    -t|--temperature)
      TEMPERATURE="$2"; shift 2;;
    -a|--alpha)
      ALPHA="$2"; shift 2;;
    -b|--beta)
      BETA="$2"; shift 2;;
    -g1|--gamma1)
      GAMMA1="$2"; shift 2;;
    -g2|--gamma2)
      GAMMA2="$2"; shift 2;;
    -e|--epsilon)
      EPSILON="$2"; shift 2;;
    -d|--device)
      GPU_ID="$2"; shift 2;;
    *)
      die "Unknown param: $1";;
  esac
done

NEW_CONFIG_TRAIN=${NEW_CONFIG_TRAIN:-"qwen2.5_1.5b_lora_dpo_temperature_${TEMPERATURE}_alpha_${ALPHA}_beta_${BETA}.yaml"}
NEW_CONFIG_EXPORT=${NEW_CONFIG_EXPORT:-"qwen2.5_1.5b_lora_dpo_temperature_${TEMPERATURE}_alpha_${ALPHA}_beta_${BETA}_merge.yaml"}

SAVE_DIR=${SAVE_DIR:-"./saved_models/qwen2.5_1.5b_dpo/temperature_${TEMPERATURE}/alpha_${ALPHA}/beta_${BETA}"}
OUT_LORA_DIR="${SAVE_DIR}/qwen2.5_1.5b_lora_dpo"
OUT_MERGE_DIR="${SAVE_DIR}/qwen2.5_1.5b_lora_dpo_merge"

run_in_env() {
  local env="$1"; shift
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
    conda run -n "$env" "$@"
}

#############################################
# Pre-flight checks
#############################################
need_cmd conda
need_cmd sed

[[ -f "$CONFIG_TRAIN" ]]  || die "Training configuration not found: $CONFIG_TRAIN"
[[ -f "$CONFIG_EXPORT" ]] || die "Export configuration not found: $CONFIG_EXPORT"

log "Parameter summary: TEMPERATURE=$TEMPERATURE, ALPHA=$ALPHA, BETA=$BETA, GAMMA1=$GAMMA1, GAMMA2=$GAMMA2, GPU_ID=$GPU_ID"
log "DP settings: epsilon=$EPSILON"
log "Environments: damper=$ENV_DAMPER, llama-factory=$ENV_LF"
log "Datasets: ${DATASET1}, ${DATASET2}"
log "Save directory: $SAVE_DIR"
log "from-step=$FROM_STEP, dry-run=$DRY_RUN"
log "----------------------------------------"

do_or_echo() {
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "+ $*"
  else
    eval "$@"
  fi
}

#############################################
# Step 1-6: damper environment
#############################################
step=1
if (( step >= FROM_STEP && step <= END_STEP )); then
  log "Step $step: Training roberta lora (env: $ENV_DAMPER)"
  cmd="run_in_env \"$ENV_DAMPER\" python train_roberta_lora.py \
    --temperature $TEMPERATURE \
    --device $DEVICE | tee \"$LOG_DIR/step${step}_roberta_lora.log\""
  do_or_echo "$cmd"
fi

step=2
if (( step >= FROM_STEP && step <= END_STEP )); then
  log "Step $step: Domain prototype clustering (env: $ENV_DAMPER)"
  cmd="run_in_env \"$ENV_DAMPER\" python domain_prototype.py \
    --temperature $TEMPERATURE \
    --device $DEVICE | tee \"$LOG_DIR/step${step}_domain_prototype.log\""
  do_or_echo "$cmd"
fi

step=3
if (( step >= FROM_STEP && step <= END_STEP )); then
  log "Step $step: Candidate dataset construction (env: $ENV_DAMPER)"
  cmd="run_in_env \"$ENV_DAMPER\" python candidate_dataset_construction_parallel.py \
    --device $DEVICE | tee \"$LOG_DIR/step${step}_candidate_dataset.log\""
  do_or_echo "$cmd"
fi

step=4
if (( step >= FROM_STEP && step <= END_STEP )); then
  log "Step $step: Batch merge (env: $ENV_DAMPER)"
  cmd="run_in_env \"$ENV_DAMPER\" python batch_merge.py | tee \"$LOG_DIR/step${step}_batch_merge.log\""
  do_or_echo "$cmd"
fi

step=5
if (( step >= FROM_STEP && step <= END_STEP )); then
  log "Step $step: Preference dataset construction (env: $ENV_DAMPER)"
  cmd="run_in_env \"$ENV_DAMPER\" python preference_dataset_construction.py \
    --temperature $TEMPERATURE \
    --preference_alpha $ALPHA \
    --device $DEVICE | tee \"$LOG_DIR/step${step}_pref_dataset.log\""
  do_or_echo "$cmd"
fi

step=6
if (( step >= FROM_STEP && step <= END_STEP )); then
  log "Step $step: LLaMA-Factory dataset preparation (env: $ENV_DAMPER)"
  cmd="run_in_env \"$ENV_DAMPER\" python dpo_dataset_pre.py \
    --temperature $TEMPERATURE \
    --preference_alpha $ALPHA | tee \"$LOG_DIR/step${step}_dpo_dataset_pre.log\""
  do_or_echo "$cmd"
fi

#############################################
# Step 7-8: llama-factory environment
#############################################
step=7
if (( step >= FROM_STEP && step <= END_STEP )); then
  log "Step $step: DPO training configuration generation and training (env: $ENV_LF)"
  do_or_echo "cp \"$CONFIG_TRAIN\" \"$NEW_CONFIG_TRAIN\""
  do_or_echo "sed -i \"s/^dataset:.*/dataset: privacy_dpo_temperature_${TEMPERATURE}_alpha_${ALPHA}_json/\" \"$NEW_CONFIG_TRAIN\""
  do_or_echo "sed -i \"s/^pref_beta:.*/pref_beta: ${BETA}/\" \"$NEW_CONFIG_TRAIN\""
  do_or_echo "sed -i \"s|^output_dir:.*|output_dir: ${OUT_LORA_DIR}|\" \"$NEW_CONFIG_TRAIN\""

  cmd="run_in_env \"$ENV_LF\" llamafactory-cli train \"$NEW_CONFIG_TRAIN\" \
    | tee \"$LOG_DIR/step${step}_lf_train.log\""
  do_or_echo "$cmd"
fi

step=8
if (( step >= FROM_STEP && step <= END_STEP )); then
  log "Step $step: LoRA export and merge (env: $ENV_LF)"
  do_or_echo "cp \"$CONFIG_EXPORT\" \"$NEW_CONFIG_EXPORT\""
  do_or_echo "sed -i \"s|^adapter_name_or_path:.*|adapter_name_or_path: ${OUT_LORA_DIR}|\" \"$NEW_CONFIG_EXPORT\""
  do_or_echo "sed -i \"s|^export_dir:.*|export_dir: ${OUT_MERGE_DIR}|\" \"$NEW_CONFIG_EXPORT\""

  cmd="run_in_env \"$ENV_LF\" llamafactory-cli export \"$NEW_CONFIG_EXPORT\" \
    | tee \"$LOG_DIR/step${step}_lf_export.log\""
  do_or_echo "$cmd"
fi

rm -f "$NEW_CONFIG_TRAIN" "$NEW_CONFIG_EXPORT"

#############################################
# Step 9-13: Back to damper environment
#############################################
step=9
if (( step >= FROM_STEP && step <= END_STEP )); then
  log "Step $step: Ours rewrite ($DATASET1, env: $ENV_DAMPER)"
  cmd="run_in_env \"$ENV_DAMPER\" python rewrite_ours.py \
    --dataset_name $DATASET1 \
    --temperature $TEMPERATURE \
    --preference_alpha $ALPHA \
    --beta $BETA \
    --similarity_threshold $GAMMA1 \
    --epsilon $EPSILON \
    --device $DEVICE | tee \"$LOG_DIR/step${step}_rewrite_${DATASET1}.log\""
  do_or_echo "$cmd"
fi

step=10
if (( step >= FROM_STEP && step <= END_STEP )); then
  log "Step $step: Ours rewrite ($DATASET2, env: $ENV_DAMPER)"
  cmd="run_in_env \"$ENV_DAMPER\" python rewrite_ours.py \
    --dataset_name $DATASET2 \
    --temperature $TEMPERATURE \
    --preference_alpha $ALPHA \
    --beta $BETA \
    --similarity_threshold $GAMMA2 \
    --epsilon $EPSILON \
    --device $DEVICE | tee -a \"$LOG_DIR/step${step}_rewrite_${DATASET2}.log\""
  do_or_echo "$cmd"
fi

step=11
if (( step >= FROM_STEP && step <= END_STEP )); then
  log "Step $step: Evaluation (env: $ENV_DAMPER)"
  cmd="run_in_env \"$ENV_DAMPER\" python evaluation_mixture_pre.py \
    --temperature $TEMPERATURE \
    --preference_alpha $ALPHA \
    --beta $BETA \
    --similarity_threshold_medical $GAMMA1 \
    --similarity_threshold_legal $GAMMA2 \
    --epsilon $EPSILON | tee \"$LOG_DIR/step${step}_eval_${DATASET1}.log\""
  do_or_echo "$cmd"
fi

step=12
if (( step >= FROM_STEP && step <= END_STEP )); then
  log "Step $step: Evaluation (env: $ENV_DAMPER)"
  cmd="run_in_env \"$ENV_DAMPER\" python evaluation_ours.py \
    --temperature $TEMPERATURE \
    --preference_alpha $ALPHA \
    --beta $BETA \
    --similarity_threshold_medical $GAMMA1 \
    --similarity_threshold_legal $GAMMA2 \
    --epsilon $EPSILON \
    --device $DEVICE | tee \"$LOG_DIR/step${step}_eval_${DATASET1}.log\""
  do_or_echo "$cmd"
fi

step=13
if (( step >= FROM_STEP && step <= END_STEP )); then
  log "Step $step: Evaluation LLM-J (env: $ENV_DAMPER)"
  cmd="run_in_env \"$ENV_DAMPER\" python evaluation_ours_llmj.py \
    --temperature $TEMPERATURE \
    --preference_alpha $ALPHA \
    --beta $BETA \
    --similarity_threshold_medical $GAMMA1 \
    --similarity_threshold_legal $GAMMA2 \
    --epsilon $EPSILON \
    --device $DEVICE | tee \"$LOG_DIR/step${step}_eval_llmj_${DATASET1}.log\""
  do_or_echo "$cmd"
fi

log "All steps completed ✅"
