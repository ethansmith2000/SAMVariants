import os
import tempfile
import inspect as _inspect
from pathlib import Path

# Ensure Hugging Face datasets cache is set early to a user-writable path.
os.environ.setdefault("HF_HOME", "/fsx/scratch/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(os.environ["HF_HOME"], "datasets"))

# Make Triton and TorchInductor use stable, user-writable cache/tmp dirs on the cluster
os.environ.setdefault("TRITON_CACHE_DIR", "/home/ethan/.triton/cache")
os.environ.setdefault("TMPDIR", "/home/ethan/tmp")
tempfile.tempdir = os.environ["TMPDIR"]

# When using datasets.map(num_proc=...) we rely on multiprocessing for parallelism,
# so we disable internal tokenizer threading to avoid oversubscription.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _safe_getsourcelines(obj):
    """
    Work around Triton+Python 3.11 \"source code not available\" issues by
    returning a dummy source snippet instead of raising, so torch.compile
    can proceed. This is only meant for tooling and does not affect numerics.
    """
    try:
        return _inspect._orig_getsourcelines(obj)  # type: ignore[attr-defined]
    except OSError as e:
        if "source code not available" in str(e):
            return ["# source code not available\n"], 0
        raise


if not hasattr(_inspect, "_orig_getsourcelines"):
    _inspect._orig_getsourcelines = _inspect.getsourcelines  # type: ignore[attr-defined]
    _inspect.getsourcelines = _safe_getsourcelines


import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.v8_api_enabled = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # 
torch.backends.cuda.allow_tensor_float_32 = True


# torch._inductor.config.triton.cudagraphs = True   # or False if capture causes overhead
# torch._inductor.config.use_mixed_mm = True        # enables faster matmul codegen
# torch._inductor.config.triton.cudagraphs = True
# torch._inductor.config.triton.cudagraph_trees = True  # More aggressive cudagraph usage
# torch._inductor.config.triton.autotune_pointwise = True
# # torch._inductor.config.triton.dense_indexing = True
# torch._inductor.config.triton.max_tiles = 8  # Increase tiling options
# torch._inductor.config.aggressive_fusion = True
# torch._inductor.config.pattern_matcher = True
# torch._inductor.config.permute_fusion = True
# torch._inductor.config.max_autotune = True
# torch._inductor.config.max_autotune_gemm = True

torch.set_num_threads(12)
torch.set_num_interop_threads(2)

# torch._inductor.config.autotune_in_subproc = True            # instead of exporting TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC
# torch._inductor.config.autotune_multi_device = True          # mirrors TORCHINDUCTOR_AUTOTUNE_MULTI_DEVICE
# # torch._inductor.config.max_autotune_gemm_search_space = "EXHAUSTIVE"



import argparse
import hashlib
import json
import logging
import math
import shutil
from itertools import chain
from typing import Any

import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)
from types import SimpleNamespace

import time
from hybrid_sam import HybridSAM
from muon import Muon
from transformer import Transformer

logger = get_logger(__name__)


def _load_overrides():
    """
    Lightweight override loader to keep the script largely argument-free.
    Priority: CLI flag --override_json, then env OVERRIDE_JSON.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--override_json", type=str, default=os.environ.get("OVERRIDE_JSON"))
    parsed, _ = parser.parse_known_args()
    override_path = parsed.override_json
    if override_path:
        with open(override_path, "r") as f:
            data = json.load(f)
        # Stash metadata so downstream code can use the sweep entry name.
        data["_override_json_path"] = override_path
        data["_override_name"] = Path(override_path).stem
        return data
    return {}


def _build_run_name(args: dict[str, Any]) -> str:
    override_name = args.get("_override_name")
    if override_name:
        return override_name

    parts = [str(args["mode"])]
    if str(args["mode"]).lower() in {"hybrid_sam", "adam_muon_perturb"}:
        parts += [
            args["hybrid_sam_ascent"],
            args["hybrid_sam_descent"],
            f"rho{args['hybrid_sam_rho']}",
        ]
    parts.append(time.strftime("%m%d_%H%M%S"))
    return "-".join(parts)


def _config_fingerprint(args: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in args.items()
        if not key.startswith("_") and key not in {"run_name", "output_dir", "wandb_run_name"}
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _build_optimizer(args, model, optimizer_grouped_parameters):
    mode = str(args.mode).lower()

    if mode in {"adamw", "adam"}:
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
        )

    if mode == "muon":
        return Muon(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            muon_lr=args.muon_learning_rate,
            beta1=args.muon_beta1,
            beta2=args.adam_beta2,
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
            ns_steps=args.muon_ns_steps,
            nesterov=args.muon_nesterov,
            muon_max_dim=args.muon_max_dim,
        )

    if mode in {"hybrid_sam", "adam_muon_perturb"}:
        return HybridSAM(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            muon_lr=args.muon_learning_rate,
            rho=args.hybrid_sam_rho,
            ascent=args.hybrid_sam_ascent,
            descent=args.hybrid_sam_descent,
            beta1=args.muon_beta1,
            beta2=args.adam_beta2,
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
            ns_steps=args.muon_ns_steps,
            nesterov=args.muon_nesterov,
            ascent_beta1=args.hybrid_sam_ascent_beta1,
            perturbation_start_step=args.hybrid_sam_perturbation_start_step,
            normalize_perturbation=args.hybrid_sam_normalize_perturbation,
            perturbation_norm=args.hybrid_sam_perturbation_norm,
            perturbation_scale=args.hybrid_sam_perturbation_scale,
            muon_max_dim=args.muon_max_dim,
            muon_fallback_ascent=args.hybrid_sam_muon_fallback_ascent,
            perturb_muon_eligible_only=args.hybrid_sam_perturb_muon_eligible_only,
            track_stats=args.hybrid_sam_track_stats,
        )

    raise ValueError(f"Unsupported optimizer mode: {args.mode}")


def _prune_checkpoints(output_dir, keep_last_n):
    """Keep only the newest `keep_last_n` step_* checkpoints.

    Each checkpoint is model (~1GB) + optimizer state (~2-3GB: exp_avg,
    exp_avg_sq, and HybridSAM's cached perturbation), so an unpruned sweep
    fills hundreds of GB. Auto-resume only ever reads the newest one.
    """
    if not keep_last_n or keep_last_n < 1 or not os.path.isdir(output_dir):
        return
    ckpts = sorted(
        (int(e.name.split("_")[1]), e.path)
        for e in os.scandir(output_dir)
        if e.is_dir() and e.name.startswith("step_") and e.name.split("_")[1].isdigit()
    )
    for _, path in ckpts[:-keep_last_n]:
        shutil.rmtree(path, ignore_errors=True)


def _unwrap_optimizer(optimizer):
    return getattr(optimizer, "optimizer", optimizer)


def _run_validation(model, eval_dataloader, accelerator, args):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            tokens = batch["input_ids"].to(accelerator.device).long()
            input_ids, targets = tokens[:, :-1], tokens[:, 1:]
            loss, logits = model(input_ids=input_ids, targets=targets)
        losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_train_batch_size)))
        if args.num_validation_batches is not None and step >= args.num_validation_batches:
            break
    return torch.mean(torch.cat(losses))


def _get_optimizer_lrs(optimizer):
    opt = _unwrap_optimizer(optimizer)
    return [group["lr"] for group in opt.param_groups]



def main():
    override_args = _load_overrides()

    args = {
        "num_validation_batches": 25,
        "validate_every": 1000,
        "dataset_name": "Skylion007/openwebtext",  # namespaced id required by newer `datasets`
        "dataset_config_name": None,
        # "dataset_name": "wikitext",
        # "dataset_config_name": "wikitext-103-v1",
        "train_file": None,
        "validation_file": None,
        "validation_split_percentage": 5,
        "model_name_or_path": "openai-community/gpt2-medium",
        # "model_name_or_path": "openai-community/gpt2",
        "config_name": None,
        "tokenizer_name": None,
        "use_slow_tokenizer": False,
        "per_device_train_batch_size": 32,

        "num_train_epochs": 2,
        # "max_train_steps": 500_000,
        "max_train_steps": 100_000,
        # "max_train_steps": 125_000,
        "gradient_accumulation_steps": 1,
        "lr_scheduler_type": "linear",
        "num_warmup_steps": 100,
        "seed": 123,
        "model_type": None,
        "block_size": 1024,
        "preprocessing_num_workers": 180,
        "overwrite_cache": False,
        "no_keep_linebreaks": False,
        "trust_remote_code": False,
        "checkpointing_steps": None,
        # keep only the newest N step_* checkpoints (auto-resume needs 1)
        "keep_last_n_checkpoints": 1,
        "discard_checkpoints_at_end": True,
        "resume_from_checkpoint": None,
        "with_tracking": True,
        "report_to": "wandb",
        "low_cpu_mem_usage": False,
        "max_grad_norm": 1.0,
        "hf_path": None,
        "base_output_dir": "model-output",
        "mode": "adamw",
        "learning_rate": 6.0e-4,
        "weight_decay": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1.0e-8,
        "muon_learning_rate": 0.02,
        "muon_beta1": 0.95,
        "muon_ns_steps": 6,
        "muon_nesterov": False,
        "muon_max_dim": 16384,
        "hybrid_sam_rho": 1.0,
        "hybrid_sam_ascent": "muon",
        "hybrid_sam_descent": "adam",
        "hybrid_sam_ascent_beta1": None,
        "hybrid_sam_perturbation_start_step": 0,
        "hybrid_sam_normalize_perturbation": True,
        "hybrid_sam_perturbation_norm": "balanced",
        "hybrid_sam_perturbation_scale": "absolute",
        "hybrid_sam_muon_fallback_ascent": "skip",
        "hybrid_sam_perturb_muon_eligible_only": False,
        "hybrid_sam_track_stats": True,
        "eval_perturbed": True,

        "compile": True,
        "compile_mode": "reduce-overhead",
        "compile_fullgraph": True,
        "compile_retrieval": False,

        "gradient_checkpointing": True,

        "num_workers": 8,

        "log_params_every_n": 100,

        # model parameters
        "hidden_size": 1024,
        "depth": 12,
        "n_head": 8,
        "ffn_type": "geglu",


        "wandb_project": "SAMVariants",

        "qk_norm": True,
  

        "hf_cache_dir": "/fsx/scratch/huggingface/hub",
        "tokenized_dataset_path": "/fsx/scratch/huggingface/tokenized/gpt2-medium_openwebtext_1024",
    }

    # Optional overrides from a JSON file for sweep scripts or manual runs
    if override_args:
        args.update(override_args)

    args["override_json_path"] = args.get("_override_json_path")
    args["override_name"] = args.get("_override_name")
    args["config_fingerprint"] = _config_fingerprint(args)

    config = AutoConfig.from_pretrained(
        args['model_name_or_path'],
        trust_remote_code=args['trust_remote_code'],
    )
    vocab_size = config.vocab_size

    run_name = _build_run_name(args)
    args["run_name"] = run_name
    args["output_dir"] = f"{args['base_output_dir']}/{run_name}"
    args["wandb_run_name"] = run_name

    # Drop private metadata helpers before namespacing.
    args.pop("_override_json_path", None)
    args.pop("_override_name", None)

    args = SimpleNamespace(**args)

    print("Running with the following arguments:", flush=True)
    print(json.dumps(vars(args), indent=2), flush=True)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.output_dir is None:
        args.output_dir = time.strftime("run_%Y%m%d_%H%M%S")

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, 
                                                            mixed_precision="bf16",
                                                            **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
    )


    print("Creating model...", flush=True)
    model = Transformer(
        dim=args.hidden_size,
        depth=args.depth,
        heads=args.n_head,
        ff_mult=4,
        vocab_size=vocab_size,
        max_seq_len=args.block_size,
        gradient_checkpointing=args.gradient_checkpointing,
        qk_norm=args.qk_norm,
        ffn_type=args.ffn_type,
    )

    print("num parameters", sum(p.numel() for p in model.parameters()), flush=True)
    model = model.to(accelerator.device)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.token_embedding.weight.shape[0]
    if len(tokenizer) > embedding_size:
        print("resizing token embeddings", len(tokenizer), embedding_size)
        model.resize_token_embeddings(len(tokenizer))

    # ---- Load or build tokenized dataset ----
    tok_path = getattr(args, "tokenized_dataset_path", None)

    if tok_path and os.path.isdir(tok_path):
        logger.info(f"Loading pre-tokenized dataset from {tok_path}")
        print(f"Loading pre-tokenized dataset from {tok_path}", flush=True)
        lm_datasets = datasets.load_from_disk(tok_path)
    else:
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split={
                "train": f"train[{args.validation_split_percentage}%:]",
                "validation": f"train[:{args.validation_split_percentage}%]",
            },
            trust_remote_code=True,
            cache_dir=args.hf_cache_dir,
            num_proc=args.preprocessing_num_workers,
        )

        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        if args.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > config.max_position_embeddings:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
                )
                block_size = min(1024, config.max_position_embeddings)
        else:
            if args.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(args.block_size, tokenizer.model_max_length)

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        def tokenize_and_group(examples):
            tokenized = tokenize_function(examples)
            concatenated_examples = {k: list(chain(*tokenized[k])) for k in tokenized.keys()}
            total_length = len(concatenated_examples[list(tokenized.keys())[0]])
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        print("Starting dataset tokenization...", flush=True)
        with accelerator.main_process_first():
            lm_datasets = raw_datasets.map(
                tokenize_and_group,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Tokenize + group into {block_size}",
            )
        print("Dataset tokenization complete.", flush=True)

        if tok_path and accelerator.is_main_process:
            logger.info(f"Saving tokenized dataset to {tok_path}")
            lm_datasets.save_to_disk(tok_path)

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size, num_workers=args.num_workers, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size, num_workers=args.num_workers, pin_memory=True
    )

    # Optimizer
    # Weight decay exclusions using 1D param strategy (robust to naming conventions)
    # - 1D params: catches all biases and all normalization weights (LayerNorm, RMSNorm, etc.)
    # - Embeddings: 2D but shouldn't have weight decay
    optimizer_grouped_parameters = []
    n_decay, n_no_decay = 0, 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr_mult = getattr(param, "lr_mult", 1.0)
        # Scalar/1D params (calibration scalars, biases, norm weights) or
        # embeddings -> no weight decay.
        if (
            param.dim() <= 1
            or "embed" in name.lower()
            or getattr(param, "_no_weight_decay", False)
        ):
            wd = 0.0
            n_no_decay += 1
        else:
            wd = args.weight_decay
            n_decay += 1
        optimizer_grouped_parameters.append(
            {
                "params": [param],
                "weight_decay": wd,
                "lr": args.learning_rate * lr_mult,
            }
        )
    print(
        f"Params with weight decay: {n_decay}, without: {n_no_decay}, "
    )


    optimizer = _build_optimizer(args, model, optimizer_grouped_parameters)

    # Print optimizer param group settings (excluding raw parameter tensors).
    print("Optimizer param groups:", flush=True)
    for idx, group in enumerate(optimizer.param_groups):
        group_view = {k: v for k, v in group.items() if k != "params"}
        print(f"  Group {idx}: {group_view}")

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Allow optimizer to scale LR scheduler warmup without touching beta/alpha warmups.
    # e.g. num_warmup_steps=2000 and optimizer lr_warmup_mult=5.0 -> 10_000 warmup steps.
    lr_warmup_mult = float(optimizer.param_groups[0].get("lr_warmup_mult", 1.0))
    base_num_warmup_steps = int(args.num_warmup_steps or 0) * accelerator.num_processes
    scaled_num_warmup_steps = int(round(base_num_warmup_steps * lr_warmup_mult))

    # Clamp to avoid pathological schedules where warmup exceeds total training steps.
    total_sched_steps = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    scaled_num_warmup_steps = min(scaled_num_warmup_steps, int(total_sched_steps))

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=scaled_num_warmup_steps,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # compile
    if args.compile:
        # can alternatively use regular model, IF we do default compile
        print("Compiling model...", flush=True)
        model = torch.compile(model, mode=args.compile_mode, fullgraph=args.compile_fullgraph)

    # Prepare everything with our `accelerator`.
    print("Preparing accelerator...", flush=True)
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    print("Accelerator ready.", flush=True)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]
        init_kwargs = {
            "wandb": {
                "name": args.wandb_run_name,
            }
        }
        project_name = args.wandb_project
        print(f"Initializing wandb tracker ({project_name})...", flush=True)
        accelerator.init_trackers(project_name, experiment_config, init_kwargs=init_kwargs)
        print("Wandb initialized.", flush=True)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    perplexity = float("nan")
    

    # "auto": resume from the latest step_* checkpoint in output_dir if any
    # exist, else start fresh. Makes interrupted runs restart-safe.
    if args.resume_from_checkpoint == "auto":
        ckpts = []
        if args.output_dir and os.path.isdir(args.output_dir):
            for entry in os.scandir(args.output_dir):
                if entry.is_dir() and entry.name.startswith("step_"):
                    ckpts.append((int(entry.name.split("_")[1]), entry.path))
        args.resume_from_checkpoint = max(ckpts)[1] if ckpts else None

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    # allocated and reserved memory
    reserved_memory = torch.cuda.memory_reserved()
    progress_bar.set_postfix(vram=f"{reserved_memory / (1024 ** 3):.2f} GB")

    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    data_start = torch.cuda.Event(enable_timing=True)
    data_end = torch.cuda.Event(enable_timing=True)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    optimizer_start = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)
    step_start = torch.cuda.Event(enable_timing=True)
    step_end = torch.cuda.Event(enable_timing=True)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        
        dataloader_iter = iter(active_dataloader)
        for step in range(len(active_dataloader)):
            model.train()
            
            # Time data loading
            data_start.record()
            batch = next(dataloader_iter)
            data_end.record()
            
            with accelerator.accumulate(model):
                step_start.record()
                # Time forward pass
                forward_start.record()
                # labels are derived from input_ids (next-token shift); the
                # tokenized dataset stores only int32 input_ids
                tokens = batch["input_ids"].to(accelerator.device).long()
                input_ids, targets = tokens[:, :-1], tokens[:, 1:]

                # Check for denoising optimizer (may be wrapped by Accelerator)
                loss, logits = model(input_ids=input_ids, targets=targets)
                forward_end.record()
                
                # Sync loss across GPUs for accurate metrics (needed for multi-GPU training)
                synced_loss = accelerator.gather(loss.detach().float()).mean()
                
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += synced_loss
                
                # Time backward pass
                backward_start.record()
                accelerator.backward(loss)
                backward_end.record()
                
                # Synchronize to get accurate timings
                torch.cuda.synchronize()
                
                # Calculate elapsed times in milliseconds
                data_time = data_start.elapsed_time(data_end)
                forward_time = forward_start.elapsed_time(forward_end)
                backward_time = backward_start.elapsed_time(backward_end)
                
                # clip the gradients
                mini_logs ={
                        "step_loss": synced_loss,
                        "lr": _get_optimizer_lrs(optimizer)[0],
                        "timer/data_load_ms": data_time,
                        "timer/forward_ms": forward_time,
                        "timer/backward_ms": backward_time,
                    }

                if args.max_grad_norm is not None:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    mini_logs["grad_norm"] = grad_norm
                
                # Time optimizer step
                optimizer_start.record()
                
                optimizer.step()

                if accelerator.sync_gradients:
                    lr_scheduler.step()

                optimizer.zero_grad(set_to_none=True)
                optimizer_end.record()
                step_end.record()
                torch.cuda.synchronize()
                
                optimizer_time = optimizer_start.elapsed_time(optimizer_end)
                mini_logs["timer/optimizer_ms"] = optimizer_time
                mini_logs["timer/step_ms"] = step_start.elapsed_time(step_end)
                
                # Log update norm / perturbation stats if the optimizer exposes them.
                opt = getattr(optimizer, "optimizer", optimizer)
                if hasattr(opt, "last_update_norm"):
                    mini_logs["optim/update_norm"] = opt.last_update_norm
                if getattr(opt, "last_stats", None):
                    mini_logs.update({f"optim/{k}": v for k, v in opt.last_stats.items()})

                if (
                    args.log_params_every_n is not None
                    and args.log_params_every_n > 0
                    and completed_steps % args.log_params_every_n == 0
                ):
                    pass
                
                accelerator.log(
                        mini_logs,
                        step=completed_steps,
                    )

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    if accelerator.is_main_process:
                        _prune_checkpoints(args.output_dir, args.keep_last_n_checkpoints)
            if completed_steps >= args.max_train_steps:
                break

            
            if completed_steps % args.validate_every == 0:
                # Between steps params hold the perturbed weights w̃; always
                # evaluate at the clean iterate w so comparisons against
                # non-SAM baselines are apples-to-apples.
                opt = _unwrap_optimizer(optimizer)
                if hasattr(opt, "unperturbed"):
                    with opt.unperturbed():
                        eval_loss = _run_validation(model, eval_dataloader, accelerator, args)
                else:
                    eval_loss = _run_validation(model, eval_dataloader, accelerator, args)

                try:
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    perplexity = float("inf")

                # Second pass at w̃: loss(w̃) - loss(w) is a free sharpness probe
                eval_loss_perturbed = None
                if args.eval_perturbed and hasattr(opt, "unperturbed"):
                    eval_loss_perturbed = _run_validation(model, eval_dataloader, accelerator, args)

                msg = f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}"
                if eval_loss_perturbed is not None:
                    msg += (
                        f" eval_loss_perturbed: {eval_loss_perturbed}"
                        f" sam_gap: {eval_loss_perturbed - eval_loss}"
                    )
                logger.info(msg)

                if args.with_tracking:
                    eval_logs = {
                        "perplexity": perplexity,
                        "eval_loss": eval_loss,
                        "train_loss": total_loss.item() / len(train_dataloader),
                        "epoch": epoch,
                        "step": completed_steps,
                    }
                    if eval_loss_perturbed is not None:
                        eval_logs["eval_loss_perturbed"] = eval_loss_perturbed
                        eval_logs["eval_sam_gap"] = eval_loss_perturbed - eval_loss
                    accelerator.log(eval_logs, step=completed_steps)

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    # Training is done: resume checkpoints are dead weight now (the final
    # unperturbed model is saved below), so reclaim their disk.
    if accelerator.is_main_process and args.output_dir:
        _prune_checkpoints(args.output_dir, keep_last_n=0 if args.discard_checkpoints_at_end else args.keep_last_n_checkpoints)

    # Training is done: permanently move back to the clean iterate w before
    # saving (the shipped weights should never include the SAM perturbation).
    final_opt = _unwrap_optimizer(optimizer)
    if hasattr(final_opt, "remove_perturbation"):
        final_opt.remove_perturbation()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        print("Saving model to", args.output_dir)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)


if __name__ == "__main__":
    main()
