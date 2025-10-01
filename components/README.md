# Reusable Components

This directory contains Kubeflow Pipelines (KFP) components that can be used for
evaluating and finetuning LLMs.

## Table of contents

- [prepare_yoda_dataset](#prepare_yoda_dataset): Prepare train/eval splits from
  the Yoda sentences dataset
- [train_model](#train_model): Distributed LoRA fine-tuning via Kubeflow Trainer
  v2, exports adapter + metrics
- [evaluate_model](#evaluate_model): Evaluate models with lm-eval-harness (vLLM)
  on configurable tasks

## prepare_yoda_dataset

**Purpose**: Download and preprocess the
[`dvgodoy/yoda_sentences`](https://huggingface.co/datasets/dvgodoy/yoda_sentences)
dataset into train/eval splits with the expected `prompt`/`completion` columns.

**Inputs (parameters)**

- **train_split_ratio (float, optional, default: 0.8)**: Ratio of data to use
  for training (0.0–1.0).

**Outputs (artifacts)**

- **yoda_train_dataset (system.Dataset)**: The train split of the dataset.
- **yoda_eval_dataset (system.Dataset)** The evaluation split of the dataset.

**Load from YAML**

```python
from kfp import components as kfp_components

prepare_yoda_dataset = kfp_components.load_component_from_url(
    "https://raw.githubusercontent.com/mprahl/pipelines-examples/main/components/yoda_dataset_component.yaml"
)
```

**Minimal usage**

```python
prep = prepare_yoda_dataset(train_split_ratio=0.8)
```

## train_model

**Purpose**: Launch a distributed training job (Kubeflow Trainer v2) to
fine-tune a base model with LoRA and export metrics and a LoRA adapter as a
model artifact. Optionally merge and save a full model.

**Prerequisites**:

1. Kubeflow Trainer v2 is installed on the cluster.
1. Ensure the pipeline-runner service account has access to create and read
   `TrainJob` objects in the `trainer.kubeflow.org` API group so that it can
   leverage Kubeflow Trainer v2.

**Inputs (artifacts)**

- **input_dataset (system.Dataset)**: Training dataset directory saved via
  `datasets.save_to_disk`.

**Inputs (parameters)**

- Required:
  - **model_name (str)**: Hugging Face model id (e.g.,
    `meta-llama/Llama-3.2-3B-Instruct`) or a path to a model directory on disk.
  - **pvc_path (str)**: Base path in the PVC for outputs.
  - **run_id (str)**: Unique run id (use `dsl.PIPELINE_JOB_ID_PLACEHOLDER`).
- Optional:
  - **epochs (int, default: 10)**
  - **lora_rank (int, default: 8)**
  - **learning_rate (float, default: 3e-4)**
  - **batch_size (int, default: 16)**
  - **max_length (int, default: 64)**
  - **max_steps (int, optional, default: None)**
  - **logging_steps (int, default: 10)**
  - **save_steps (int, optional, default: None)**
  - **save_strategy (str, default: "epoch")**
  - **optimizer (str, default: "adamw_torch")**
  - **adam_beta1 (float, default: 0.9)**
  - **adam_beta2 (float, default: 0.999)**
  - **adam_epsilon (float, default: 1e-8)**
  - **weight_decay (float, default: 0.01)**
  - **use_flash_attention (bool, default: False)**
  - **num_nodes (int, default: 2)**
  - **trainer_runtime (str, default: "torch-distributed")**

**Outputs (artifacts)**

- **output_model (system.Model)**: LoRA adapter directory (model + tokenizer).
- **output_metrics (system.Metrics)**

**Load**

```python
from kfp import components as kfp_components

train_model = kfp_components.load_component_from_url(
    "https://raw.githubusercontent.com/mprahl/pipelines-examples/main/components/train_component.yaml"
)
```

**Minimal usage**

```python
from kfp import dsl

train = train_model(
    input_dataset=prep.outputs["yoda_train_dataset"],
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    pvc_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
    run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
)
```

**Minimal usage for pipeline with evaluation**

```python
from kfp import dsl
import kfp.kubernetes

train_model_op = train_model(
    input_dataset=prep.outputs["yoda_train_dataset"],
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    # Note, this must use a workspace in the PipelineConfig.
    pvc_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
    run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
    num_nodes=2,
).set_cpu_request("2")
  .set_cpu_limit("2")
  .set_memory_request("30Gi")
  .set_memory_limit("30Gi")
  .set_accelerator_type("nvidia.com/gpu")
  .set_accelerator_limit("1")
```

## evaluate_model

**Purpose**: Evaluate a model with lm-eval-harness (vLLM backend) on
configurable tasks; optionally include a custom translation dataset. When a
custom translation dataset is provided, you can also log the prompts and
responses generated during evaluation.

**Inputs (artifacts)**

- **custom_translation_dataset (system.Dataset, optional)**: Dataset saved via
  `datasets.save_to_disk` for a custom translation task.
- **lora_adapter (system.Model, optional)**: LoRA adapter directory (model +
  tokenizer) to apply during evaluation. Provide this when `model_path` points
  to a base model and you want to evaluate with a fine-tuned adapter.

**Inputs (parameters)**

- Required:
  - **model_path (str)**: Hugging Face model id (e.g.,
    `meta-llama/Llama-3.2-3B-Instruct`) or a path to a model directory on disk.
- Optional:
  - **batch_size (int, default: 1)**
  - **limit (int, optional, default: None)**
  - **max_model_len (int, default: 4096)**
  - **gpu_memory_utilization (float, default: 0.8)**
  - **dtype (str, default: "bfloat16")**
  - **add_bos_token (bool, default: True)**
  - **include_classification_tasks (bool, default: True)**
  - **include_summarization_tasks (bool, default: True)**
  - **log_prompts (bool, default: True)**: When using a custom translation
    dataset, logs an array of prompt/response pairs to an output artifact.
  - **verbosity (str, default: "INFO")**
  - **max_batch_size (int, optional, default: None)**

**Outputs (artifacts)**

- **output_results (system.Artifact)**: JSON file with evaluation results.
- **output_prompts (system.Artifact)**: JSON file containing an array of objects
  in the form `{ "prompt": "...", "response": "..." }`. This is produced only
  when a `custom_translation_dataset` is provided and `log_prompts=True`.
- **output_metrics (system.Metrics)**

**Load**

```python
from kfp import components as kfp_components

evaluate_model = kfp_components.load_component_from_url(
    "https://raw.githubusercontent.com/mprahl/pipelines-examples/main/components/eval_component.yaml"
)
```

**Minimal usage**

```python
eval_step = evaluate_model(
    model_path="meta-llama/Llama-3.2-3B-Instruct",
    # custom_translation_dataset=prep.outputs["yoda_eval_dataset"],  # optional
)
```

**Minimal usage for pipeline with training and custom dataset**

```python
import kfp.kubernetes

eval_model_op = (
    evaluate_model(
        model_path="meta-llama/Llama-3.2-3B-Instruct",
        custom_translation_dataset=prepare_dataset_op.outputs["yoda_eval_dataset"],
        lora_adapter=train_model_op.outputs["output_model"],  # optional
    )
    .set_caching_options(enable_caching=False)
    .set_accelerator_type("nvidia.com/gpu")
    .set_accelerator_limit("1")
    .set_cpu_request("4000m")
    .set_memory_request("80G")
).after(train_model_op)
# Remove this if the model isn't gated
kfp.kubernetes.use_secret_as_env(
    task=eval_model_op,
    secret_name="hf-token",
    secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
)
```
