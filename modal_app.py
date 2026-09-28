"""
Modal application definition -- HuggingFace Transformers backend.

Each judge model gets its own Modal class so that the weights are loaded
exactly once per container in @modal.enter() and then reused across
batches of texts.

This version:
- Uses batched GPU inference.
- Prints per-sample progress after each batch completes.
- Processes ONLY the first 4,000 rows of the original CSV.
- Resumes already-checkpointed samples within those first 4,000 rows.
- Removes the max_length/max_new_tokens warning.
- Suppresses the BPE tokenizer cleanup warning.
- Supports a 4th dataset ("survey") alongside nemotron/enron/wnut17 --
  registered in config.EXTRA_CORPORA; run_judge_job(judge_key, "survey")
  works for all 3 judges. The 3 original corpora take the unchanged
  code path.

judge_b/judge_c model swap history (cost + panel independence):
Llama-3.3-70B and Mistral-Large-2 (both dense bf16) were first
replaced with Qwen3-Next-80B-A3B-Instruct-FP8 and Mistral Small 4
119B-A6B (both MoE) to cut GPU-time cost. judge_c then went dense FP8
(Llama-3.3-70B-Instruct-FP8-dynamic) to restore a third distinct
model vendor, then swapped again to Llama-4-Scout-17B-16E-Instruct
(INT4/W4A16) -- MoE again, keeping the Meta vendor slot while cutting
active params per token from 70B to 17B. The INT4 build hit a CUDA
OOM mid-generation, so judge_c swapped a fourth time, same
architecture, to the FP8 build of that checkpoint -- which OOM'd for
the SAME underlying reason: compressed-tensors decompresses the whole
model to full precision by default on first forward pass, regardless
of INT4 vs FP8. The actual fix was `use_optimized_inference=True`
(CompressedTensorsConfig), which routes FP8 layers through fused FP8
kernels instead of decompressing -- no checkpoint swap needed. See
config.py JUDGES/GPU_CONFIG/GENERATION_PARAMS comments and the JudgeC
class below for the full history and remaining caveats.
"""

import modal

from config import (
    MODAL_APP_NAME,
    GPU_CONFIG,
    MODEL_CACHE_VOLUME_NAME,
    DATA_VOLUME_NAME,
    REMOTE_DATA_ROOT,
    GENERATION_PARAMS,
    INFERENCE_BATCH_SIZE,
    PIPELINE_TASKS,
    JUDGES,
    EXTRA_CORPORA,
)

app = modal.App(MODAL_APP_NAME)

model_cache = modal.Volume.from_name(
    MODEL_CACHE_VOLUME_NAME,
    create_if_missing=True,
)

data_volume = modal.Volume.from_name(
    DATA_VOLUME_NAME,
    create_if_missing=True,
)

# huggingface-secret must contain HF_TOKEN. Not strictly required for
# gating anymore now that judge_b/judge_c are both ungated (Apache
# 2.0), but harmless to keep -- also avoids anonymous-download rate
# limits on HF for these multi-GB checkpoints.
HF_SECRET = modal.Secret.from_name("huggingface-secret")


# -------------------------------------------------------------------
# Transformers image
# -------------------------------------------------------------------

transformers_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "triton==3.4",
        "kernels==0.16.0",
        "huggingface_hub[hf_transfer]",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/root/.cache/huggingface",
    })
    .add_local_python_source("config", "prompts")
)

# Separate image for judge_b/judge_c (both compressed-tensors FP8
# checkpoints now -- judge_c was briefly INT4/W4A16, but swapped back
# to FP8 after an OOM under that build; see the module docstring and
# config.py). NOT sharing
# judge_a's image above on purpose: judge_a's pinned triton==3.4 (needed for gpt-oss's MXFP4 path) is INCOMPATIBLE
# with the kernel these FP8 checkpoints dispatch to at runtime --
# transformers auto-downloads `kernels-community/finegrained-fp8` from
# the HF Hub the first time an FP8 linear layer runs, and that
# downloaded kernel's own code does
# `from triton.runtime.autotuner import JITFunction`, an older triton
# API that 3.4 no longer exposes. Leaving triton UNPINNED here lets
# pip resolve a version actually compatible with that fetched kernel,
# without touching judge_a's already-verified image at all.
fp8_transformers_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "kernels==0.16.0",
        "huggingface_hub[hf_transfer]",
        "compressed-tensors",
        # judge_c loads via the "image-text-to-text" pipeline task
        # (see PIPELINE_TASKS in config.py), which pulls in
        # AutoProcessor -> Llama4ImageProcessor for the vision side of
        # the checkpoint even though we only ever send text -- that
        # processor class hard-requires torchvision and pillow to
        # import at all, regardless of whether an image is ever
        # passed in.
        "torchvision",
        "pillow",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/root/.cache/huggingface",
    })
    .add_local_python_source("config", "prompts")
)

CACHE_MOUNT = {
    "/root/.cache/huggingface": model_cache
}


# -------------------------------------------------------------------
# Shared container-side helpers
# -------------------------------------------------------------------

def _load_pipeline(model_id: str, task: str = "text-generation", model_kwargs: dict = None):
    """
    Build a generation pipeline for LEFT-padded batching.

    `task` matters: most judges are plain causal-LM checkpoints, where
    "text-generation" is correct. A checkpoint registered on
    HuggingFace under "image-text-to-text" (judge_c's Llama 4 Scout
    quantized checkpoint) must be loaded with that same task, or
    transformers resolves it to the wrong model class -- see the
    PIPELINE_TASKS comment in config.py for why.

    `model_kwargs` is forwarded to the underlying `from_pretrained()`
    call. judge_c uses this to pass
    `quantization_config=CompressedTensorsConfig(use_optimized_inference=True)`
    -- see the JudgeC class comment for why that flag matters.
    """

    from transformers import pipeline

    pipe = pipeline(
        task,
        model=model_id,
        dtype="auto",
        device_map="auto",
        model_kwargs=model_kwargs or {},
    )

    # The model may have max_length=20 in its generation_config.
    # Our application explicitly uses max_new_tokens, so remove
    # max_length to avoid the repeated Transformers warning.
    if hasattr(pipe, "generation_config"):
        pipe.generation_config.max_length = None

    # Plain text-generation pipelines expose `.tokenizer` directly.
    # image-text-to-text pipelines wrap a processor instead, but that
    # processor still carries the underlying tokenizer at
    # `.tokenizer` -- fall back to that if the pipeline itself
    # doesn't have one.
    tok = getattr(pipe, "tokenizer", None)

    if tok is None:
        processor = getattr(pipe, "processor", None)
        tok = getattr(processor, "tokenizer", None) if processor else None

    if tok is None:
        raise RuntimeError(
            f"Could not locate a tokenizer on the '{task}' pipeline "
            f"for {model_id}"
        )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Decoder-only models must be left-padded for batched generation.
    tok.padding_side = "left"

    # Avoid the BPE tokenizer cleanup warning.
    tok.clean_up_tokenization_spaces = False

    return pipe


def _extract_reply(out) -> str:
    """
    Normalise whatever the pipeline returns into one raw reply string.

    With chat-formatted input, generated_text can contain the full
    conversation. The model answer is taken from the last message.

    GPT-OSS can additionally expose thinking/content fields. We prefer
    content and fall back to thinking if generation was truncated.
    """

    if isinstance(out, list):
        if not out:
            return ""
        out = out[0]

    gen = (
        out.get("generated_text", "")
        if isinstance(out, dict)
        else out
    )

    if isinstance(gen, str):
        return gen

    if isinstance(gen, list) and gen:
        last = gen[-1]

        if not isinstance(last, dict):
            return str(last)

        content = last.get("content") or ""

        if isinstance(content, list):
            content = " ".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict)
            )

        if not str(content).strip():
            content = last.get("thinking") or ""

        return str(content)

    return str(gen)


def _generate_batch(
    pipe,
    judge_key: str,
    texts: list,
    state: dict,
) -> list:
    """
    Run the classification prompt over texts.

    IMPORTANT:
    This remains BATched inference. The GPU processes the supplied
    texts as a batch.

    Per-sample progress is printed by run_judge_job AFTER this function
    returns the complete batch.
    """

    from prompts import build_messages

    params = GENERATION_PARAMS[judge_key]

    conversations = [
        build_messages(t)
        for t in texts
    ]

    call_kwargs = dict(
        max_new_tokens=params["max_new_tokens"],
        do_sample=False,
        # Per-judge now (config.py) -- judge_b runs a larger batch on
        # its H200's spare VRAM without touching judge_a/judge_c.
        batch_size=INFERENCE_BATCH_SIZE[judge_key],
    )

    # image-text-to-text pipelines (judge_c) take the conversation via
    # the `text=` keyword rather than as a positional argument -- see
    # PIPELINE_TASKS in config.py.
    is_multimodal_pipe = PIPELINE_TASKS.get(judge_key) == "image-text-to-text"

    def _call(**extra_kwargs):
        if is_multimodal_pipe:
            return pipe(text=conversations, **call_kwargs, **extra_kwargs)
        return pipe(conversations, **call_kwargs, **extra_kwargs)

    cte = params.get("chat_template_kwargs")

    if cte and state.get("supports_cte", True):
        try:
            return [
                _extract_reply(o)
                for o in _call(chat_template_kwargs=cte)
            ]

        except (TypeError, ValueError) as e:
            print(
                f"[{judge_key}] chat_template_kwargs unsupported "
                f"({e}); continuing without it",
                flush=True,
            )

            state["supports_cte"] = False

    return [
        _extract_reply(o)
        for o in _call()
    ]


# -------------------------------------------------------------------
# Judge A -- GPT-OSS-120B
# -------------------------------------------------------------------

@app.cls(
    image=transformers_image,
    gpu=GPU_CONFIG["judge_a"],
    volumes=CACHE_MOUNT,
    secrets=[HF_SECRET],
    scaledown_window=300,
    timeout=60 * 60 * 3,
)
class JudgeA:

    @modal.enter()
    def load(self):
        self.state = {}

        self.pipe = _load_pipeline(
            JUDGES["judge_a"]["model_id"]
        )

    @modal.method()
    def generate(self, texts: list) -> list:
        return _generate_batch(
            self.pipe,
            "judge_a",
            texts,
            self.state,
        )


# -------------------------------------------------------------------
# Judge B -- Qwen3-Next-80B-A3B-Instruct-FP8 (MoE, 3B active/token)
# Official FP8 checkpoint, ~80GB -- fits a single H200 with room to
# spare (see config.py GPU_CONFIG), unlike the ~160GB bf16 release.
# -------------------------------------------------------------------

@app.cls(
    image=fp8_transformers_image,
    gpu=GPU_CONFIG["judge_b"],
    volumes=CACHE_MOUNT,
    secrets=[HF_SECRET],
    scaledown_window=300,
    timeout=60 * 60 * 3,
)
class JudgeB:

    @modal.enter()
    def load(self):
        self.state = {}

        self.pipe = _load_pipeline(
            JUDGES["judge_b"]["model_id"]
        )

    @modal.method()
    def generate(self, texts: list) -> list:
        return _generate_batch(
            self.pipe,
            "judge_b",
            texts,
            self.state,
        )


# -------------------------------------------------------------------
# Judge C -- Llama-4-Scout-17B-16E-Instruct-FP8-dynamic (MoE,
# 17B active / 109B total, 16 experts, ~109GB FP8 checkpoint)
# RedHatAI's compressed-tensors dynamic FP8 checkpoint. Swapped in to
# replace the INT4/W4A16 build of the same checkpoint after that one
# OOM'd mid-generation -- but the FP8 build OOM'd too, for the exact
# same underlying reason: by DEFAULT, compressed-tensors decompresses
# the WHOLE model to full precision (bf16) on the first forward pass,
# regardless of whether the checkpoint is INT4 or FP8. For a ~109GB
# FP8 checkpoint that means a transient ~218GB of bf16 weights --
# comfortably past a 141GB H200, matching the OOM seen at ~26% into
# decompression.
#
# FIX: `use_optimized_inference=True` on CompressedTensorsConfig
# routes FP8 layers through fused FP8 kernels directly against the
# compressed weights instead -- no decompression step, no 2x memory
# blowup. This is opt-in in transformers (not automatic), which is
# almost certainly why judge_b's official Qwen3-Next FP8 checkpoint
# has been fine (loaded/routed differently) while this one wasn't.
# Passed in via model_kwargs below.
#
# CAVEAT: run the smoke_test entrypoint on judge_c before a full
# corpus run -- `use_optimized_inference` + the image-text-to-text
# pipeline path together haven't been proven end-to-end yet.
# -------------------------------------------------------------------

@app.cls(
    image=fp8_transformers_image,
    gpu=GPU_CONFIG["judge_c"],
    volumes=CACHE_MOUNT,
    secrets=[HF_SECRET],
    scaledown_window=300,
    timeout=60 * 60 * 4,
)
class JudgeC:

    @modal.enter()
    def load(self):
        self.state = {}

        from transformers import CompressedTensorsConfig

        self.pipe = _load_pipeline(
            JUDGES["judge_c"]["model_id"],
            task=PIPELINE_TASKS["judge_c"],
            model_kwargs={
                "quantization_config": CompressedTensorsConfig(
                    use_optimized_inference=True
                ),
            },
        )

    @modal.method()
    def generate(self, texts: list) -> list:
        return _generate_batch(
            self.pipe,
            "judge_c",
            texts,
            self.state,
        )


JUDGE_CLASSES = {
    "judge_a": JudgeA,
    "judge_b": JudgeB,
    "judge_c": JudgeC,
}


# -------------------------------------------------------------------
# Detached orchestrator
# -------------------------------------------------------------------

orchestrator_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pandas")
    .add_local_python_source(
        "config",
        "prompts",
        "parsing",
        "checkpoint_io",
    )
)


@app.function(
    image=orchestrator_image,
    volumes={REMOTE_DATA_ROOT: data_volume},
    timeout=60 * 60 * 12,
)
def run_judge_job(
    judge_key: str,
    corpus_name: str,
    chunk_size: int = 256,
    resume: bool = True,
):
    """
    Run one judge over ONLY the first 4,000 rows of the source CSV.

    GPU inference:
        batch-wise

    Progress:
        per-sample AFTER each batch finishes

    Example with chunk_size=256:

        GPU processes samples 1-256
        ↓
        checkpoint saved
        ↓
        1/4000 done
        2/4000 done
        ...
        256/4000 done

        GPU processes samples 257-512
        ↓
        checkpoint saved
        ↓
        257/4000 done
        ...
        512/4000 done

    If resume=True, already-checkpointed samples among the first 4,000
    are skipped.
    """

    from pathlib import Path
    import pandas as pd

    from parsing import parse_label
    from checkpoint_io import (
        load_checkpoint,
        append_checkpoint,
    )

    # ---------------------------------------------------------------
    # Validate judge
    # ---------------------------------------------------------------

    assert judge_key in JUDGE_CLASSES, (
        f"Unknown judge: {judge_key}"
    )

    # ---------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------

    root = Path(REMOTE_DATA_ROOT)

    # 4th dataset support: corpora registered in config.EXTRA_CORPORA
    # (currently just "survey") can override the source filename / id
    # column / row cap. For nemotron / enron / wnut17 this is None and
    # every line below behaves exactly as it did before.
    corpus_cfg = EXTRA_CORPORA.get(corpus_name)

    src_filename = (
        corpus_cfg.get("file", f"{corpus_name}_sourced.csv")
        if corpus_cfg
        else f"{corpus_name}_sourced.csv"
    )

    src_path = (
        root
        / "sourced"
        / src_filename
    )

    if not src_path.exists():
        raise FileNotFoundError(
            f"Missing {src_path} on volume "
            f"'{DATA_VOLUME_NAME}'. Push your sourced data there "
            f"first, e.g.:\n"
            f"  modal volume put {DATA_VOLUME_NAME} "
            f"data/sourced /sourced"
        )

    # ---------------------------------------------------------------
    # Load source dataset
    # ---------------------------------------------------------------

    df = pd.read_csv(src_path)

    print(
        f"[{judge_key}/{corpus_name}] "
        f"source dataset contains {len(df)} rows",
        flush=True,
    )

    # ---------------------------------------------------------------
    # Extra corpora only (e.g. survey): map the CSV's own id/text
    # columns onto the `sample_id` / `text` names the rest of this
    # function, the checkpoint files and downstream voting all use.
    # Original columns are kept (copy, not rename). Skipped entirely
    # for nemotron / enron / wnut17.
    # ---------------------------------------------------------------

    if corpus_cfg:

        id_col = corpus_cfg.get("id_column", "sample_id")
        text_col = corpus_cfg.get("text_column", "text")

        missing_cols = [
            c for c in (id_col, text_col)
            if c not in df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"[{corpus_name}] {src_path} is missing required "
                f"column(s) {missing_cols}; found {list(df.columns)}"
            )

        # Checkpoints are keyed by sample_id, so duplicate ids would
        # silently collapse rows -- fail loudly instead.
        if df[id_col].duplicated().any():
            raise ValueError(
                f"[{corpus_name}] id column '{id_col}' contains "
                f"duplicate values; ids must be unique"
            )

        if id_col != "sample_id":
            df["sample_id"] = df[id_col]

        if text_col != "text":
            df["text"] = df[text_col]

    # ---------------------------------------------------------------
    # IMPORTANT:
    # Select ONLY the first 4,000 ORIGINAL rows.
    #
    # This happens BEFORE removing checkpointed rows.
    #
    # Therefore the target is always:
    #
    #     original CSV rows 1 -> 4000
    #
    # not "4,000 new samples".
    # ---------------------------------------------------------------

    MAX_SAMPLES = 3000

    # Extra corpora may set their own cap in config.EXTRA_CORPORA
    # (None = judge every row). Original corpora keep the 3000 above.
    if corpus_cfg:
        MAX_SAMPLES = corpus_cfg.get("max_samples", MAX_SAMPLES)

    target_df = (
        df.head(MAX_SAMPLES).copy()
        if MAX_SAMPLES is not None
        else df.copy()
    )

    print(
        f"[{judge_key}/{corpus_name}] "
        f"targeting first {len(target_df)} rows only",
        flush=True,
    )

    # ---------------------------------------------------------------
    # Checkpoint
    # ---------------------------------------------------------------

    judged_dir = root / "judged"

    ckpt_path = (
        judged_dir
        / "checkpoints"
        / f"{corpus_name}_{judge_key}.jsonl"
    )

    done = (
        load_checkpoint(ckpt_path)
        if resume
        else {}
    )

    # Only count checkpointed samples that belong to our
    # first-4000 target.
    target_ids = set(
        target_df["sample_id"].tolist()
    )

    target_done = {
        sid: value
        for sid, value in done.items()
        if sid in target_ids
    }

    if target_done:
        print(
            f"[{judge_key}/{corpus_name}] "
            f"resuming: {len(target_done)}/"
            f"{len(target_df)} target rows already labeled",
            flush=True,
        )

    # ---------------------------------------------------------------
    # IMPORTANT:
    # Work only on first 4,000 rows.
    # Remove already checkpointed samples.
    # ---------------------------------------------------------------

    pending = target_df[
        ~target_df["sample_id"].isin(target_done.keys())
    ]

    texts = (
        pending["text"]
        .astype(str)
        .tolist()
    )

    ids = pending["sample_id"].tolist()

    print(
        f"[{judge_key}/{corpus_name}] "
        f"{len(texts)} samples remaining to process",
        flush=True,
    )

    # ---------------------------------------------------------------
    # Run batches
    # ---------------------------------------------------------------

    if texts:

        judge = JUDGE_CLASSES[judge_key]()

        for start in range(
            0,
            len(texts),
            chunk_size,
        ):

            chunk_ids = ids[
                start:start + chunk_size
            ]

            chunk_texts = texts[
                start:start + chunk_size
            ]

            batch_number = (
                start // chunk_size
            ) + 1

            batch_end = min(
                start + len(chunk_texts),
                len(texts),
            )

            print(
                f"[{judge_key}/{corpus_name}] "
                f"starting batch {batch_number}: "
                f"{start + 1}-{batch_end} "
                f"of {len(texts)} remaining",
                flush=True,
            )

            # -------------------------------------------------------
            # GPU inference.
            #
            # THIS IS BATCHED.
            # -------------------------------------------------------

            raw_outputs = judge.generate.remote(
                chunk_texts
            )

            # -------------------------------------------------------
            # Parse returned outputs.
            # -------------------------------------------------------

            new_rows = []

            for sid, raw in zip(
                chunk_ids,
                raw_outputs,
            ):
                label, status = parse_label(raw)

                new_rows.append({
                    "sample_id": sid,
                    "label": label,
                    "status": status,
                })

            # -------------------------------------------------------
            # Save completed batch to checkpoint.
            # -------------------------------------------------------

            append_checkpoint(
                ckpt_path,
                new_rows,
            )

            # Persist immediately to Modal Volume.
            data_volume.commit()

            # -------------------------------------------------------
            # Per-sample progress.
            #
            # The entire batch has completed before these lines run.
            # -------------------------------------------------------

            for r in new_rows:

                target_done[
                    r["sample_id"]
                ] = r

                print(
                    f"[{judge_key}/{corpus_name}] "
                    f"{len(target_done)}/{len(target_df)} done",
                    flush=True,
                )

            print(
                f"[{judge_key}/{corpus_name}] "
                f"batch {batch_number} completed "
                f"({start + 1}-{batch_end})",
                flush=True,
            )

    else:

        print(
            f"[{judge_key}/{corpus_name}] "
            f"nothing to do -- all "
            f"{len(target_df)} target rows already checkpointed",
            flush=True,
        )

    # ---------------------------------------------------------------
    # Create final output.
    #
    # IMPORTANT:
    # The output contains ONLY the first 4,000 source rows.
    # ---------------------------------------------------------------

    target_df[
        f"{judge_key}_label"
    ] = target_df[
        "sample_id"
    ].map(
        lambda s: target_done[s]["label"]
    )

    target_df[
        f"{judge_key}_parse_status"
    ] = target_df[
        "sample_id"
    ].map(
        lambda s: target_done[s]["status"]
    )

    out_path = (
        judged_dir
        / f"{corpus_name}_{judge_key}.csv"
    )

    target_df.to_csv(
        out_path,
        index=False,
    )

    data_volume.commit()

    print(
        f"Saved -> {out_path} "
        f"on volume '{DATA_VOLUME_NAME}'",
        flush=True,
    )

    print(
        f"Pull it down with: "
        f"modal volume get {DATA_VOLUME_NAME} "
        f"/judged data/judged",
        flush=True,
    )

    return str(out_path)


# -------------------------------------------------------------------
# Local smoke test
# -------------------------------------------------------------------

@app.local_entrypoint()
def smoke_test(judge: str = "judge_a"):
    """
    Quick end-to-end sanity check on one judge.

        modal run modal_app.py --judge judge_a
    """

    samples = [
        "Hey, are we still on for lunch at noon?",
        "My SSN is 123-45-6789 and my Chase account number is 000123456789.",
    ]

    outputs = JUDGE_CLASSES[judge]().generate.remote(
        samples
    )

    for text, raw in zip(
        samples,
        outputs,
    ):
        print(
            f"\nTEXT: {text}\nRAW : {raw!r}"
        )