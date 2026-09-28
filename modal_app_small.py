"""
Modal application definition -- HuggingFace Transformers backend,
SMALL-MODEL VARIANT.

Same design as the full-size modal_app.py (each judge gets its own
Modal class, weights loaded once in @modal.enter(), batched GPU
inference, per-sample checkpointing/resume), with three differences:

1. Judges are ~7-8B checkpoints (see config_small.py JUDGES) instead
   of 70-120B. All three are plain causal-LM checkpoints, so this
   variant needs only ONE Modal image -- no separate FP8/
   compressed-tensors image, no torchvision/pillow multimodal
   dependency the big Llama-4-Scout judge_c required.

2. Only one dataset is supported: whatever is registered in
   config_small.CORPORA (currently just "survey"). run_judge_job
   looks the corpus up there and fails loudly with instructions if
   it isn't registered, rather than silently falling back to a
   guessed filename -- see the CORPORA comment in config_small.py for
   how to register a new dataset.

3. GPUs are a single L4 per judge (see config_small.GPU_CONFIG)
   instead of H100/H200.
"""

import modal

from config_small import (
    MODAL_APP_NAME,
    GPU_CONFIG,
    MODEL_CACHE_VOLUME_NAME,
    DATA_VOLUME_NAME,
    REMOTE_DATA_ROOT,
    GENERATION_PARAMS,
    INFERENCE_BATCH_SIZE,
    PIPELINE_TASKS,
    JUDGES,
    CORPORA,
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

# huggingface-secret must contain HF_TOKEN. Required here (unlike the
# full-size config's note that it was "harmless but not strictly
# needed"): judge_c (Llama-3.1-8B-Instruct) is gated on HuggingFace
# under the Llama license, so a token with accepted-license access is
# needed to download it. judge_a/judge_b are ungated, but the token
# also avoids anonymous-download rate limits on their checkpoints.
HF_SECRET = modal.Secret.from_name("huggingface-secret")


# -------------------------------------------------------------------
# Transformers image -- single shared image for all three judges.
#
# Unlike the full-size app, none of these three checkpoints need
# compressed-tensors / FP8 kernel routing, so there's no
# triton-version conflict to keep separate images apart for. triton
# is still pinned to 3.4 for gpt-oss-20b's native MXFP4 path (same
# requirement as the 120B judge_a in the full-size config); judge_b/
# judge_c are plain bf16 and don't care about the triton version, so
# sharing this image with them is safe.
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
        # HF_HUB_ENABLE_HF_TRANSFER is deprecated (Hub now defaults to
        # the Xet transfer protocol) -- dropped to avoid the warning.
        #
        # Xet itself has been flaky in this environment (connection
        # resets hitting the xet-read-token endpoint during
        # @modal.enter() model downloads, e.g. on judge_d's
        # mistralai/Mistral-7B-Instruct-v0.3 pull). Disabling it falls
        # back to the classic HTTP downloader, which doesn't need that
        # extra round trip and has been the more reliable path here.
        "HF_HUB_DISABLE_XET": "1",
        "HF_HOME": "/root/.cache/huggingface",
    })
    .add_local_python_source("config_small", "prompts")
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

    `task` is kept as a parameter (rather than hardcoded) so a future
    multimodal small model could still be loaded under
    "image-text-to-text" the way the full-size config's judge_c was,
    without touching this function -- none of the current three
    judges need it, though.

    `model_kwargs` is forwarded to the underlying `from_pretrained()`
    call, kept for the same reason -- unused by all three judges here.
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
    # (image-text-to-text pipelines wrap a processor instead -- not
    # relevant to any current judge, but the fallback is cheap
    # insurance if PIPELINE_TASKS ever gains one.)
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
    This remains BATCHed inference. The GPU processes the supplied
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
        batch_size=INFERENCE_BATCH_SIZE[judge_key],
    )

    # None of the current judges are multimodal, but kept generic --
    # see PIPELINE_TASKS / _load_pipeline comments.
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
# Judge A -- GPT-OSS-20B (OpenAI, native MXFP4)
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
# Judge B -- Qwen2.5-7B-Instruct (Alibaba, dense, bf16)
# -------------------------------------------------------------------

@app.cls(
    image=transformers_image,
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
# Judge C -- Llama-3.1-8B-Instruct (Meta, dense, bf16)
#
# Plain "text-generation" checkpoint -- no CompressedTensorsConfig /
# quantization_config needed, unlike the full-size config's
# Llama-4-Scout judge_c. Gated on HuggingFace: HF_SECRET must hold a
# token that has accepted the Llama 3.1 license.
# -------------------------------------------------------------------

@app.cls(
    image=transformers_image,
    gpu=GPU_CONFIG["judge_c"],
    volumes=CACHE_MOUNT,
    secrets=[HF_SECRET],
    scaledown_window=300,
    timeout=60 * 60 * 3,
)
class JudgeC:

    @modal.enter()
    def load(self):
        self.state = {}

        self.pipe = _load_pipeline(
            JUDGES["judge_c"]["model_id"]
        )

    @modal.method()
    def generate(self, texts: list) -> list:
        return _generate_batch(
            self.pipe,
            "judge_c",
            texts,
            self.state,
        )


# -------------------------------------------------------------------
# Judge D -- Mistral-7B-Instruct-v0.3 (Mistral AI, dense, bf16)
#
# Plain "text-generation" checkpoint, same shape as judge_b/judge_c.
# Ungated on HuggingFace -- no license acceptance needed -- but it
# still rides on HF_SECRET since that secret is already required for
# judge_c and also helps avoid anonymous-download rate limits here.
# -------------------------------------------------------------------

@app.cls(
    image=transformers_image,
    gpu=GPU_CONFIG["judge_d"],
    volumes=CACHE_MOUNT,
    secrets=[HF_SECRET],
    scaledown_window=300,
    timeout=60 * 60 * 3,
)
class JudgeD:

    @modal.enter()
    def load(self):
        self.state = {}

        self.pipe = _load_pipeline(
            JUDGES["judge_d"]["model_id"]
        )

    @modal.method()
    def generate(self, texts: list) -> list:
        return _generate_batch(
            self.pipe,
            "judge_d",
            texts,
            self.state,
        )


JUDGE_CLASSES = {
    "judge_a": JudgeA,
    "judge_b": JudgeB,
    "judge_c": JudgeC,
    "judge_d": JudgeD,
}


# -------------------------------------------------------------------
# Detached orchestrator
# -------------------------------------------------------------------

orchestrator_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pandas")
    .add_local_python_source(
        "config_small",
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
    Run one judge over a corpus registered in config_small.CORPORA.

    GPU inference:
        batch-wise

    Progress:
        per-sample AFTER each batch finishes

    Example with chunk_size=256 and a 500-row corpus:

        GPU processes samples 1-256
        ↓
        checkpoint saved
        ↓
        1/500 done
        ...
        256/500 done

        GPU processes samples 257-500
        ↓
        checkpoint saved
        ↓
        257/500 done
        ...
        500/500 done

    If resume=True, already-checkpointed samples are skipped.

    Unlike the full-size app's run_judge_job, an unregistered
    corpus_name is a hard error here rather than a silent fallback to
    a guessed "<corpus_name>_sourced.csv" path -- see the CORPORA
    comment in config_small.py for how to register a new dataset.
    """

    from pathlib import Path
    import pandas as pd

    from parsing import parse_label
    from checkpoint_io import (
        load_checkpoint,
        append_checkpoint,
    )

    # ---------------------------------------------------------------
    # Validate judge + corpus
    # ---------------------------------------------------------------

    assert judge_key in JUDGE_CLASSES, (
        f"Unknown judge: {judge_key}"
    )

    corpus_cfg = CORPORA.get(corpus_name)

    if corpus_cfg is None:
        raise ValueError(
            f"Unknown corpus '{corpus_name}'. Register it in "
            f"CORPORA in config_small.py first -- see the comment "
            f"there for the file/id_column/text_column/max_samples "
            f"fields it needs. Currently registered: "
            f"{sorted(CORPORA.keys())}"
        )

    # ---------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------

    root = Path(REMOTE_DATA_ROOT)

    src_path = root / "sourced" / corpus_cfg["file"]

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
    # Map the CSV's own id/text columns onto the `sample_id` / `text`
    # names the rest of this function, the checkpoint files, and
    # downstream voting all use. Original columns are kept (copy, not
    # rename).
    # ---------------------------------------------------------------

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
    # Row cap: corpus_cfg["max_samples"] (None = judge every row).
    # ---------------------------------------------------------------

    max_samples = corpus_cfg.get("max_samples")

    target_df = (
        df.head(max_samples).copy()
        if max_samples is not None
        else df.copy()
    )

    print(
        f"[{judge_key}/{corpus_name}] "
        f"targeting {len(target_df)} rows",
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

    # Only count checkpointed samples that belong to our target rows.
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
    # Remove already checkpointed samples from the work queue.
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

        modal run modal_app_small.py --judge judge_a
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