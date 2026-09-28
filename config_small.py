"""
Central configuration for the privacy-risk labeling pipeline
-- SMALL-MODEL VARIANT.

Same shape as the original config.py, but:
  1. Judges are ~7-8B instead of 70-120B (one per vendor family, same
     panel-independence idea, much cheaper/faster to run).
  2. Only ONE dataset is judged: the pre-labeled "survey" corpus.
     The original 3 raw corpora (nemotron/enron/wnut17) and their
     sourcing/oversampling machinery are dropped entirely for this
     variant -- see CORPORA below for how to bring a dataset back.
  3. GPUs are sized down to match the smaller checkpoints.

Edit paths, model names, and sampling parameters here -- nothing
else in the project should hardcode these values.
"""
from pathlib import Path

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
DATA_DIR = Path("data")
SOURCED_DIR = DATA_DIR / "sourced"    # pre-curated / pre-sourced CSVs live here
JUDGED_DIR = DATA_DIR / "judged"      # per-(judge, corpus) labeled CSVs
VOTED_DIR = DATA_DIR / "voted"        # majority-vote combined labels
FINAL_DIR = DATA_DIR / "final"        # balanced training corpus

for _d in (SOURCED_DIR, JUDGED_DIR, VOTED_DIR, FINAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# ---------------------------------------------------------------
# Corpus registry
#
# This is now the ONLY place datasets are declared -- there's no
# separate "3 raw corpora needing sourcing" vs. "extra pre-labeled
# corpora" split like the full-size config has, because this variant
# only ever judges one, already-curated dataset.
#
# To add another dataset later (bringing back nemotron/enron/wnut17,
# or something new), you do NOT need to touch modal_app.py -- just:
#   1. Drop its CSV under data/sourced/ (run it through sourcing.py's
#      random-draw/oversample step first if it needs one).
#   2. Add an entry below with the same 4 fields the survey entry
#      uses. That's it; run_judge_job(judge_key, "<your_new_key>")
#      works immediately for all three judges.
#
# Fields:
#   file        -- CSV filename under <data root>/sourced/
#   id_column   -- unique row-id column in that CSV. run_judge_job
#                  copies it into `sample_id`, which checkpointing and
#                  output code key on. The original column is kept.
#   text_column -- column holding the text to judge (copied to `text`
#                  if it isn't already called that).
#   max_samples -- row cap; None = judge every row.
#
# The survey CSV's human columns (mean_rating, median_rating,
# std_rating, num_ratings, majority, rating_resolved, label3) are
# passed straight through to the judged output, so judge labels sit
# next to the human label3 for agreement analysis.
# ---------------------------------------------------------------
CORPORA = {
    "survey": {
        "file":        "survey_with_label3.csv",
        "id_column":   "text_id",
        "text_column": "text",
        "max_samples": None,   # small, already-curated -- judge every row
    },
    # "nemotron": {"file": "nemotron_sourced.csv", "id_column": "sample_id", "text_column": "text", "max_samples": 3_000},
    # ^ example of how a second dataset would be registered once it exists under data/sourced/.
}

# ---------------------------------------------------------------
# LLM judge panel -- small-model variant
#
# Same panel-independence idea as the full-size config (3 different
# labs), just one size class down (~7-8B instead of 70-120B) so the
# whole panel runs on cheap single-GPU instances:
#
#   judge_a: OpenAI  -- gpt-oss-20b
#            OpenAI's open-weight line only ships two sizes (20B and
#            120B) -- there's no 7-8B checkpoint in this family, so
#            20B is the closest same-family "small" option. Still
#            natively MXFP4-quantized, same as the 120B judge_a in
#            the full-size config, so the same reasoning-channel /
#            max_new_tokens=512 handling below still applies.
#   judge_b: Alibaba -- Qwen2.5-7B-Instruct
#            Dense, Instruct-only (no thinking trace), same family as
#            the full-size config's Qwen3-Next-80B-A3B-Instruct judge.
#   judge_c: Meta    -- Llama-3.1-8B-Instruct
#            Dense, plain text-generation checkpoint -- unlike the
#            full-size config's Llama-4-Scout, this one is NOT
#            registered under "image-text-to-text" on HuggingFace, so
#            none of the multimodal-pipeline / torchvision / pillow /
#            CompressedTensorsConfig handling the big version needed
#            applies here. Gated on HuggingFace (Llama license) --
#            HF_SECRET below is required, not just harmless-to-keep.
#   judge_d: Mistral AI -- Mistral-7B-Instruct-v0.3
#            Dense, plain text-generation checkpoint, same shape as
#            judge_b/judge_c (no thinking trace, ungated on
#            HuggingFace -- HF_SECRET isn't required for this one,
#            but it's already needed for judge_c so there's no reason
#            to special-case it).
#
# All four are plain causal-LM checkpoints loaded with the ordinary
# "text-generation" pipeline task -- see PIPELINE_TASKS.
#
# NOTE on MIN_AGREEING_JUDGES below: going from 3 to 4 judges means a
# 2-vote threshold no longer guarantees a unique majority (a 2-2 split
# across two different labels is now possible). Whatever code
# resolves MIN_AGREEING_JUDGES into a single label per row needs to
# handle/reject ties explicitly -- it isn't handled by this constant
# alone. Left at 2 here since that logic lives outside this config;
# bump to 3 if you'd rather require a strict >=3/4 supermajority
# instead.
# ---------------------------------------------------------------
JUDGES = {
    "judge_a": {"label": "GPT-OSS-20B",           "model_id": "openai/gpt-oss-20b"},
    "judge_b": {"label": "Qwen2.5-7B-Instruct",   "model_id": "Qwen/Qwen2.5-7B-Instruct"},
    "judge_c": {"label": "Llama-3.1-8B-Instruct", "model_id": "meta-llama/Llama-3.1-8B-Instruct"},
    "judge_d": {"label": "Mistral-7B-Instruct-v0.3", "model_id": "mistralai/Mistral-7B-Instruct-v0.3"},
}

# All three judges are ordinary causal-LM checkpoints in this variant
# -- no multimodal wrapper class to work around. Kept as a per-judge
# dict (rather than hardcoding "text-generation" everywhere) so a
# future multimodal small model could still override this per-judge,
# same as the full-size config does.
PIPELINE_TASKS = {
    "judge_a": "text-generation",
    "judge_b": "text-generation",
    "judge_c": "text-generation",
    "judge_d": "text-generation",
}

# Per-judge decoding settings for the HuggingFace Transformers backend.
# All judges decode greedily (do_sample=False) -- deterministic labels,
# so a re-run reproduces the same votes.
#
# judge_a (gpt-oss-20b) still emits its analysis (reasoning) channel
# before the final channel like the 120B version did, so it keeps the
# larger token budget and the low reasoning-effort template kwarg.
#
# judge_b (Qwen2.5-7B-Instruct), judge_c (Llama-3.1-8B-Instruct), and
# judge_d (Mistral-7B-Instruct-v0.3) are all plain instruct models
# with no chain-of-thought -- tiny token cap is enough for a one-line
# label.
GENERATION_PARAMS = {
    "judge_a": {
        "max_new_tokens": 512,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    },
    "judge_b": {"max_new_tokens": 16},
    "judge_c": {"max_new_tokens": 16},
    "judge_d": {"max_new_tokens": 16},
}

# How many conversations the transformers pipeline processes in one
# forward pass inside the container. Raise it for throughput, lower it
# if you hit CUDA OOM. These are starting guesses, NOT benchmarked --
# small checkpoints leave a lot of headroom on a 24GB L4 (see
# GPU_CONFIG), so batches can run noticeably larger than the full-size
# config's. judge_a stays smallest of the three: its 512-token budget
# means a much bigger transient KV cache per sequence than judge_b/c's
# 16-token cap.
INFERENCE_BATCH_SIZE = {
    "judge_a": 16,
    "judge_b": 16,
    "judge_c": 32,
    "judge_d": 32,
}

VALID_LABELS = ("LOW", "MEDIUM", "HIGH")

# ---------------------------------------------------------------
# Majority vote + reliability
# ---------------------------------------------------------------
MIN_AGREEING_JUDGES = 2   # >=2 of 3 must agree, else the row is dropped

# ---------------------------------------------------------------
# Balance and truncate
# ---------------------------------------------------------------
SAMPLES_PER_CLASS = 3_000

# ---------------------------------------------------------------
# Modal
#
# Distinct app/volume names from the full-size config so both
# variants can coexist in the same Modal workspace without colliding.
# ---------------------------------------------------------------
MODAL_APP_NAME = "privacy-risk-judge-panel-small"
MODEL_CACHE_VOLUME_NAME = "privacy-judge-model-cache-small"
DATA_VOLUME_NAME = "privacy-judge-data-small"
REMOTE_DATA_ROOT = "/data"   # mount point for DATA_VOLUME_NAME inside Modal containers

# Rough bf16/MXFP4 weight footprints (what must fit in VRAM):
#   judge_a  gpt-oss-20b               ~13GB (native MXFP4)
#   judge_b  Qwen2.5-7B-Instruct       ~15GB (bf16, 7.6B params)
#   judge_c  Llama-3.1-8B-Instruct     ~16GB (bf16, 8B params)
#   judge_d  Mistral-7B-Instruct-v0.3  ~15GB (bf16, 7.2B params)
#
# All four fit with room to spare on a single 24GB L4 -- no sharding,
# no multi-GPU needed, and L4 is Modal's cheapest GPU tier, which is
# the point of this variant. Starting guesses, not benchmarked --
# bump to "A10G:1" (also 24GB, faster compute) or up a tier if you hit
# CUDA OOM at the batch sizes above.
GPU_CONFIG = {
    "judge_a": "H100:1",   # gpt-oss-20b, MXFP4 -- ~13GB weights, ~11GB free for KV cache/batching
    "judge_b": "L4:1",   # Qwen2.5-7B-Instruct, bf16 -- ~15GB weights, ~9GB free
    "judge_c": "H100:1",   # Llama-3.1-8B-Instruct, bf16 -- ~16GB weights, ~8GB free
    "judge_d": "H100:1",   # Mistral-7B-Instruct-v0.3, bf16 -- ~15GB weights, ~9GB free
}