# """
# Central configuration for the privacy-risk labeling pipeline.
# Edit paths, model names, and sampling parameters here -- nothing
# else in the project should hardcode these values.
# """
# from pathlib import Path

# # ---------------------------------------------------------------
# # Paths
# # ---------------------------------------------------------------
# DATA_DIR = Path("data")
# RAW_DIR = DATA_DIR / "raw"            # put nemotron.csv, enron.csv, wnut17.csv here
# SOURCED_DIR = DATA_DIR / "sourced"    # random oversampled draws land here
# JUDGED_DIR = DATA_DIR / "judged"      # per-(judge, corpus) labeled CSVs
# VOTED_DIR = DATA_DIR / "voted"        # majority-vote combined labels
# FINAL_DIR = DATA_DIR / "final"        # balanced training corpus

# for _d in (SOURCED_DIR, JUDGED_DIR, VOTED_DIR, FINAL_DIR):
#     _d.mkdir(parents=True, exist_ok=True)

# RAW_FILES = {
#     "nemotron": RAW_DIR / "nemotron.csv",
#     "enron":    RAW_DIR / "enron.csv",
#     "wnut17":   RAW_DIR / "wnut17.csv",
# }

# # Set to a column name to force it, or None to auto-detect the
# # longest average string column (same heuristic the original
# # notebook used).
# TEXT_COLUMN_HINTS = {
#     "nemotron": None,
#     "enron":    "text",
#     "wnut17":   "text",
# }

# # ---------------------------------------------------------------
# # Sourcing / random oversampling
# # (pipeline image 2: "3 source corpora -> random draw, oversample
# #  2-2.5x". Earlier version weighted the draw toward PII-proxy-
# #  positive rows via regex; removed -- see sourcing.py for why.)
# # ---------------------------------------------------------------
# BASE_SAMPLE_SIZE = {          # pre-oversample target per corpus
#     "nemotron": 4_500,
#     "enron":    4_000,
#     "wnut17":   1_500,
# }
# OVERSAMPLE_FACTOR_RANGE = (2.0, 2.5)   # actual draw = BASE * factor
# TEXT_FILTER = {
#     "min_chars": 20,
#     "max_chars": 4000,
#     "min_words": 4,
# }

# RANDOM_SEED = 42

# # How many rows pandas reads per chunk while streaming a large raw
# # CSV in sourcing.py. Lower this if you're memory-constrained, raise
# # it for fewer (cheaper) chunk-boundary crossings on a fast machine.
# READ_CHUNK_SIZE = 50_000

# # ---------------------------------------------------------------
# # LLM judge panel (pipeline image 2: "LLM-judge panel (parallel)")
# # ---------------------------------------------------------------
# JUDGES = {
#     "judge_a": {"label": "GPT-OSS-120B",    "model_id": "openai/gpt-oss-120b"},
#     "judge_b": {"label": "Llama-3.3-70B",   "model_id": "meta-llama/Llama-3.3-70B-Instruct"},
#     "judge_c": {"label": "Mistral-Large-2", "model_id": "mistralai/Mistral-Large-Instruct-2407"},
# }

# # Per-judge decoding settings for the HuggingFace Transformers backend.
# # All judges decode greedily (do_sample=False) -- deterministic labels,
# # so a re-run reproduces the same votes.
# #
# # judge_a needs a MUCH larger budget than the other two: gpt-oss emits
# # its analysis (reasoning) channel before the final channel, so a
# # 16-token cap would get truncated long before it ever reaches the
# # "Label: ..." line. The other two answer in the final channel
# # immediately, so a tiny cap keeps them fast and cheap.
# GENERATION_PARAMS = {
#     "judge_a": {
#         "max_new_tokens": 512,
#         "chat_template_kwargs": {"reasoning_effort": "low"},
#     },
#     "judge_b": {"max_new_tokens": 16},
#     "judge_c": {"max_new_tokens": 16},
# }

# # How many conversations the transformers pipeline processes in one
# # forward pass inside the container. Raise it for throughput, lower it
# # if you hit CUDA OOM on the bigger judges.
# INFERENCE_BATCH_SIZE = 4

# VALID_LABELS = ("LOW", "MEDIUM", "HIGH")

# # ---------------------------------------------------------------
# # Majority vote + reliability
# # (pipeline image 1: "Majority vote: 3 judges + Fleiss' kappa"
# #  -> "No majority: sample dropped, logged")
# # ---------------------------------------------------------------
# MIN_AGREEING_JUDGES = 2   # >=2 of 3 must agree, else the row is dropped

# # ---------------------------------------------------------------
# # Balance and truncate
# # (pipeline image 1: "Balance and truncate: 3,000 samples per class"
# #  -> "Final training corpus: 9,000 samples, class-weighted")
# # ---------------------------------------------------------------
# SAMPLES_PER_CLASS = 3_000

# # ---------------------------------------------------------------
# # Modal
# # ---------------------------------------------------------------
# MODAL_APP_NAME = "privacy-risk-judge-panel"
# MODEL_CACHE_VOLUME_NAME = "privacy-judge-model-cache"

# # A second volume, separate from the model-weights cache above, that
# # holds your DATA (sourced/judged/checkpoints CSVs) so the detached
# # orchestrator (run_judge_job in modal_app.py) can read and write it
# # without touching your local disk. Sync it with the local data/
# # folder using `modal volume put` / `modal volume get` -- see README
# # "Running unattended (survives closing your laptop)".
# DATA_VOLUME_NAME = "privacy-judge-data"
# REMOTE_DATA_ROOT = "/data"   # mount point for DATA_VOLUME_NAME inside Modal containers

# # Starting guesses, not benchmarked -- adjust to what actually
# # fits/runs cost-effectively on your Modal plan and quota.
# # NOTE: transformers loads these at their native dtype (bf16 for
# # judge_b/judge_c, MXFP4 for gpt-oss), with no vLLM paged-KV memory
# # manager -- so leave headroom. Rough weight footprints: judge_a ~60GB,
# # judge_b ~141GB, judge_c ~246GB. device_map="auto" shards across
# # whatever GPUs the function is given.
# GPU_CONFIG = {
#     "judge_a": "H100:1",   # gpt-oss-120b, MXFP4 quantized -- fits one card
#     "judge_b": "H100:2",   # llama-3.3-70b bf16 -- tight; bump to 4 if OOM
#     "judge_c": "H100:4",   # mistral-large-2 (~123B params) bf16
# }





# Qwen added



"""
Central configuration for the privacy-risk labeling pipeline.
Edit paths, model names, and sampling parameters here -- nothing
else in the project should hardcode these values.
"""
from pathlib import Path

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"            # put nemotron.csv, enron.csv, wnut17.csv here
SOURCED_DIR = DATA_DIR / "sourced"    # random oversampled draws land here
JUDGED_DIR = DATA_DIR / "judged"      # per-(judge, corpus) labeled CSVs
VOTED_DIR = DATA_DIR / "voted"        # majority-vote combined labels
FINAL_DIR = DATA_DIR / "final"        # balanced training corpus

for _d in (SOURCED_DIR, JUDGED_DIR, VOTED_DIR, FINAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_FILES = {
    "nemotron": RAW_DIR / "nemotron.csv",
    "enron":    RAW_DIR / "enron.csv",
    "wnut17":   RAW_DIR / "wnut17.csv",
}

# Set to a column name to force it, or None to auto-detect the
# longest average string column (same heuristic the original
# notebook used).
TEXT_COLUMN_HINTS = {
    "nemotron": None,
    "enron":    "text",
    "wnut17":   "text",
}

# ---------------------------------------------------------------
# Sourcing / random oversampling
# (pipeline image 2: "3 source corpora -> random draw, oversample
#  2-2.5x". Earlier version weighted the draw toward PII-proxy-
#  positive rows via regex; removed -- see sourcing.py for why.)
# ---------------------------------------------------------------
BASE_SAMPLE_SIZE = {          # pre-oversample target per corpus
    "nemotron": 4_500,
    "enron":    4_000,
    "wnut17":   1_500,
}
OVERSAMPLE_FACTOR_RANGE = (2.0, 2.5)   # actual draw = BASE * factor
TEXT_FILTER = {
    "min_chars": 20,
    "max_chars": 4000,
    "min_words": 4,
}

RANDOM_SEED = 42

# How many rows pandas reads per chunk while streaming a large raw
# CSV in sourcing.py. Lower this if you're memory-constrained, raise
# it for fewer (cheaper) chunk-boundary crossings on a fast machine.
READ_CHUNK_SIZE = 50_000

# ---------------------------------------------------------------
# 4th dataset: human-rated survey (extra / pre-labeled corpora)
#
# The survey is NOT one of the 3 raw corpora above, so it is
# deliberately NOT added to RAW_FILES / BASE_SAMPLE_SIZE /
# TEXT_COLUMN_HINTS -- it needs no sourcing/oversampling (it is a
# small, already-curated set that should be judged in full), and
# leaving those dicts alone means nemotron / enron / wnut17 and
# sourcing.py behave exactly as before.
#
# It only needs to be run through the same 3-judge panel. modal_app.py
# (run_judge_job) looks the corpus up here; any corpus_name that is
# NOT in this dict follows the original code path untouched.
#
# Fields:
#   file        -- CSV filename under <data root>/sourced/
#   id_column   -- unique row-id column in that CSV. run_judge_job
#                  copies it into `sample_id`, which the checkpointing
#                  and output code key on. The original column is kept.
#   text_column -- column holding the text to judge (copied to `text`
#                  if it isn't already called that).
#   max_samples -- row cap; None = judge every row (the 3 main corpora
#                  are capped at 3,000 inside run_judge_job).
#
# The CSV's human columns (mean_rating, median_rating, std_rating,
# num_ratings, majority, rating_resolved, label3) are passed straight
# through to the judged output, so judge labels sit next to the human
# label3 for agreement analysis.
# ---------------------------------------------------------------
EXTRA_CORPORA = {
    "survey": {
        "file":        "survey_with_label3.csv",
        "id_column":   "text_id",
        "text_column": "text",
        "max_samples": None,
    },
}

# ---------------------------------------------------------------
# LLM judge panel (pipeline image 2: "LLM-judge panel (parallel)")
#
# judge_b/judge_c history:
#
#   original judge_b: Llama-3.3-70B-Instruct   70B active  (dense, bf16)
#   original judge_c: Mistral-Large-2          123B active (dense, bf16)
#   -> swapped to MoE to cut GPU-time cost:
#   swap-1 judge_b: Qwen3-Next-80B-A3B-Instruct-FP8   3B active   (80B total MoE)
#   swap-1 judge_c: Mistral Small 4 119B-A6B          ~6.5B active (119B total MoE)
#   -> judge_c swapped again, dense FP8, to restore panel vendor
#      independence (Meta vs. judge_a's OpenAI, judge_b's Alibaba):
#   swap-2 judge_c: Llama-3.3-70B-Instruct-FP8-dynamic   70B active (dense, FP8)
#   -> judge_c swapped a third time, back to MoE, keeping the Meta
#      vendor slot but cutting active params way down again:
#   swap-3 judge_c: Llama-4-Scout-17B-16E-Instruct (W4A16)  17B active (109B total MoE, 16 experts)
#   -> swap-3's INT4 checkpoint hit a CUDA OOM inside compressed-
#      tensors' decompression hook (unpack_from_int32) partway through
#      generation -- that hook unpacks the WHOLE model's INT4 weights
#      back out to a wider dtype before the matmul, which spikes VRAM
#      well above the ~55GB the packed weights alone suggested. Swapped
#      a fourth time, same architecture, to the FP8 build of the same
#      checkpoint, since FP8 compressed-tensors checkpoints don't go
#      through that unpack-the-whole-model path -- they keep weights in
#      native fp8 and matmul against them more directly (presumably why
#      judge_b's Qwen3-Next FP8 checkpoint has been fine):
#   current judge_c: Llama-4-Scout-17B-16E-Instruct-FP8-dynamic  17B active (109B total MoE, 16 experts)
#
# judge_c runs RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic --
# a compressed-tensors dynamic FP8 (weights + activations) quant.
# ~109GB on disk (vs. INT4's theoretical ~55GB), leaving only ~32GB
# free on a 141GB H200 for KV cache and batching -- noticeably less
# headroom than the INT4 build promised on paper, hence the much
# smaller INFERENCE_BATCH_SIZE below (32 -> 8). Keeps the 17B-active
# MoE speed benefit and the Meta vendor slot. UNVERIFIED: we haven't
# proven the multimodal `image-text-to-text` pipeline path end-to-end
# on this exact checkpoint yet, even though the OOM-class of failure
# the INT4 build hit should be avoided by staying in native FP8. Run
# `modal run modal_app.py --judge judge_c` (the smoke test) before
# committing to a full corpus run.
#
# judge_b runs the official FP8 checkpoint (Qwen/Qwen3-Next-80B-A3B-
# Instruct-FP8, ~80GB) rather than the native bf16 release (~160GB):
# bf16 wouldn't fit with headroom on anything smaller than 3xH100/
# 2xH200, whereas FP8 fits a single H200 (141GB) with ~60GB to spare
# for the KV cache and batching -- see GPU_CONFIG below.
#
# Panel independence: judge_a is OpenAI (GPT-OSS), judge_b is Alibaba
# (Qwen), judge_c is Meta (Llama) -- three different labs.
#
# All three current models are UNGATED (Apache 2.0 / Llama license,
# no HF gating required) -- HF_SECRET below isn't strictly required
# for access, but it's harmless to keep (also avoids anonymous-
# download rate limits on these multi-GB checkpoints).
# ---------------------------------------------------------------
JUDGES = {
    "judge_a": {"label": "GPT-OSS-120B",                    "model_id": "openai/gpt-oss-120b"},
    "judge_b": {"label": "Qwen3-Next-80B-A3B-Instruct-FP8", "model_id": "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"},
    "judge_c": {"label": "Llama-4-Scout-17B-16E-Instruct-FP8", "model_id": "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic"},
}

# Pipeline task per judge. judge_a/judge_b are plain causal-LM
# checkpoints -- "text-generation" is correct. judge_c's checkpoint
# is registered on HuggingFace under the "Image-Text-to-Text" task:
# it's a dump of the full multimodal Llama4ForConditionalGeneration
# wrapper (language_model.*/vision_model.* prefixed weights), even
# though we only ever send it text. Loading it with
# pipeline("text-generation", ...) makes transformers resolve it to
# Llama4ForCausalLM instead -- a different class with a different,
# unprefixed key layout -- which fails to load almost every weight
# and then crashes inside compressed-tensors' decompression hook (the
# `group_size=0` pydantic error your smoke test hit is a downstream
# symptom of that, not an independent checkpoint bug). Loading it
# with the matching "image-text-to-text" task fixes the class
# resolution; the pipeline still accepts pure text-only conversations.
PIPELINE_TASKS = {
    "judge_a": "text-generation",
    "judge_b": "text-generation",
    "judge_c": "image-text-to-text",
}

# Per-judge decoding settings for the HuggingFace Transformers backend.
# All judges decode greedily (do_sample=False) -- deterministic labels,
# so a re-run reproduces the same votes.
#
# judge_a needs a MUCH larger budget than the other two: gpt-oss emits
# its analysis (reasoning) channel before the final channel, so a
# 16-token cap would get truncated long before it ever reaches the
# "Label: ..." line.
#
# judge_b (Qwen3-Next-80B-A3B-Instruct) is Instruct-only -- it never
# emits a thinking trace, so it answers immediately like the old
# dense judge_b did. Tiny token cap stays valid.
#
# judge_c (Llama-4-Scout-17B-16E-Instruct) is a plain instruct model
# with no reasoning/thinking mode -- no chat_template_kwargs needed.
# It's natively multimodal but used text-only here; text-only Instruct
# output has no chain-of-thought, so the same tiny 16-token cap the
# dense Llama-3.3-70B judge_c used still applies.
GENERATION_PARAMS = {
    "judge_a": {
        "max_new_tokens": 512,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    },
    "judge_b": {"max_new_tokens": 16},
    "judge_c": {"max_new_tokens": 16},
}

# How many conversations the transformers pipeline processes in one
# forward pass inside the container. Raise it for throughput, lower it
# if you hit CUDA OOM. Per-judge now (was a single global int) so
# judge_b can use the extra VRAM headroom from its FP8 checkpoint +
# H200 without changing judge_a/judge_c, which don't have that room.
# judge_b's 32 is a reasoned starting point, NOT benchmarked: its
# checkpoint (~80GB FP8) leaves ~60GB free on an H200, plenty of
# headroom for short (16-token) sequences at that batch size.
# judge_c came DOWN from 32 to 8 with the swap to the FP8 checkpoint
# (see JUDGES comment above): the FP8 weights are ~109GB on disk vs.
# INT4's theoretical ~55GB, leaving only ~32GB free on a 141GB H200
# for KV cache and batching, so the batch size needs to come down hard
# to stay safely inside that shrunk headroom. Both judge_b/judge_c run
# short sequences (16 output tokens), unlike judge_a's much larger
# 512-token budget, which is why judge_a stays at 8 despite its
# smaller weight footprint. If you hit CUDA OOM on judge_b or judge_c,
# step this back down first.
INFERENCE_BATCH_SIZE = {
    "judge_a": 16,
    "judge_b": 32,
    "judge_c": 16,
}

VALID_LABELS = ("LOW", "MEDIUM", "HIGH")

# ---------------------------------------------------------------
# Majority vote + reliability
# (pipeline image 1: "Majority vote: 3 judges + Fleiss' kappa"
#  -> "No majority: sample dropped, logged")
# ---------------------------------------------------------------
MIN_AGREEING_JUDGES = 2   # >=2 of 3 must agree, else the row is dropped

# ---------------------------------------------------------------
# Balance and truncate
# (pipeline image 1: "Balance and truncate: 3,000 samples per class"
#  -> "Final training corpus: 9,000 samples, class-weighted")
# ---------------------------------------------------------------
SAMPLES_PER_CLASS = 3_000

# ---------------------------------------------------------------
# Modal
# ---------------------------------------------------------------
MODAL_APP_NAME = "privacy-risk-judge-panel"
MODEL_CACHE_VOLUME_NAME = "privacy-judge-model-cache"

# A second volume, separate from the model-weights cache above, that
# holds your DATA (sourced/judged/checkpoints CSVs) so the detached
# orchestrator (run_judge_job in modal_app.py) can read and write it
# without touching your local disk. Sync it with the local data/
# folder using `modal volume put` / `modal volume get` -- see README
# "Running unattended (survives closing your laptop)".
DATA_VOLUME_NAME = "privacy-judge-data"
REMOTE_DATA_ROOT = "/data"   # mount point for DATA_VOLUME_NAME inside Modal containers

# Starting guesses, not benchmarked -- adjust to what actually
# fits/runs cost-effectively on your Modal plan and quota.
# NOTE: transformers loads these at their native dtype (MXFP4 for
# gpt-oss, compressed-tensors FP8 for both judge_b and judge_c), with
# no vLLM paged-KV memory manager -- so leave headroom.
# Rough TOTAL weight footprints (what must fit in VRAM): judge_a
# ~60GB, judge_b ~80GB FP8 (80B total, MoE), judge_c ~109GB FP8
# (109B total, MoE, 16 experts / 17B active). device_map="auto"
# shards across whatever GPUs the function is given, though none of
# these need sharding on an H200.
#
# judge_b was bf16 on H100:2 (~160GB weights, ~0GB headroom -- unsafe)
# before switching to the official FP8 checkpoint on a single H200
# (~80GB weights, ~60GB headroom): cheaper AND safer, one GPU instead
# of two, no sharding.
#
# judge_c went dense (123B active, Mistral-Large-2) -> MoE (~6.5B
# active, Mistral Small 4) -> dense FP8 (70B active, Llama-3.3-70B)
# -> MoE again (17B active, Llama-4-Scout-17B-16E, INT4) -> same
# architecture but FP8 instead of INT4. The INT4 build's theoretical
# ~55GB/~86GB-headroom numbers didn't hold in practice: compressed-
# tensors' decompression hook unpacks the WHOLE model's INT4 weights
# before the matmul, which OOM'd mid-generation well above that
# on-disk figure (see JUDGES comment above). FP8 avoids that unpack
# path, but at ~109GB weights it only leaves ~32GB headroom on a
# 141GB H200 -- less than the INT4 build promised on paper, but
# without the unpack-driven OOM spike, and INFERENCE_BATCH_SIZE for
# judge_c has been cut accordingly (32 -> 8). Still keeps the 17B
# active-param GPU-time saving vs. dense FP8, and the third distinct
# model vendor in the panel (Meta, vs. judge_a's OpenAI and judge_b's
# Alibaba).
GPU_CONFIG = {
    "judge_a": "H200:1",   # gpt-oss-120b, MXFP4 quantized -- fits one card
    "judge_b": "H200:1",   # Qwen3-Next-80B-A3B-Instruct-FP8 -- ~80GB FP8 checkpoint, ~60GB free on a 141GB H200
    "judge_c": "H200:1",   # Llama-4-Scout-17B-16E-Instruct-FP8-dynamic -- ~109GB FP8 checkpoint, ~32GB free on a 141GB H200
}