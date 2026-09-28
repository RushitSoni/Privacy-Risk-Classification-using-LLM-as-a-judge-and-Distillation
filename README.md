# Privacy Risk Labeling Pipeline — 3-Judge Panel + Majority Vote

Implements the two pipeline diagrams end to end:

1. **Sourcing** — 3 raw corpora -> proxy-stratified draw (oversampled
   2-2.5x toward PII-proxy-positive rows, via cheap regex
   heuristics) -> ready for the 3-judge LLM panel.
2. **Judging** — GPT-OSS-120B, Llama-3.3-70B, Mistral-Large-2 each
   run separately over the sourced text (one Modal GPU class per
   judge, same "one model per session" pattern as the original
   notebook). Inference is **HuggingFace Transformers**
   (`pipeline("text-generation", ...)`), not vLLM.
3. **Voting / Balancing** — majority vote across the 3 judges +
   Fleiss' kappa -> no-majority rows dropped and logged -> balanced
   to N samples/class -> final class-weighted training corpus.

Student (DistilBERT) training is the **next stage** and is only
stubbed here (`--stage train_student_stub`) -- it consumes
`data/final/final_training_corpus.csv` + `class_weights.csv`.

## Layout

```
privacy_pipeline/
  config.py            # <- all paths, models, sampling knobs live here
  prompts.py           # shared classification prompt (no reasoning field)
  parsing.py           # shared "Label: LOW/MEDIUM/HIGH" output parser
  sourcing.py          # proxy-stratified draw stage (local, no GPU)
  modal_app.py         # Modal GPU classes (transformers), one per judge
                        #   + run_judge_job: detached orchestrator (see below)
  checkpoint_io.py      # shared .jsonl checkpoint read/write, used locally & remotely
  judges_runner.py     # local-side caller: run one judge at a time
  voting.py            # majority vote + Fleiss' kappa + drop/log
  balance.py           # balance & truncate + class weights
  pipeline.py          # CLI orchestrator -- this is what you run
  requirements.txt
  data/
    raw/      nemotron.csv, enron.csv, wnut17.csv   <- put yours here
    sourced/  per-corpus proxy-stratified draws
    judged/   per-(judge, corpus) labeled CSVs
    voted/    per-corpus majority-vote results + dropped-row logs
    final/    final_training_corpus.csv + class_weights.csv
```

## Inference backend

Each judge is a Modal **class**, not a function, so the weights load
once in `@modal.enter()` and the same warm container serves every
chunk of text:

```python
@app.cls(gpu="H100", image=transformers_image, ...)
class JudgeA:
    @modal.enter()
    def load(self):
        self.pipe = pipeline("text-generation", model=MODEL_ID,
                             dtype="auto", device_map="auto")

    @modal.method()
    def generate(self, texts: list) -> list: ...
```

Image packages match the verified gpt-oss-120b reference run:
`torch`, `transformers`, `accelerate`, `triton==3.4`,
`kernels==0.16.0` (the last two are what make gpt-oss-120b's MXFP4
weights run on a single H100). No `vllm`, no `openai-harmony` —
gpt-oss's Harmony formatting is applied by its own chat template, and
Transformers splits the reply into `thinking` (analysis channel) and
`content` (final channel) for us.

Two knobs control batching, and they're independent:

- `chunk_size` (arg to `run_single_judge`, default 256) — texts per
  remote call, i.e. how often you see progress and how much is lost
  if a call fails.
- `INFERENCE_BATCH_SIZE` (`config.py`, default 8) — conversations per
  forward pass inside the container. **Lower this first if you hit
  CUDA OOM.**

`GENERATION_PARAMS` in `config.py` gives judge_a 512 `max_new_tokens`
while judge_b/judge_c get 16. That asymmetry is deliberate: gpt-oss
emits reasoning before its final answer, so a 16-token cap would be
truncated long before reaching the `Label:` line. It runs at
`reasoning_effort="low"` to keep that preamble short.

## Resumability and running unattended

The judging stage checkpoints progress to
`data/judged/checkpoints/{corpus}_{judge}.jsonl` after every chunk
(one JSON line per labeled row). If a run stops partway through --
crash, Ctrl-C, network blip -- re-running the exact same command
skips every `sample_id` already in that file and only sends the
remaining rows to the GPU. You don't lose progress, and you don't
pay to re-label rows twice. To force a clean re-run instead, delete
the checkpoint file or pass `resume=False`.

That checkpointing works the same way in both of the two ways to run
the judging stage:

**1. Local (attended) -- your laptop must stay open**

```bash
python pipeline.py --stage judge --judge judge_a
```

`judges_runner.py` drives the loop from your machine: each chunk is a
blocking call to the GPU on Modal, but the *loop* itself runs
locally. Close the laptop, lose wifi, or let it sleep, and the loop
stops -- resumable on your next run, but not running while you're
away.

**2. Detached (unattended) -- survives closing your laptop**

Here the orchestrating loop itself runs as a Modal function
(`run_judge_job` in `modal_app.py`), not on your machine, so
`--detach` keeps it running on Modal's infrastructure after your CLI
disconnects. Because the container's filesystem doesn't persist or
connect back to your laptop, data moves through a second Modal
Volume instead of local disk:

```bash
# one-time: push sourced data up
modal volume put privacy-judge-data data/sourced /sourced

# launch detached -- one corpus/judge combo per invocation
modal run --detach modal_app.py::run_judge_job \
    --judge-key judge_a --corpus-name enron

# check on it any time, from anywhere
modal app logs privacy-risk-judge-panel

# once it's done (or to check progress so far), pull results back down
modal volume get privacy-judge-data /judged data/judged
```

Run `modal run modal_app.py::run_judge_job --help` to confirm the
exact flag names for your installed Modal CLI version -- these can
shift slightly between versions.

Since both paths write the identical checkpoint format, you can start
a run locally, kill it, push the partial checkpoint up
(`modal volume put privacy-judge-data data/judged/checkpoints /judged/checkpoints`),
and resume it detached -- or the reverse.

## Setup

```bash
pip install -r requirements.txt
modal token new                     # one-time, links this machine to your Modal account
modal secret create huggingface-secret HF_TOKEN=hf_...   # for gated Llama-3.3 / Mistral-Large weights
```

Drop your three source CSVs into `data/raw/` as `nemotron.csv`,
`enron.csv`, `wnut17.csv` (any text column name — auto-detected
unless you set `TEXT_COLUMN_HINTS` in `config.py`).

## Running

Everything runs from your local machine; the GPU inference happens
on Modal.

```bash
python pipeline.py --stage source                 # local, fast, no GPU

python pipeline.py --stage judge --judge judge_a   # GPT-OSS-120B,   ~1x H100
python pipeline.py --stage judge --judge judge_b   # Llama-3.3-70B,  ~2x H100
python pipeline.py --stage judge --judge judge_c   # Mistral-Large-2,~4x H100

python pipeline.py --stage vote                    # local, combines the 3 judge CSVs
python pipeline.py --stage balance                 # local, writes final_training_corpus.csv
```

Or all at once (expensive — runs all three judges back-to-back):

```bash
python pipeline.py --stage all
```

## Notes / things to check before trusting this at scale

- **Not run against live infra here.** This environment has no
  network access to modal.com or the model hubs, so this is a
  correct-by-construction scaffold, not pre-verified working code.
  Smoke-test one judge before pointing it at the full corpora:

  ```bash
  modal run modal_app.py --judge judge_a
  ```

  That sends two hardcoded sample texts through and prints the raw
  generations, so you can confirm the model loads and the prompt
  format produces a parseable `Label: ...` line.
- **GPU sizing has less slack than on vLLM.** Transformers loads
  judge_b/judge_c at native bf16 with no paged-KV memory manager
  (~141GB and ~246GB of weights respectively). `H100:2` for judge_b
  is tight — bump it to `H100:4` if you OOM, or drop
  `INFERENCE_BATCH_SIZE` first.
- **Throughput will be lower than vLLM** for the same GPU spend;
  transformers has no continuous batching. Since generations here are
  very short (16 tokens for two of the three judges) the gap is
  narrower than usual, but budget for it on 10k+ row corpora.
- `prompts.py` is intentionally identical across all three corpora
  (no `[value]tag`-specific special-casing) and intentionally has
  no fixed if/then rule table — this is the fix for the tag-leakage
  and rule-heavy-prompt issues discussed alongside this project.
- `proxy_pii_score()` in `sourcing.py` is a cheap regex heuristic,
  **not** a real PII/NER detector — it only sets sampling weight
  during the draw, it never touches the labels judges produce.
- GPU sizes in `config.GPU_CONFIG` are starting guesses — adjust to
  what actually fits and is cost-effective on your Modal plan.
- `MIN_AGREEING_JUDGES = 2` (in `config.py`) means 2-of-3 agreement
  counts as a majority; set to `3` for unanimous-only voting.
- `voting.py` excludes `UNKNOWN` (parse-fallback) rows from the
  Fleiss' kappa calculation, since mixing a "no valid label"
  category into agreement stats distorts it — those rows can still
  count toward the no-majority drop if the two valid judges
  disagree with each other.
