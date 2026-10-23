# Gajda

Denoise MRI machine noise from speech recordings to improve Whisper transcription. Transcription quality is the metric — not SNR, not how clean the audio sounds.

## Data

- WAVs live in `transfer_340619_files_84dc77d0/Records_for_VUT_analysis/` (subject × session × trial). Raw WAVs are read-only; write outputs to a separate directory.
- Speech is Czech, mostly short sentences.
- Some trials are nonsense speech: the subject repeats "pataka pataka". Nonsense is signal — preserve it through denoising. It is a tuning hazard: an over-aggressive filter erases it first.

## Pipeline

denoise → transcribe → compare transcripts.

Every denoise parameter (bands, thresholds, strength) must be exposed as a CLI arg or config value, tunable without code edits. Tuning loop: pick a few representative trials (sentence + pataka), sweep parameters, judge only by the resulting transcripts. Keep every run's transcripts for comparison.

## Whisper API

OpenAI-compatible endpoint at `https://llm.ai.e-infra.cz/v1`; the API key is in the environment. `~/.pi/agent/models.json` lists current E-infra models. Rate limited, at most 4 requests at one time.

## Environment

Use the project `.venv` (numpy, scipy, librosa, noisereduce, requests). There is no package manifest.
