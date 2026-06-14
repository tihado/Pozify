# Pozify Docs Index

This folder is ordered by purpose so the current workflow is easier to follow.

## Current Workflow

Read in this order if you want to understand and run the current project:

1. [../README.md](../README.md)
2. [10-overview-build-small-hackathon-report.md](10-overview-build-small-hackathon-report.md)
3. [20-router-training-report.md](20-router-training-report.md)
4. [21-router-huggingface-release.md](21-router-huggingface-release.md)
5. [30-coach-modal-training.md](30-coach-modal-training.md)
6. [31-coach-training-report.md](31-coach-training-report.md)
7. [40-data-custom-collection-guide.md](40-data-custom-collection-guide.md)

## Presentation / Demo

- [11-overview-demo-video-transcript.md](11-overview-demo-video-transcript.md)

## Notes

- Current coach-summary cloud runtime default: `Qwen/Qwen3-14B`
- Current fine-tuned coach-summary Hugging Face repo is best used through local merged-model
  inference, not HF serverless `chat_completion`
- Current app entrypoint: `uv run python app.py`
