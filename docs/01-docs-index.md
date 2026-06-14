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

- Current coach-summary runtime default: `build-small-hackathon/pozify-coach-summary1`
- The Hugging Face provider tries `chat_completion` first, then `text_generation` for non-chat
  model repos; local merged-model inference remains the most predictable fine-tuned path
- Current app entrypoint: `uv run python app.py`
