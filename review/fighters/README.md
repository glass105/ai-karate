# Fighter Review Artifacts

This directory is the canonical GitHub location for lightweight review outputs organized by fighter name.

## Directory layout

```text
review/fighters/
  <fighter-name>/
    latest/
      manifest.json
      summary.json
    runs/
      <run-label>/
        manifest.json
        summary.json
```

Example:

```text
review/fighters/gabriel/latest/summary.json
review/fighters/gabriel/latest/manifest.json
review/fighters/gabriel/runs/rtmw-yolo26l-hybridsort-arcface-roi20y20-full-32gb/summary.json
```

## What belongs in GitHub

Commit small, reviewable files only:

- `manifest.json`
- `summary.json`
- short Markdown notes, if useful

Do not commit generated heavy artifacts:

- annotated `.mp4` files
- full frame dumps
- large debug logs
- large per-frame CSV files
- model weights

Those should stay on the RunPod volume, `videos/output/`, Google Drive, S3, or another artifact store.

## Fighter naming

Use lowercase folder names with hyphens if needed:

```text
gabriel
john-smith
fighter-a
```

The display name inside `manifest.json` or `summary.json` can keep normal capitalization, for example `Gabriel`.

## Latest vs runs

`latest/` should contain the current best review summary for the fighter.

`runs/<run-label>/` should be used when preserving older or experimental run summaries. Experiment branches should not be the source of truth for fighter profiles. If a useful experiment output is promoted, copy the small review files here under `main` and keep the large generated artifacts outside Git.
