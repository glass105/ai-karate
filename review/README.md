# Pipeline Review Logs

`review/latest` contains the most recent pipeline logs under stable filenames so external reviewers can always use the same GitHub paths:

- `summary.json`: run configuration, identity totals, and strike totals.
- `tracks.csv`: per-frame identity and strike diagnostics.
- `process.log`: console output and ID confidence scores.
- `manifest.json`: source run, sizes, and SHA-256 hashes.

The annotated video is intentionally excluded from Git. After downloading a completed run, publish its logs with:

```powershell
.\scripts\publish_latest_logs.ps1 -SourceDirectory "videos\output\RUN_DIRECTORY" -Push
```

Each publication overwrites the files in `review/latest` and commits the new versions to the current branch.
