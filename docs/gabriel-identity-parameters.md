# Gabriel Identity And Tracking Parameters

This note summarizes the primary parameters used to identify and track Gabriel in the RunPod video pipeline after the DeepFace ArcFace and split-exclude-score updates.

## Current Primary Parameters

| Parameter | Current | What it does | Higher means | Lower means |
|---|---:|---|---|---|
| `--arena-roi` | `0.2,0.1,0.8,0.9` | Only candidates inside this region are considered. | Smaller/tighter ROI if narrowed; fewer background mistakes, but more risk of losing Gabriel. | Wider ROI if expanded; catches more Gabriel movement, but more background people. |
| `--fighter-candidate-limit` | `4` | Only the 4 largest foreground candidates are considered. | More candidates; may recover Gabriel but risks background people. | Fewer candidates; cleaner, but can miss Gabriel if he is partially detected. |
| `--fighter-a-start` | `left` | Side where Gabriel starts and resets after stoppages/lineups. Use `left` or `right`. | Not numeric. | Not numeric. |
| `--fighter-a-glove-color` | `red` | Selects Gabriel's expected glove color for the run: `red`, `white`, `blue`, or `none`. The selected color is required for initial lock/reset. | Not numeric. | Not numeric. |
| `--fighter-a-require-red-gloves` | selected when glove color is `red` | Requires red gloves for initial lock/reset. | Boolean, not numeric. | Off would be looser and risk wrong IDs. |
| `--fighter-a-require-white-gloves` | selected when glove color is `white` | Requires white gloves for initial lock/reset. | Boolean, not numeric. | Off would be looser and risk wrong IDs. |
| `--fighter-a-require-blue-gloves` | selected when glove color is `blue` | Requires blue gloves for initial lock/reset. | Boolean, not numeric. | Off would be looser and risk wrong IDs. |
| `--fighter-a-min-red-glove-score` | `0.15` | Minimum red-at-hand/wrist score. | Stricter. Too high can remove Gabriel when gloves blur or occlude. | Looser. Too low can select red shirts/background. |
| `--fighter-a-min-white-glove-score` | `0.02` | Minimum white-at-hand/wrist score. | Stricter. Too high can remove Gabriel when white gloves blur or blend with the uniform. | Looser. Too low can confuse white gi sleeves for gloves. |
| `--fighter-a-min-blue-glove-score` | `0.15` | Minimum blue-at-hand/wrist score. | Stricter. Too high can remove Gabriel when blue gloves blur or occlude. | Looser. Too low can select blue background/mat/signage. |
| `--fighter-a-reject-red-gloves` | off unless passed | Rejects candidates whose strongest visible glove color is red. | Boolean. | Off avoids false rejects from mixed/blurred color evidence. |
| `--fighter-a-reject-white-gloves` | off unless passed | Rejects candidates whose strongest visible glove color is white. | Boolean. | Off avoids false rejects from white sleeves/gi fabric near wrists. |
| `--fighter-a-reject-blue-gloves` | off unless passed | Rejects candidates whose strongest visible glove color is blue. | Boolean. | Off allows blue-glove candidates; turn on when Gabriel is not wearing blue. |
| `--fighter-a-require-standing` | on | Requires active standing fighter. | Boolean. | Off risks seated/referee/background people. |
| `--fighter-a-min-standing-score` | `0.45` | Standing confidence threshold. | Stricter; may reject crouched/lunging Gabriel. | Looser; may include seated/non-fighter people. |
| `--fighter-a-enable-face-match` | on | Enables Gabriel face matching. | Boolean. | Off removes a strong identity cue. |
| `--face-match-backend` | `deepface-arcface` | Face identity backend. | Not numeric. | Not numeric. |
| `--fighter-a-min-face-match-score` | `0.45` | Minimum Gabriel face match score. | Stricter; fewer false Gabriel matches, but may miss side/blurred face. | Looser; more recoveries, but higher wrong-face risk. |
| `--fighter-a-reject-face-mismatch` | on | Rejects visible faces that mismatch Gabriel. | Boolean. | Off makes face mismatch only informational. |
| `--fighter-a-exclude-reference-images` | `reference/exclude` | Negative examples: never Gabriel. | More/better excludes help, but bad excludes can confuse body matching. | Fewer excludes reduce false rejection but allow known wrong people. |
| `--fighter-a-min-exclude-face-match-score` | `0.45` | Face score needed to reject as excluded person. | Stricter; fewer false exclude rejections, but may miss excluded faces. | Looser; stronger blocking, but may reject Gabriel by face false match. |
| `--fighter-a-min-exclude-body-match-score` | `0.97` | Body/crop exclude threshold. With hard veto on, this blocks candidates even if red/pose/continuity are strong. | Stricter; less likely Gabriel is hurt by crop similarity. | Looser; more aggressive blocking of known excluded people. |
| `--fighter-a-min-exclude-reference-match-score` | `0.90` | Legacy/final exclude threshold. Still present for compatibility. | Stricter if used directly. | Looser if used directly. |
| `--fighter-a-exclude-reference-hard-veto` | off unless passed | Makes exclude references a true hard-negative rule. Red gloves, pose match, and tracker continuity cannot rescue an excluded candidate. | Boolean. | Off keeps legacy soft body-exclude behavior. |
| `--fighter-a-exclude-veto-confirmation-frames` | `4` | Consecutive exclude-veto frames before dropping an already locked Gabriel track. | Slower/more tolerant; fewer false drops. | Faster/more aggressive; removes wrong excluded locks sooner. |
| `--fighter-a-exclude-allow-strong-face-match` | on unless disabled | Allows a strong Gabriel face match to override body-only exclude similarity. Exclude face matches still reject. | Boolean. | Off means body exclude remains a hard veto in strict mode. |
| `--fighter-a-strong-face-match-score` | `0.75` | Gabriel face score needed for the optional body-exclude override. | Stricter; only very strong face matches override body exclude. | Looser; more chance of false face override. |
| `--locked-fighter-exclude-grace-score` | `0.98` | Exclude score needed before threatening already locked Gabriel. | Safer for Gabriel lock; extreme exclude needed to drop him. | More aggressive dropping; can lose Gabriel from false exclude. |
| `--locked-fighter-min-continuity-score` | `0.55` | Continuity needed to keep locked Gabriel through ambiguity. | Stricter; can drop Gabriel during occlusion/crossing. | Looser; keeps lock longer, but may preserve wrong ID. |
| `--locked-fighter-drop-confirmation-frames` | `15` | Frames required before dropping locked Gabriel for hard issues. | More stable; slower to drop bad lock. | More reactive; easier to lose Gabriel. |
| `--confirmed-lock-min-frames` | `30` | Frames before Gabriel is considered firmly locked. | More conservative; slower firm lock. | Faster firm lock; may lock wrong candidate early. |
| `--identity-recovery-confirmation-frames` | `8` | Frames required before accepting recovered Gabriel. | Stricter; fewer ID jumps, slower recovery. | Faster recovery, more wrong switches. |
| `--identity-switch-confirmation-frames` | `12` | Frames required before switching Gabriel to another track. | Firmer lock; fewer switches, slower correction. | Faster correction, but more erratic switching. |
| `--reset-to-start-side-after-missing` | `18` | Missing-frame delay before using `--fighter-a-start` as the reset side. | Waits longer; less premature reset. | Resets sooner; can recover faster but may snap wrong. |
| `--lineup-pause-frames` | `45` | Frames of low motion before detecting reset/lineup. | Stricter reset detection; fewer false resets. | More resets; can re-anchor too often. |
| `--lineup-motion-threshold` | `0.07` | Max motion allowed for lineup pause. | Looser pause detection if higher; may detect resets during action. | Stricter; may miss real stoppages. |
| `--lineup-separation-threshold` | `1.35` | Required fighter separation for lineup/reset. | Requires cleaner separation; fewer false resets. | Easier reset detection; more false re-anchors. |

## Most Sensitive Knobs

These are the first parameters to tune when Gabriel tracking becomes erratic:

```text
--fighter-a-min-red-glove-score
--fighter-a-min-white-glove-score
--fighter-a-min-blue-glove-score
--fighter-a-start
--fighter-a-min-face-match-score
--fighter-a-min-exclude-face-match-score
--fighter-a-min-exclude-body-match-score
--fighter-a-exclude-reference-hard-veto
--fighter-a-exclude-veto-confirmation-frames
--identity-switch-confirmation-frames
--identity-recovery-confirmation-frames
--locked-fighter-min-continuity-score
--lineup-pause-frames
```

## Tuning Rules Of Thumb

- For match thresholds, higher is stricter and lower is looser.
- For confirmation-frame parameters, higher is firmer/slower and lower is faster/jumpier.
- With `--fighter-a-glove-color none`, glove color is not positive identity evidence; enabled reject-glove flags still reject candidates by hand/wrist glove color.
- With `--fighter-a-start left`, Gabriel starts and resets left. With `--fighter-a-start right`, Gabriel starts and resets right.
- For Gabriel stability, prefer increasing confirmation frames before lowering identity thresholds.
- Exclude face matching should remain a hard rejection.
- Use `--fighter-a-exclude-reference-hard-veto` when known excluded people are still receiving Gabriel's yellow box.
- With hard veto on, lower `--fighter-a-exclude-veto-confirmation-frames` removes bad locked tracks faster; higher values reduce flicker from one-frame false excludes.

## Current Glove-None Setup

For runs using:

```text
--fighter-a-start left
--fighter-a-glove-color none
--fighter-a-reject-blue-gloves
--fighter-a-reject-red-gloves
```

the identity behavior is:

```text
start/reset side = left
positive glove evidence = disabled
red gloves = rejection-only
blue gloves = rejection-only
exclude references = hard-negative identity filter when enabled
```

## Split Exclude Score Meaning

The latest pipeline separates exclude evidence into:

```text
xb / excl_body  = crop/body similarity to exclude references
xf / excl_face  = face similarity to exclude references
x  / excl_final = final reported exclude score
```

The intended behavior is:

```text
exclude_face = hard rejection
exclude_body = soft warning by default; hard rejection when --fighter-a-exclude-reference-hard-veto is enabled
```

The hard-veto mode is better when the exclude folder contains specific people or items that should never be tagged as Gabriel. The only optional exception is `--fighter-a-exclude-allow-strong-face-match`, which lets a strong Gabriel face match override a body-only exclude crop match; exclude face matches still reject.
