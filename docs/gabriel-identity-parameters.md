# Gabriel Identity And Tracking Parameters

This note summarizes the primary parameters used to identify and track Gabriel in the RunPod video pipeline after the DeepFace ArcFace and split-exclude-score updates.

## Current Primary Parameters

| Parameter | Current | What it does | Higher means | Lower means |
|---|---:|---|---|---|
| `--arena-roi` | `0.2,0.1,0.8,0.9` | Only candidates inside this region are considered. | Smaller/tighter ROI if narrowed; fewer background mistakes, but more risk of losing Gabriel. | Wider ROI if expanded; catches more Gabriel movement, but more background people. |
| `--fighter-candidate-limit` | `4` | Only the 4 largest foreground candidates are considered. | More candidates; may recover Gabriel but risks background people. | Fewer candidates; cleaner, but can miss Gabriel if he is partially detected. |
| `--fighter-a-require-red-gloves` | on | Requires red gloves for initial lock/reset. | Boolean, not numeric. | Off would be looser and risk wrong IDs. |
| `--fighter-a-min-red-glove-score` | `0.15` | Minimum red-at-hand/wrist score. | Stricter. Too high can remove Gabriel when gloves blur or occlude. | Looser. Too low can select red shirts/background. |
| `--fighter-a-reject-blue-gloves` | on | Rejects visible blue-glove candidates. | Boolean. | Off would allow opponent confusion. |
| `--fighter-a-require-standing` | on | Requires active standing fighter. | Boolean. | Off risks seated/referee/background people. |
| `--fighter-a-min-standing-score` | `0.45` | Standing confidence threshold. | Stricter; may reject crouched/lunging Gabriel. | Looser; may include seated/non-fighter people. |
| `--fighter-a-enable-face-match` | on | Enables Gabriel face matching. | Boolean. | Off removes a strong identity cue. |
| `--face-match-backend` | `deepface-arcface` | Face identity backend. | Not numeric. | Not numeric. |
| `--fighter-a-min-face-match-score` | `0.45` | Minimum Gabriel face match score. | Stricter; fewer false Gabriel matches, but may miss side/blurred face. | Looser; more recoveries, but higher wrong-face risk. |
| `--fighter-a-reject-face-mismatch` | on | Rejects visible faces that mismatch Gabriel. | Boolean. | Off makes face mismatch only informational. |
| `--fighter-a-exclude-reference-images` | `reference/exclude` | Negative examples: never Gabriel. | More/better excludes help, but bad excludes can confuse body matching. | Fewer excludes reduce false rejection but allow known wrong people. |
| `--fighter-a-min-exclude-face-match-score` | `0.45` | Face score needed to reject as excluded person. | Stricter; fewer false exclude rejections, but may miss excluded faces. | Looser; stronger blocking, but may reject Gabriel by face false match. |
| `--fighter-a-min-exclude-body-match-score` | `0.95` | Body/crop exclude threshold. Now soft unless Gabriel evidence is weak. | Stricter; less likely Gabriel is hurt by crop similarity. | Looser; more likely exclude body crop fights Gabriel. |
| `--fighter-a-min-exclude-reference-match-score` | `0.80` | Legacy/final exclude threshold. Still present for compatibility. | Stricter if used directly. | Looser if used directly. |
| `--locked-fighter-exclude-grace-score` | `0.96` | Exclude score needed before threatening already locked Gabriel. | Safer for Gabriel lock; extreme exclude needed to drop him. | More aggressive dropping; can lose Gabriel from false exclude. |
| `--locked-fighter-min-continuity-score` | `0.60` | Continuity needed to keep locked Gabriel through ambiguity. | Stricter; can drop Gabriel during occlusion/crossing. | Looser; keeps lock longer, but may preserve wrong ID. |
| `--locked-fighter-drop-confirmation-frames` | `10` | Frames required before dropping locked Gabriel for hard issues. | More stable; slower to drop bad lock. | More reactive; easier to lose Gabriel. |
| `--confirmed-lock-min-frames` | `30` | Frames before Gabriel is considered firmly locked. | More conservative; slower firm lock. | Faster firm lock; may lock wrong candidate early. |
| `--identity-recovery-confirmation-frames` | `8` | Frames required before accepting recovered Gabriel. | Stricter; fewer ID jumps, slower recovery. | Faster recovery, more wrong switches. |
| `--identity-switch-confirmation-frames` | `12` | Frames required before switching Gabriel to another track. | Firmer lock; fewer switches, slower correction. | Faster correction, but more erratic switching. |
| `--reset-to-start-side-after-missing` | `18` | Missing-frame delay before using left-side reset logic. | Waits longer; less premature reset. | Resets sooner; can recover faster but may snap wrong. |
| `--lineup-pause-frames` | `45` | Frames of low motion before detecting reset/lineup. | Stricter reset detection; fewer false resets. | More resets; can re-anchor too often. |
| `--lineup-motion-threshold` | `0.07` | Max motion allowed for lineup pause. | Looser pause detection if higher; may detect resets during action. | Stricter; may miss real stoppages. |
| `--lineup-separation-threshold` | `1.35` | Required fighter separation for lineup/reset. | Requires cleaner separation; fewer false resets. | Easier reset detection; more false re-anchors. |

## Most Sensitive Knobs

These are the first parameters to tune when Gabriel tracking becomes erratic:

```text
--fighter-a-min-red-glove-score
--fighter-a-min-face-match-score
--fighter-a-min-exclude-face-match-score
--fighter-a-min-exclude-body-match-score
--identity-switch-confirmation-frames
--identity-recovery-confirmation-frames
--locked-fighter-min-continuity-score
--lineup-pause-frames
```

## Tuning Rules Of Thumb

- For match thresholds, higher is stricter and lower is looser.
- For confirmation-frame parameters, higher is firmer/slower and lower is faster/jumpier.
- For Gabriel stability, prefer increasing confirmation frames before lowering identity thresholds.
- Exclude face matching should remain a hard rejection.
- Exclude body/crop matching should remain a soft warning unless Gabriel evidence is weak.

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
exclude_body = soft warning unless Gabriel evidence is weak
```

This prevents Gabriel from being dropped just because his white gi, black belt, or mat background resembles an exclude reference crop.
