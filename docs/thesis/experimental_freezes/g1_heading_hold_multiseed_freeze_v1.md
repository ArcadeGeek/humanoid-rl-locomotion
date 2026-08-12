# G1 Heading-Hold Multi-Seed Scientific Freeze -- v1

- Freeze date: `2026-08-12`
- Git branch: `feat/g1-locomotion-task`
- Git HEAD: `ec415eb`
- Freeze ID: `g1_heading_hold_multiseed_freeze_v1`

## Frozen controller

- Frozen Upright680 + external heading hold
- body-frame `vx = 0.4 m/s`, `vy = 0.0 m/s`
- `Kp = 0.5 s^-1`
- smooth deadband `1 deg`
- `|wz_cmd| <= 0.15 rad/s`
- target heading `0 rad`
- controller implementation frozen at commit `ec415eb`

## Experimental design

- Development/reference trained policy: seed42.
- Reference development perturbations: seeds 40-49.
- Reference holdout perturbations: seeds 50-59.
- Confirmatory independent fine-tuning seeds: 101, 202, 303, 404, 505.
- Confirmatory perturbation seeds: 60-69, paired OFF/ON within each trained policy.
- Inferential unit: trained policy.
- The five confirmatory policies share the same frozen Scratch600 bootstrap; they are not five independent end-to-end Scratch trainings.

## Confirmatory policy-level results

| seed | heading improvement | ON final |heading| | wz RMSE Delta | vx abs-error Delta | root tilt RMS Delta | root tilt p95 Delta | overall |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 101 | 87.07% | 3.895 deg | -1.71% | -0.21% | +4.22% | +4.30% | FAIL |
| 202 | 90.22% | 4.116 deg | -0.61% | +0.91% | +9.79% | +9.87% | FAIL |
| 303 | 89.52% | 4.624 deg | -1.44% | -0.85% | +12.10% | +11.18% | FAIL |
| 404 | 63.34% | 1.067 deg | -0.09% | +0.13% | -0.69% | -0.52% | FAIL |
| 505 | 86.06% | 3.881 deg | +4.12% | -0.21% | +12.24% | +9.90% | FAIL |

## Across-policy confirmatory synthesis

- Full practical guardrail PASS: **0/5**.
- Heading improved: **5/5**.
- Heading improvement >=70%: **4/5**.
- Final |heading| improvement: mean **83.24%**, median **87.07%**.
- ON final |heading|: mean **3.516 deg**, median **3.895 deg**.
- wz RMSE change: mean **+0.05%**.
- vx abs-error change: mean **-0.05%**.
- Root tilt RMS change: mean **+7.53%**; worsened in **4/5**, exceeded +5% in **3/5**.
- Root tilt p95 change: mean **+6.94%**; worsened in **4/5**, exceeded +5% in **3/5**.

## Frozen interpretation

The frozen external heading-hold controller generalizes strongly for directional correction across the five independent fine-tuning policies, while leaving velocity tracking nearly unchanged on average. However, strict multidimensional robust acceptance does not generalize: 0/5 policies satisfy every unchanged practical guardrail.

The most recurrent trade-off is root tilt. Other gait/contact violations are policy-dependent. Seed404 is a floor-effect case: it fails only the relative >=70% heading criterion while achieving about 1.067 deg final absolute heading ON; this is a post-hoc interpretation and does not change the predeclared FAIL.

No controller/reward/guardrail retuning was performed after observing the five-policy replication results.

## Provenance

Confirmatory policy effects and across-policy aggregates in this freeze are derived directly from the raw schema-7 OFF/ON summary.json files for perturbation seeds 60-69. Machine-readable checkpoint SHA256 values, campaign paths, nominal evaluation paths, guardrails, raw policy-level metric means, and percentage changes are stored in the companion JSON manifest.
