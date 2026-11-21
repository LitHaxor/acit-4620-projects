# Step 4 Collapse & Optimization Diagnostics (`collapse.md`)

- **Rule:** `learning` if `last train_loss < 0.5 * epoch-1 train_loss`, else `not learning`; `last R2 ~ 0` if `|last R2| < 0.05`.

## 1. Per-Fold Training Loss, Validation Loss, MAE, R2 & Verdict

| Experiment | Fold | Ep1 tr_loss | Ep5 tr_loss | Ep10 tr_loss | Ep20 tr_loss | Last tr_loss (Ep) | Min val_loss (Ep) | Last val_loss | Min MAE (Ep) | Last MAE | Last R2 | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `E4_mdnet_t_baseline_ep60` | 0 | 2.1619 | 2.1391 | 2.1284 | 2.1428 | 2.1728 (ep 60) | 2.1131 (ep 5) | 2.1697 | 67.09 (ep 48) | 67.64 | 0.0287 | **not learning; last R2 ~ 0** |
| `E4_mdnet_t_baseline_ep60` | 1 | 2.1686 | 2.1500 | 2.1548 | 2.1513 | 2.1935 (ep 60) | 2.0274 (ep 2) | 2.1293 | 62.11 (ep 4) | 63.23 | -0.0007 | **not learning; last R2 ~ 0** |
| `E4_mdnet_t_baseline_ep60` | 2 | 2.1948 | 2.1707 | 2.1634 | 2.1554 | 2.1973 (ep 60) | 2.0168 (ep 1) | 2.1026 | 61.90 (ep 1) | 63.93 | -0.0008 | **not learning; last R2 ~ 0** |
| `E4_mdnet_t_baseline_ep60` | 3 | 2.0914 | 2.0904 | 2.0914 | 2.0768 | 2.1312 (ep 60) | 2.3116 (ep 22) | 2.3811 | 74.28 (ep 31) | 74.66 | -0.0110 | **not learning; last R2 ~ 0** |
| `E4_mdnet_t_baseline_ep60` | 4 | 2.1880 | 2.1996 | 2.1511 | 2.1500 | 2.1971 (ep 60) | 1.9968 (ep 7) | 2.1010 | 60.55 (ep 46) | 63.33 | 0.0110 | **not learning; last R2 ~ 0** |
| `E4_mobilenetv3_s_dcm_concat` | 0 | 2.0820 | 1.9477 | 2.0299 | 2.0215 | 2.0246 (ep 30) | 1.8482 (ep 1) | 2.0747 | 54.46 (ep 1) | 62.64 | 0.1014 | **not learning; last R2 != 0** |
| `E4_mobilenetv3_s_dcm_concat` | 1 | 2.0422 | 1.9286 | 1.9344 | 1.9657 | 1.9982 (ep 30) | 1.9623 (ep 2) | 2.0815 | 56.47 (ep 1) | 61.11 | -0.0457 | **not learning; last R2 ~ 0** |
| `E4_mobilenetv3_s_dcm_concat` | 2 | 2.0750 | 1.9721 | 1.9750 | 1.9524 | 1.9879 (ep 30) | 1.8359 (ep 2) | 2.0286 | 51.43 (ep 2) | 58.86 | 0.0724 | **not learning; last R2 != 0** |
| `E4_mobilenetv3_s_dcm_concat` | 3 | 1.9952 | 1.8091 | 1.8254 | 1.8608 | 1.8157 (ep 30) | 2.2205 (ep 1) | 2.3193 | 59.74 (ep 1) | 67.36 | -0.0618 | **not learning; last R2 != 0** |
| `E4_mobilenetv3_s_dcm_concat` | 4 | 2.0892 | 1.9614 | 1.9438 | 1.8436 | 1.7713 (ep 30) | 1.8344 (ep 2) | 1.9053 | 51.25 (ep 21) | 52.60 | 0.2118 | **not learning; last R2 != 0** |
| `E4_mobilenetv3_s_baseline` | 0 | 2.0306 | 1.3249 | 1.0176 | 0.5789 | 0.4696 (ep 30) | 1.5262 (ep 4) | 2.2139 | 37.15 (ep 26) | 37.44 | 0.6607 | **learning; last R2 != 0** |
| `E4_mobilenetv3_s_baseline` | 1 | 1.9785 | 1.3070 | 0.9972 | 0.5689 | 0.4644 (ep 30) | 1.8069 (ep 6) | 2.5000 | 47.05 (ep 22) | 48.40 | 0.1935 | **learning; last R2 != 0** |
| `E4_mobilenetv3_s_baseline` | 2 | 2.0287 | 1.3165 | 0.9875 | 0.5825 | 0.4637 (ep 30) | 1.8249 (ep 4) | 2.5115 | 46.69 (ep 23) | 47.78 | 0.1981 | **learning; last R2 != 0** |
| `E4_mobilenetv3_s_baseline` | 3 | 1.9196 | 1.2507 | 0.9468 | 0.5536 | 0.4523 (ep 30) | 2.1298 (ep 3) | 2.8131 | 56.06 (ep 3) | 58.95 | 0.0050 | **learning; last R2 ~ 0** |
| `E4_mobilenetv3_s_baseline` | 4 | 2.0173 | 1.3121 | 1.0179 | 0.5700 | 0.4604 (ep 30) | 1.7096 (ep 2) | 2.4445 | 45.26 (ep 11) | 48.10 | 0.1546 | **learning; last R2 != 0** |
| `E3_mdnet_t_reg_lr1e-3_ep60` | 0 | 0.8942 | 0.7867 | 0.7202 | 0.5759 | 0.0442 (ep 60) | 0.3821 (ep 41) | 0.4186 | 46.71 (ep 41) | 48.37 | 0.4639 | **learning; last R2 != 0** |
| `E3_mdnet_t_reg_lr1e-3_ep60` | 1 | 0.8655 | 0.7790 | 0.6386 | 0.5077 | 0.0229 (ep 60) | 0.5071 (ep 25) | 0.5299 | 49.97 (ep 47) | 50.74 | 0.2148 | **learning; last R2 != 0** |
| `E3_mdnet_t_reg_lr1e-3_ep60` | 2 | 0.9237 | 0.7998 | 0.6643 | 0.5149 | 0.0277 (ep 60) | 0.4920 (ep 38) | 0.5337 | 53.53 (ep 48) | 53.97 | 0.1519 | **learning; last R2 != 0** |
| `E3_mdnet_t_reg_lr1e-3_ep60` | 3 | 0.7230 | 0.6719 | 0.5258 | 0.3642 | 0.0144 (ep 60) | 1.0346 (ep 18) | 1.1473 | 60.86 (ep 27) | 64.06 | 0.0196 | **learning; last R2 ~ 0** |
| `E3_mdnet_t_reg_lr1e-3_ep60` | 4 | 0.8659 | 0.7884 | 0.7366 | 0.5731 | 0.0282 (ep 60) | 0.4241 (ep 28) | 0.4654 | 47.86 (ep 44) | 49.31 | 0.2627 | **learning; last R2 != 0** |

## 2. Per-Experiment Verdict Summary

| Experiment | Folds Analyzed | Train-Loss Verdict (`last < 0.5 * ep1`) | Validation R2 Verdict (`|last R2| < 0.05`) |
| :--- | :---: | :---: | :---: |
| `E4_mdnet_t_baseline_ep60` | 5/5 | 0/5 `learning` | 5/5 `last R2 ~ 0` |
| `E4_mobilenetv3_s_dcm_concat` | 5/5 | 0/5 `learning` | 1/5 `last R2 ~ 0` |
| `E4_mobilenetv3_s_baseline` | 5/5 | 5/5 `learning` | 1/5 `last R2 ~ 0` |
| `E3_mdnet_t_reg_lr1e-3_ep60` | 5/5 | 5/5 `learning` | 1/5 `last R2 ~ 0` |

## 3. Diagnostic Findings

1. **`E4_mdnet_t_baseline_ep60` (`not learning; last R2 ~ 0` on 5/5 folds):** `train_loss` does not decrease from epoch 1 (`2.09–2.19`) to epoch 60 (`2.13–2.20`), `last MAE = 66.56 +- 4.88` matches the constant median predictor (`66.21`), and `last R2` is `~0` (`-0.0110` to `+0.0287`, mean `0.0054`) on all 5 folds. By contrast, the regression-only reference **`E3_mdnet_t_reg_lr1e-3_ep60`** (`learning` on 5/5 folds; `last R2 != 0` on 4/5 folds, `R2 = 0.0196` on Fold 3) drives `train_loss` down by >95% (`0.72–0.92 -> 0.014–0.044`) and reaches `last R2 = 0.2226 +- 0.1629` (`MAE = 53.29 +- 6.38`). Adding the PFRML DCM + 2-layer Transformer + prototype classification head (`L_cls + 0.5 L_reg`) onto a scratch MDNet-T backbone stalls optimization.
2. **`E4_mobilenetv3_s_dcm_concat` (`not learning` on 5/5 folds; `last R2 ~ 0` on Fold 1 and negative `-0.0618` on Fold 3, mean `R2 = 0.0556 +- 0.1127`):** `train_loss` barely moves from epoch 1 (`2.00–2.09`) to epoch 30 (`1.77–2.02`, ratio `0.85–0.98 > 0.5`), whereas `E4_mobilenetv3_s_baseline` (`dcm_fusion='sum'`) reduces `train_loss` from `~2.00` to `~0.46`. Replacing element-wise sum with `4 x 224 -> 224` concatenation + `LayerNorm2d` before the Transformer blocks optimization.
3. **`E4_mobilenetv3_s_baseline` (`learning` on 5/5 folds; `last R2 != 0` on 4/5 folds, `0.0050` on Fold 3, mean `R2 = 0.2424 +- 0.2467`):** `train_loss` drops from `1.92–2.03` to `0.45–0.47` (`< 0.25x` of epoch 1). Note that `val_loss` reaches its minimum at epochs 2–6 (`1.53–2.13`) and rises to `2.21–2.81` by epoch 30 due to classification cross-entropy overconfidence, even while validation `MAE` improves from `48.85` (best-loss epoch) to `48.14` (last epoch).
