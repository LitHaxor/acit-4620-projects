# Step 5c Epoch Decision (`step5c_decision.md`)

- **Item 1 Experiment:** `E5_fusion_r1_p00_ep7` (`E5_fusion_r1_p00_epN` with `N = 7`)
- **Candidate Epoch Count (`N`):** `7`
- **Item 1 5-Fold Last-Epoch Mean MAE:** `44.40 +- 7.30` (F0=55.09, F1=35.76, F2=43.09, F3=47.36, F4=40.69)
- **Decision Threshold:** `46.95` (`E5_fusion_r1_p00` 3-epoch centered MA at epoch 8 / benchmark reference)
- **Decision Rule:** If 5-fold mean last-epoch MAE `< 46.95`, set `E_final = N = 7`; otherwise set `E_final = 30`.
- **Result:** `mean_last_epoch_mae (44.40) < threshold (46.95) -> E_final = N = 7`
- **Selected Final Training Epochs:** `E_final = 7`
