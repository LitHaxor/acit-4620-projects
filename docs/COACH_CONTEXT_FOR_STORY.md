# Coach context for the project story (ACIT4620)

Written by the Claude "prompt coach" chat that guided this project (Nov 2026). Purpose: record the reasoning, decisions, problems and interpretations that are NOT visible in the repo, so the story agent can explain *why* things were done.

**Rules for the reader (agent):**
- The repo wins. If anything here disagrees with the code, `results/REPORT_NUMBERS.md` or `docs/EXPERIMENTS.md`, trust the repo and list the mismatch.
- Numbers in this file are for orientation only. Quote numbers from `results/REPORT_NUMBERS.md` and `results/final_comparison.md`.
- Everything here is tagged [coach-note]. Interpretations are marked HYPOTHESIS unless the repo has evidence.

---

## 1. Assignment and constraints [coach-note]

- Course ACIT4620 (OsloMet). Deadline for code AND report: Dec 1, 2026. The report is the main graded part: understanding the papers and connecting them to the project.
- Open-ended: any NN/DL project; novelty not required; replicating papers is fine.
- The report needs a literature review of two IEEE Transactions papers (PFRML and MDNet). The coach never saw the assignment brief or rubric (word count, structure and AI-declaration wording are unknown to the coach).
- AI use is allowed only with a disclaimer stating what the AI was used for.
- The author has 7 years of software development experience but only basic ML knowledge; wanted concepts and maths explained in plain words.
- The author wanted to finish the experiments fast (other reports pending), so late in the project the plan was compressed to about three days and the iPhone/Core ML demo was dropped.
- Hardware: MacBook Pro, Apple M5 Pro, 24 GB, PyTorch on the MPS GPU backend.

## 2. Phases and approximate timeline [coach-note] (exact timestamps are in docs/development_log.md and file metadata)

1. ~Nov 12: data pipeline (cleaning, splits, transforms), then metrics/losses and baselines E0 (median, ridge on haze statistics) and E1 (metadata MLP).
2. ~Nov 13: image baselines (ResNet-18 scratch/pretrained, MobileNetV3-S pretrained), MDNet-T (Paper 2 backbone) from scratch, then a resumable queue: Stage 1 extended fold-0 runs, Stage 2 config selection by a fixed rule, Stage 3 five-fold CV for each model family. Also E3b (see decisions).
3. ~Nov 13-14: PFRML head built (DCM, light Transformer, prototype head, attention pooling, physics gate, multitask loss), wired into training, ladder defined, pilot run, then the five-fold ablation queue (finished ~Nov 15).
4. ~Nov 15: image + metadata fusion code, fusion CV queue, final-model training and one-shot test evaluation, report-number generation. Experiments ended the same day.
5. After: repository cleanup, then writing.

## 3. Decisions and the reasons behind them [coach-note]

1. **Split by station, always.** Many photos share a station; random splits would let the model recognise places seen in training. A second grouping, spatial clusters (stations within ~25 km linked into clusters, via haversine distance and single-linkage clustering), was used for fusion because GPS is an input there.
2. **No colour jitter.** Colour and brightness are the physical haze signal. Only flips and small geometric changes.
3. **Hyperparameters only on validation folds; test set used once per final model.** The final-evaluation script refuses to evaluate the same model name twice.
4. **Pretrained backbone is the main choice.** Pretraining was the biggest lever (about 8 MAE points over from-scratch). MobileNetV3-Small was chosen as the main backbone: statistically tied with ResNet-18 and E3b, about twice as fast per epoch, ~12x fewer parameters.
5. **E3b is the author's own adaptation**, not part of MDNet: pretrained MobileNetV3-S + a small block of multiscale dilated convolutions, to test Paper 2's idea in a data-limited setting. It must be described as an adaptation.
6. **Stage 2 selection rule fixed in advance**: switch from default lr 1e-3 / 30 epochs only if fold-0 validation MAE improves by at least 1.0. Limitation: tuned on fold 0, which is also part of the CV averages (slightly optimistic for tuned models).
7. **Last-epoch reporting.** Early work reported the checkpoint with the lowest validation loss. For the PFRML runs that checkpoint was usually epoch 1-6 of 30, before the classifier had trained, and the validation loss contains a cross-entropy term that rises as the classifier gets over-confident while MAE keeps improving. For ResNet-18 pretrained, best-epoch selection on fold 3 chose epoch 1 and made the CV mean look worse (46.16 best-epoch vs 44.05 last-epoch). So from the Step 5 stage onward and in all summary tables, results are last-epoch. Best-epoch numbers must not be quoted.
8. **Statistics with five folds.** A paired Wilcoxon test cannot reach p < 0.05 with n = 5 (minimum two-sided p = 0.0625), so CV comparisons use mean paired difference ± standard error and folds won. For final test comparisons a station-level bootstrap (1,000 resamples of stations with replacement, seed 42) gives 95% confidence intervals, including paired differences. Skill score = 1 - MAE / MAE_median per fold removes the effect of fold difficulty.
9. **Ablations on all five folds, not fold 0 only**: per-fold MAE varies by about ±10 and fold 0 is the easiest fold, so one fold cannot resolve effects of 1-2 MAE.
10. **PFRML head variants.** The paper's Eq. 5 writes the regression MLP on the backbone feature chi directly, while its Fig. 7 labels a "prototype-weighted feature". The first implementation used [chi; prototype-weighted feature]. After reading the paper text the literal Eq. 5 variant was added as an extra run; both scored the same (about 48.0-48.1). The paper's own backbone follows ResNet-50 + ViT, so MobileNetV3-S and MDNet-T are substitutions. The paper trains with lr 2e-4, batch 16 and a 1024-256-1 MLP (verified only for the parts of the paper that were read: the regression head and implementation details).
11. **Fusion design.** Metadata = 5 numbers (GPS on a sphere + sin/cos of month). A small MLP embeds it and the embedding is concatenated to the image feature before the regression MLP. Modality dropout (zeroing the metadata for a fraction of training samples) and an auxiliary vision-only head were added to fight the shortcut. Evaluated with and without metadata ("meta-off") to see whether the model uses the image. Fusion was evaluated under spatial-cluster CV because that is the setting where location is least helpful.
12. **Fusion final model rule, fixed before the test run**: choose the fusion config with the lowest five-fold spatial last-epoch MAE (R1-neck fusion without modality dropout). Because fusion overfit quickly, a shorter schedule was chosen from the validation curves (N = 7 epochs), then validated by a fresh five-fold CV; the final model uses 7 epochs only if that CV mean was below 46.95 (the 30-epoch CV mean 47.45 minus a 0.5 margin). It was (44.40). Caveat: N was picked from the same folds, so the 44.40 is slightly optimistic; the test result is the clean check. The other final models used 30 epochs.
13. **Final models** were trained on all 8,176 training images (fixed epochs, last epoch) and evaluated once on the 2,920-image test set and on the >25 km subset (test images more than 25 km from any training image; 717 images, 205 stations).

## 4. Problems met and lessons (report these honestly) [coach-note]

- **The shortcut.** GPS + month alone gets low error on the test set because 63% of test images lie within 10 km of a training station. Far from training stations it fails (R2 about 0).
- **PFRML's prototype + multitask head did not help**: same neck, regression-only (R1) beat the full prototype head (R2) in CV and on the test set. Toggles (attention pooling, physics gate, class weights) gave at most about 1 MAE of improvement in CV, within noise. DCM concatenation (T4) trained badly.
- **PFRML head on MDNet-T from scratch did not train** (collapsed to near-constant predictions; train loss barely decreased). A retry at the paper's lr (2e-4, fold 0 only) improved but stayed worse than MDNet-T with a plain head. This is a training failure in our setting, not evidence against the idea. HYPOTHESES: Transformer + prototype head from scratch on ~8k images needs more data/epochs or a different optimisation setup; lr too high.
- **MDNet-T from scratch was not better than ResNet-18 from scratch.** Possible reasons are untested.
- **Fold 3 is hard for every model, including the constant-median predictor.** A few extreme stations (Delhi/NCR, Saltillo/Monterrey, Chiang Rai, Pune cells) hold most of fold 3's high-AQI tail. Fold 0 has more tail images but is easy, so it is where the tail sits, not how large it is.
- **Models overfit**: training loss falls to ~0.006 while validation loss stays flat; fusion peaks around epochs 4-14.
- **Spatial CV and the >25 km test subset disagree on images vs location.** In spatial CV location-only beat image-only in all five folds; on the >25 km test subset image-only beat location-only (CI touching zero). Unresolved; the subset is small and may be a different kind of "far".
- **Agent reliability.** The coding agent once silently skipped an instruction, once reported test counts it had not re-run, produced summary rows for experiments that had no curve or log files (those rows were ignored), and once pasted a per-epoch table whose per-fold columns did not match the generated file. Lesson: numbers are quoted only from generated files, and every stage was verified with checks (test count, file hashes, reproducing old results from saved checkpoints).
- **Early summary-script bug**: matching curve files by name prefix mixed runs; fixed to exact experiment-id matching.

## 5. Results orientation [coach-note] (re-read all numbers from REPORT_NUMBERS.md)

- Median predictor MAE about 66; metadata-only MLP: CV 43.1 (station) / 46.0 (spatial); test 40.3, >25 km 49.8.
- Image models, five-fold station CV, last epoch: pretrained ResNet-18 / E3b / PFRML-neck regression-only / plain MobileNetV3-S all about 44-46 (tied); scratch models about 53; PFRML full head about 47-48.
- Fusion, spatial CV: fusion without dropout 47.5 (30 epochs) and 44.4 (7 epochs); location-only 46.0; image-only 50.7; fusion depends strongly on location (metadata zeroed: about 52.6-53.5).
- Final test: fusion best on the full test (35.8), better than location-only and image-only with paired CIs excluding zero; on the >25 km subset fusion ties image-only and beats location-only; plain image model beats the PFRML full head on both. Fusion with metadata zeroed is worse than the plain image model.

## 6. Established vs hypothesis [coach-note]

ESTABLISHED (supported by paired CIs or consistent across folds): pretraining helps; PFRML full head is worse than a plain image model here; fusion beats both single modalities on the full test; metadata alone fails far from training stations.
HYPOTHESIS (not tested): why the prototype head costs accuracy (capacity competition between the classification and regression tasks; over-confident classifier); why fusion overfits (the metadata branch memorises locations); why scratch MDNet-T-based models fail; why spatial CV and the >25 km subset disagree.

## 7. Limitations list [coach-note]

Single seed (differences below about 1-2 MAE are within noise); hyperparameters tuned on fold 0; fusion epoch count chosen from the same CV folds; small >25 km subset (wide CIs; tail metrics are 0 for every model there and should be ignored); backbone substitutions relative to the papers; MPS GPU not bit-exact so retraining reproduces approximately; dataset version not recorded; PFRML Eq. 5 vs Fig. 7 ambiguity; crowdsourced global images with possible label noise (not examined).

## 8. Dropped or not done [coach-note]

iPhone/Core ML demo (time), Grad-CAM and attention maps, prototype-mixture uncertainty plots, sky-only vs ground-only analysis, ordinal regulariser, geolocation encodings (SatCLIP-style), ParNet attention (all parked in the spec), extra seeds, retraining the PFRML-on-MDNet-T model with a tuned optimisation recipe.

## 9. Verification debts (do not assert as fact) [coach-note]

- Section and equation references in docs/SPEC.md, paper_notes.md and related_work_notes.md were written by the coding agent; only the PFRML regression-head description and implementation details were checked against the paper text by the coach. MDNet details were not verified by the coach.
- The agent's tally "14 of 17 papers have severe leakage" is unverified; check Kamble et al., Nathvani et al. and Byler et al. in full text before using it. Several paper markdown files have garbled tables; do not quote numbers from them.
- Station counts differ between older docs (3,256 / 2,527) and the audit (3,259); use the audit.
- The E0 ridge alpha varies by fold; confirm it was chosen on the training portion only.
- MPS seconds per epoch say nothing about on-device (iPhone) latency.

## 10. Roles of AI tools (facts for the author's own disclosure; the author writes the disclosure) [coach-note]

- Claude (separate chats): acted as prompt coach (wrote prompts for the coding agent, reviewed outputs critically, proposed experiment designs and decision rules, explained ML concepts and statistics) and as tutor. The author reviewed and approved decisions and ran all training in a terminal.
- Gemini Pro in Antigravity inside Zed (coding agent): wrote and edited the code, tests, scripts and docs under the coach's prompts; verified with tests and file checks.
- NotebookLM: used by the author to explain the papers and to help draft the literature review.
- The author: ran the experiments, made the final calls, and writes the report.

## 11. Things the story must not claim

- That PFRML or MDNet "do not work" in general (only that these ideas did not help on this dataset and setup).
- That fusion "solves" the shortcut (it relies on location; meta-off is worse than a plain image model).
- That the image model is better than location-only far from training stations as a proven fact (the CI touches zero).
- Any best-epoch number as a main result.
- Any speed or accuracy claim about phones.
- Any number not found in results/REPORT_NUMBERS.md.
