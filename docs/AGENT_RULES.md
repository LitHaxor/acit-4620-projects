# Agent Rules & Operating Workflow

## Core Workflow & Constraints
1. **Context Efficiency:** At the start of a fresh session, read `docs/AGENT_RULES.md`, `docs/SPEC.md`, and `docs/PROGRESS.md` in order. Do not read full papers or `docs/PLAN.md` unless `docs/SPEC.md` is unclear on a specific detail; if so, read only the relevant section and inform the user.
2. **Step-by-Step Execution:** Build only the step requested by the user, run the specified acceptance checks and `pytest` tests, keep console output brief, update `docs/PROGRESS.md`, and stop for user review.
3. **Splitting & Data Integrity:** Always split or cross-validate grouped by `station_id` or `spatial_cluster` (`StratifiedGroupKFold`). Never use a random image split.
4. **Physics-Preserving Augmentations:** Never apply hue, saturation, brightness, colour jitter, or grayscale transforms (colour saturation and brightness encode the physical Mie scattering AQI signal). Use whole-image resizing ($224 \times 224$, keeping both sky and ground) and gentle geometric transforms only.
5. **Core ML & Hardware Compatibility:** Keep models lightweight for MacBook Pro (M5 Pro, 24 GB, PyTorch `mps`) and iPhone Core ML export. Flag or avoid ops that convert poorly to Core ML / Apple Neural Engine (e.g., dynamic `AdaptiveMaxPool2d`).

---

## Idea sources
- The two core papers (**PFRML**, Wang et al., IEEE TGRS 2026; and **MDNet**, Wang et al., IEEE TII 2024) are the spine: the final model must visibly implement their core ideas.
- The other 17 papers in `papers/markdown/` are a toolbox. You may propose and use ideas from them. Read `docs/related_work_notes.md` (to be created) first; open a full paper only when needed, one at a time, and tell the user which.
- Every borrowed idea gets a config toggle and its own ablation, is cited (paper + section) in the code docstring and in `docs/SPEC.md`, and never replaces the spine.
- Small toggled ideas: you may add them yourself. Anything that changes the architecture or costs more than ~half a day: propose it to the user first.
