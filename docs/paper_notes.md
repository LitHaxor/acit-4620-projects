> Agent-written notes created during development. Paper citations, section numbers, and equation references were written by the coding agent and are not all independently verified against the paper text.

# Spine Paper Notes (`PFRML` & `MDNet`)

> Distilled from [`docs/PLAN.md`](PLAN.md) (Sections 1, 2 & 7.1). Both papers form the architectural spine of this project.

---

## 1. PFRML — Prototype Feature Representation-Guided Multitask Learning

* **Citation:** G. Wang, J. Yang, K. Jiang, K. Gu, L.-K. Wong, W. Lin, and G. Zhai, *"A Prototype Feature Representation-Guided Multitask Learning Framework for Image-Based $\text{PM}_{2.5}$ Concentration Monitoring,"* **IEEE TGRS**, vol. 64, Art. no. 4109315, pp. 1–15, 2026. (`papers/markdown/A_Prototype_Feature_Representation-Guided_Multitask_Learning_Framework_for_Image-Based_PM2.5_Concentration_Monitoring.md`)

### Problem & Core Idea
* **Physical & statistical challenges (Sec. I, II-A, Fig. 1–2, Sec. III-D):** Fixed monitoring stations are costly and sparse (Sec. I). Atmospheric Mie scattering degrades image contrast and color saturation as $\text{PM}_{2.5}$ rises (Sec. I, Fig. 1, Sec. III-D), but real-world datasets are **long-tailed**: majority "Good"/"Moderate" gradients dominate backpropagation, scattering rare "Very Unhealthy"/"Hazardous" tail features into head-class regions (Sec. I, Fig. 2, Sec. III-C3, Fig. 7).
* **Coarse-to-fine prototype multitask paradigm (Sec. I, II-A, II-C, II-D, Fig. 4–5):** Pairs 6-class pollution level classification with continuous regression. Classification learns $K = 6$ class prototypes to pull tail samples into compact, well-separated clusters (Sec. II-A, II-C, Fig. 4, Fig. 7), regularizing the feature space for continuous estimation (Sec. II-D, Fig. 5).

### Architecture (`115.2M` params / `439.4 MB`, Sec. II-B–D, III-A3, III-E, Fig. 3, Fig. 5, Table I)
1. **CNN Hierarchical Extractor (Sec. II-B, Table I):** ResNet-50 stem + `Layer1`–`Layer4` on $224 \times 224$ input yields multi-scale maps $C_1 [B,256,56,56]$, $C_2 [B,512,28,28]$, $C_3 [B,1024,14,14]$, and $C_4 [B,2048,7,7]$.
2. **Dimension Consistency Module (DCM, Sec. II-B, Fig. 3, Table I):** Projects $C_1\dots C_4$ to $D=224$ channels via $1 \times 1$ conv; spatially aligns all stages to $14 \times 14$ using $4 \times 4$ MaxPool ($C_1$), $2 \times 2$ MaxPool ($C_2$), $1 \times 1$ MaxPool ($C_3$), and $\times 2$ bilinear upsampling + $1 \times 1$ MaxPool ($C_4$), followed by norm + avg-pool. *Text/Table discrepancy:* Sec. II-B prose says "concatenated", whereas **Table I (`DCM Stage 5`)** specifies `Element-wise Summation` $\to [B, 224, 14, 14]$.
3. **Transformer Global Encoder (Sec. II-B, Table I):** Flattens $14 \times 14$ ($M=196$ tokens) + prepends `[CLS]` ($197 \times 224$) + 2D learnable positional encoding $P \in \mathbb{R}^{197 \times 224}$ $\to$ 12-layer ViT ($\text{Dim}=224$) $\to$ Global Average Pooling $\to \chi_i \in \mathbb{R}^{224}$.
4. **Prototype Classification Branch (Sec. II-C, Eq. 1–3, Fig. 4–5, Table II):** Learns $K=6$ prototypes $C'=\{c'_1,\dots,c'_6\}$ updated as cluster means $c'_k = \frac{1}{|S_k|}\sum_{\chi_i \in S_k}\chi_i$ (Eq. 2). Computes squared Euclidean distance $d(\chi_i, c'_k) = \|\chi_i - c'_k\|_2^2$ (Eq. 1) and negative-distance Softmax probabilities $p(y=k \mid \chi_i) = \text{Softmax}(-d(\chi_i, c'_k))$ (Eq. 3) over 6 pollution bins (Table II).
5. **Regression Branch $\hbar(\cdot)$ (Sec. II-D, III-A3, Fig. 5, Fig. 7):** Single 3-layer MLP (`1024 -> 256 -> 1`, Sec. III-A3). *Ambiguity:* Eq. (5) & Fig. 5 write $\hbar(\chi_i)$ directly on $\chi_i$, while Sec. I, II-A, and Fig. 7 legend cite `"prototype-weighted feature"` guiding regression.

### Losses, Training & Key Results
* **Hybrid loss (Sec. II-D, Eq. 4–6, Sec. III-C5, Fig. 9):** $L = L_{\text{cro}} + \beta L_{\text{mse}}$ combining multiclass cross-entropy $L_{\text{cro}}$ (Eq. 4) and MSE $L_{\text{mse}}$ (Eq. 5; published PDF has a typo minus sign), with $\beta = 0.5$ selected over $\beta \in [0.1, 1.0]$ (Sec. III-C5, Fig. 9). Cross-domain adaptation adds $L_{\text{batch}}^{\text{MMD}} + L_{\text{city}}^{\text{MMD}}$ (Sec. III-B2).
* **Training & datasets (Sec. III-A1, III-A3):** PyTorch (RTX 4090, bs=16, AdamW, lr $2\times 10^{-4}$ decaying $10\times$ every 3 epochs, Sec. III-A3). Evaluated on **BUT** (750 Beijing images, 70/10/20 random), **India–Nepal** (12,240 daylight images, 8 cities: 5 train / 1 val / 2 unseen test cities), and **NTU** (1,300 RGB-FIR pairs, 70/10/20 random).
* **Results & interpretability (Sec. III-A2, III-B–D, Tables III–VIII, Fig. 7, Fig. 10):**
  * Intradataset SOTA (Table III): BUT ($\text{PLCC}=0.950, \text{SRCC}=0.912, \text{RMSE}=11.530$), India–Nepal unseen cities ($\text{PLCC}=0.734, \text{SRCC}=0.751, \text{RMSE}=62.269$, vs. PE-ResNet18 $0.269$ and PE-SwinT $0.356$), NTU ($\text{PLCC}=0.907, \text{RMSE}=12.554$).
  * Ablations: CNN+ViT beats CNN-only ($\text{PLCC } 0.483 \to 0.734$, Table V); DCM beats single-scale/concat ($\text{RMSE } 80.932 \to 71.685 \to 62.269$, Table VI); Prototypes boost Hazardous tail $F_1$ from $0.461 \to 0.667$ (BUT) and $0.482 \to 0.616$ (India–Nepal) (Table VII); Prototypes 1–6 monotonically track physical degradation—Sobel contrast drops $0.0384 \to 0.0214$ and HSV saturation drops $0.3212 \to 0.0986$ (Sec. III-D, Table VIII); spatial attention focuses on sky/distant scenes (Fig. 10).

### Limitations & Project Adaptation
* **Limitations (Sec. III-B2, III-E):** Heavyweight `115.2M` footprint unsuitable for mobile (Sec. III-E); zero-shot India–Nepal $\to$ NTU PLCC drops to $0.318$ ($0.395$ w/ MMD, Table IV, Sec. III-E); omits total epochs, prototype update cadence, and distance scaling/normalization.
* **In our project:** Replaces ResNet-50 with `MDNet-T`, uses DCM `Element-wise Summation` (Table I) with static pooling (Core ML safe) + 2-layer ViT ($\text{Dim}=224$), normalizes/temperature-scales prototype distances, feeds `[\chi_i; \tilde{c}_i]` to the single regression MLP, and scales AQI by $1/100$ so $\beta=0.5$ balances $L_{\text{cls}}$ and $L_{\text{reg}}$.

---

## 2. MDNet — Lightweight Multiscale Dilated CNN

* **Citation:** Z. Wang, Y. Yang, and F. Wu, *"A Lightweight Air Quality Monitoring Method Based on Multiscale Dilated Convolutional Neural Network,"* **IEEE TII**, vol. 20, no. 12, pp. 14184–14192, Dec. 2024. (`papers/markdown/A_Lightweight_Air_Quality_Monitoring_Method_Based_on_Multiscale_Dilated_Convolutional_Neural_Network.md`)

### Problem & Core Idea
* **Mobile/edge constraint & global haze physics (Sec. I, II, III-A, IV-D, Table III):** Accurate air quality models like DOViT (`58.1M` params, `11.3G` FLOPs) and ConvNeXt-S (`49.5M`, `8.7G`) are too large for smartphones and IoT edge devices (Sec. I, Table III). Haze is a global atmospheric signal spanning sky and building light reflections rather than localized objects (Sec. I, III-A, IV-D).
* **Multiscale dilated convolutions (Sec. I, III-A, III-B, Fig. 1, Eq. 3a–3b, 6–9):** Replaces ConvNeXt's $7 \times 7$ depthwise kernel with three parallel dilated depthwise branches fused by addition. Dilation $d$ expands effective kernel size $k' = k + (k - 1)(d - 1)$ (Eq. 3a), grows cumulative receptive field to $\text{RF}=2288$ (`MDNet`) or $1136$ (`MDNet-T`), and reduces parameters by $R = 1 - \lceil k/d \rceil^2 / k^2$ (Eq. 3b, 6–9, Table III).

### Architecture (Sec. III-A, III-B, Fig. 1–2, Fig. 4, Table I, III)
1. **Four-Stage Backbone (Fig. 1a–b, Table I, III):** Accepts $224 \times 224 \times 3$ input; patchify stem ($4 \times 4$ conv, stride $4$) $\to 56 \times 56 \times 96$. Four stages with widths $[96, 192, 384, 768]$ and spatial sizes $[56\times 56, 28\times 28, 14\times 14, 7\times 7]$, linked by `Down Conv` (`LayerNorm` + $2\times 2$ conv, stride $2$; Fig. 1b). Ends with GAP $\to$ `LayerNorm` $\to$ Linear head (Fig. 1a).
   * **`MDNet-T`:** depths $[3, 3, 9, 3]$ $\implies$ **`4.9M` params**, **`0.7G` FLOPs**, **$\text{RF}=1136$** (Table I, III).
   * **`MDNet`:** depths $[3, 3, 27, 3]$ $\implies$ **`7.7M` params**, **`1.3G` FLOPs**, **$\text{RF}=2288$** (Table I, III).
2. **Inside the MD Block (Fig. 1c–d, Fig. 2f, Table I, Eq. 1, 11–12):** Three parallel depthwise (`groups=C`) branches summed element-wise (`Map_add`, Eq. 1):
   * Branch 1: $k=3, d=2, s=1, p=2 \implies k'=5$ ($d5\times 5$ in Table I).
   * Branch 2: $k=2, d=2, s=1, p=1 \implies k'=3$ ($d3\times 3$ in Table I; samples 4 corners of $3\times 3$).
   * Branch 3: $k=1, d=1, s=1, p=0 \implies k'=1$.
   * Summed features pass through `GELU` (Eq. 11–12), pointwise $1 \times 1$ conv ($C \to C$), `LayerNorm`, and residual shortcut ($\oplus$).
3. **Negative Result — Intra-Block Self-Attention (Sec. III-B3, Fig. 4, Eq. 10, Table III):** Replacing `Map_add` with QKV self-attention inside the MD block drops accuracy from **$94.2\%$ to $52.6\%$** (`MDNet`) and **$50.0\%$** (`MDNet-T`) because intra-block attention mixes foreground/background clutter with subtle haze cues (Sec. III-B3, IV-B).

### Loss, Augmentations & Key Results
* **Label Smoothing Cross-Entropy (Sec. III-A, Eq. 2, Table II):** Over $C=6$ AQI levels (`0–50`, `51–100`, `101–150`, `151–200`, `201–300`, `>300`, Table II) with soft targets $1-\varepsilon$ and $\varepsilon/(C-1)$ to mitigate boundary label noise.
* **Dataset & Physics-Preserving Augmentation Rule (Sec. III-C, IV-A, Fig. 5–6, Table II, VII):** Evaluated on **GAOs-2** (1,054 Beijing mobile daylight photos, top sky / bottom ground, 4 weather types; Fig. 6, Table II), resized to $224\times 224$ and normalized (Sec. IV-A). **Strictly bans `ColorJitter` and `Grayscale`** (Sec. III-C, p. 14189: *"because the AQI information is hidden in it"*); tests geometric/blur robustness via perspective, rotation, masking, and Gaussian blur (Fig. 5, Table VII).
* **Results & ablations (Sec. IV-B–D, Tables III–VII, Fig. 7):**
  * **94.2% top-1 accuracy** (`7.7M` params, `1.3G` FLOPs, `24.6 ms` latency), beating DOViT (`92.8%`, `58.1M`), EfficientNetV2-S (`92.5%`, `22.1M`), ConvNeXt-S (`91.7%`, `49.5M`), and MobileNetV3-S (`90.6%`, `2.5M`) (Table III).
  * Three dilated scales $(1,2,3)$ (`94.2%`) beat two-scale models (`89.3–89.8%`) and dense $3\times 3$ `MDNet-133` (`89.7%`) (Table IV); `GELU` beats `ReLU` by $+2.4\%$ (`94.2%` vs. `91.8%`, Table VI). Grad-CAM (Fig. 7, Sec. IV-D) confirms attention covers both sky and building light reflections.

### Limitations & Project Adaptation
* **Limitations (Sec. IV-A, IV-D):** Classification-only (no continuous AQI prediction, Sec. IV-D); small single-city dataset (1,054 Beijing images, no station-grouped split mentioned, Sec. IV-A, IV-D); omits optimizer, LR, batch size, epochs, split ratio, and $\varepsilon$.
* **In our project:** `MDNet-T` (`4.9M` params) provides the exact $[56, 28, 14, 7]$ hierarchy for PFRML's DCM, enforces the `ColorJitter`/`Grayscale` ban in `src/data/transforms.py`, uses EPA 6-bin AQI thresholds (Table II) + Label Smoothing CE (Eq. 2), and retains an optional sparse $3\times 3$ weight expansion for $(k=2, d=2, p=1)$ during Core ML export.
