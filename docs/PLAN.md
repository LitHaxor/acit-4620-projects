# ACIT4620 Project Plan: Explainable, Mobile-Ready PM2.5 AQI Estimation from Street-Level Imagery

> **Status:** Read-Only Analysis & Architecture Plan (Awaiting Approval)  
> **Target Hardware:** MacBook Pro (M5 Pro, 24 GB Unified Memory, PyTorch `mps` backend) & iPhone Core ML export target  
> **Dataset:** PM25Vision (`datasets/train` and `datasets/test`)  
> **Primary Literature:**
> 1. **Paper 1 (PFRML):** G. Wang, J. Yang, K. Jiang, K. Gu, L.-K. Wong, W. Lin, and G. Zhai, *"A Prototype Feature Representation-Guided Multitask Learning Framework for Image-Based $\text{PM}_{2.5}$ Concentration Monitoring,"* **IEEE Transactions on Geoscience and Remote Sensing (TGRS)**, vol. 64, Art. no. 4109315, pp. 1–15, 2026. ([PDF](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/papers/A_Prototype_Feature_Representation-Guided_Multitask_Learning_Framework_for_Image-Based_PM2.5_Concentration_Monitoring.pdf) | [Markdown](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/papers/markdown/A_Prototype_Feature_Representation-Guided_Multitask_Learning_Framework_for_Image-Based_PM2.5_Concentration_Monitoring.md))
> 2. **Paper 2 (MDNet):** Z. Wang, Y. Yang, and F. Wu, *"A Lightweight Air Quality Monitoring Method Based on Multiscale Dilated Convolutional Neural Network,"* **IEEE Transactions on Industrial Informatics (TII)**, vol. 20, no. 12, pp. 14184–14192, Dec. 2024. ([PDF](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/papers/A_Lightweight_Air_Quality_Monitoring_Method_Based_on_Multiscale_Dilated_Convolutional_Neural_Network.pdf) | [Markdown](file:///Users/darth/Projects/ACIT4620/acit-4620-projects/papers/markdown/A_Lightweight_Air_Quality_Monitoring_Method_Based_on_Multiscale_Dilated_Convolutional_Neural_Network.md))

---

## 1. Plain-Language Summary of the Two Core IEEE Papers

### 1.1 Paper 1: Prototype Feature Representation-Guided Multitask Learning (PFRML, IEEE TGRS 2026)

* **Problem Statement (Sec. I, II-A, Fig. 2):**  
  Physical $\text{PM}_{2.5}$ sensors (gravimetric, $\beta$-ray absorption, tapered element oscillating microbalance) are accurate but expensive and geographically sparse (Sec. I). Photographs capture atmospheric light scattering and absorption (Mie scattering), which visibly degrade image contrast and color saturation as particulate concentration rises (Sec. I, Fig. 1, Sec. III-D). However, existing deep learning estimators struggle in complex outdoor scenes and fail on high-pollution days because real-world air quality data follows a **long-tailed distribution**: most images belong to "Good" or "Moderate" head classes, while "Very Unhealthy" and "Hazardous" tail events are rare (Sec. I, Fig. 2). During backpropagation, gradients from majority classes dominate parameter updates, causing tail-class representations to scatter into head-class boundaries (Sec. I, Sec. III-C3, Fig. 7).

* **Core Method (Sec. I, II-A, II-C, II-D):**  
  PFRML turns single-task regression into a **coarse-to-fine multitask learning paradigm** combining 6-class pollution level classification with continuous regression (Sec. I, II-C). Because coarse category classification is easier to optimize than exact continuous regression, the classification branch uses **prototype learning**—learning $K=6$ data-driven cluster centers ("prototypes") in feature space—to pull rare tail-class samples into compact, well-separated clusters (Sec. II-A, II-C, Fig. 4, Fig. 7). The regression branch then predicts continuous pollution levels from this regularized feature space (Sec. II-D, Fig. 5).

* **Architecture (Sec. II-B, Fig. 3, Fig. 5, Table I):**
  1. **CNN Hierarchical Extractor (Sec. II-B, Table I):** An input image ($224 \times 224$) passes through a ResNet-50 stem and 4 stages (`Layer1`–`Layer4`), producing multi-scale feature maps $C_1 \in \mathbb{R}^{B \times 256 \times 56 \times 56}$, $C_2 \in \mathbb{R}^{B \times 512 \times 28 \times 28}$, $C_3 \in \mathbb{R}^{B \times 1024 \times 14 \times 14}$, and $C_4 \in \mathbb{R}^{B \times 2048 \times 7 \times 7}$. Shallow layers capture fine local edge/contrast distortions, while deeper layers capture semantic context.
  2. **Dimension Consistency Module (DCM) (Sec. II-B, Fig. 3, Table I):** Because $C_1 \dots C_4$ have different spatial sizes and channel counts, each stage is projected to 224 channels via a $1 \times 1$ convolution, spatially aligned to $14 \times 14$ (using $4 \times 4$ MaxPool on $C_1$, $2 \times 2$ MaxPool on $C_2$, $1 \times 1$ MaxPool on $C_3$, and $\times 2$ bilinear upsampling + $1 \times 1$ MaxPool on $C_4$), normalized, and average-pooled into four $[B, 224, 14, 14]$ tensors. *(Note: Sec. II-B prose states the unified features are "concatenated", whereas Table I `DCM Stage 5` explicitly specifies `Input [B,224,14,14]x4 -> Element-wise Summation -> Output [B,224,14,14]`.)*
  3. **Transformer Global Encoder (Sec. II-B, Table I):** The $14 \times 14$ feature map ($M = 196$ patches of dimension $C = 224$) is flattened, prepended with a `[CLS]` token ($197$ tokens total), summed with a 2D learnable positional encoding $P \in \mathbb{R}^{197 \times 224}$, and processed by a 12-layer Vision Transformer (ViT) encoder ($\text{Dim} = 224$) to capture global long-range dependencies across the scene. Global average pooling produces the final image embedding $\chi_i \in \mathbb{R}^{224}$ (Sec. II-B).
  4. **Prototype Classification & MLP Regression Heads (Sec. II-C, II-D, III-A3, Fig. 5):**
     * **Prototype Branch:** Computes squared Euclidean distance $d(\chi_i, c'_k) = \|\chi_i - c'_k\|_2^2$ (Eq. 1) between $\chi_i$ and $K=6$ category prototypes $C' = \{c'_1, \dots, c'_6\}$, where each prototype is iteratively updated as the mean of assigned cluster samples $c'_k = \frac{1}{|S_k|} \sum_{\chi_i \in S_k} \chi_i$ (Eq. 2). Distances are converted to class probabilities via a negative-distance Softmax:
       $$p(y = k \mid \chi_i) = \frac{\exp(-d(\chi_i, c'_k))}{\sum_{j=1}^K \exp(-d(\chi_i, c'_j))} \quad \text{(Paper 1, Eq. 3)}$$
     * **Regression Branch $\hbar(\cdot)$:** A 3-layer MLP with hidden/output widths $1024 \to 256 \to 1$ estimates continuous concentration (Sec. II-D, Sec. III-A3). *(Note: Fig. 5 and Eq. 5 write $\hbar(\chi_i)$ directly on the shared feature $\chi_i$, while Sec. I, II-A, and the legend of Fig. 7 also refer to "prototype-weighted feature" guiding regression.)*

* **Loss Functions (Sec. II-D, Eq. 4–6, Sec. III-B2):**
  * **Classification Loss ($L_{\text{cro}}$, Eq. 4):** Standard multiclass cross-entropy over the 6 pollution pseudo-labels (Table II):
    $$L_{\text{cro}} = -\frac{1}{N} \sum_{i=1}^N \log p(y = y_i \mid \chi_i)$$
  * **Regression Loss ($L_{\text{mse}}$, Eq. 5):** Mean Squared Error between predicted $\hbar(\chi_i)$ and ground-truth $\hat{y}_i$:
    $$L_{\text{mse}} = \frac{1}{N} \sum_{i=1}^N \left(\hbar(\chi_i) - \hat{y}_i\right)^2$$
    *(Note: Eq. 5 in the published PDF contains a typographical minus sign in front of the sum; MSE is strictly non-negative during minimization.)*
  * **Multitask Hybrid Loss (Eq. 6):** $L = L_{\text{cro}} + \beta \cdot L_{\text{mse}}$, with $\beta = 0.5$ chosen via grid ablation over $\beta \in [0.1, 1.0]$ (Sec. III-C5, Fig. 9).
  * **Domain Adaptation Extension (Sec. III-B2):** For cross-dataset generalization, adds batch-level and city-level Maximum Mean Discrepancy regularizers: $L' = L_{\text{cro}} + \beta L_{\text{mse}} + L_{\text{batch}}^{\text{MMD}} + L_{\text{city}}^{\text{MMD}}$.

* **Training Setup & Datasets (Sec. III-A1, III-A3):**
  * Trained in PyTorch on an RTX 4090 (24 GB), batch size 16, AdamW optimizer, initial learning rate $2 \times 10^{-4}$ decaying by $10\times$ every 3 epochs (Sec. III-A3).
  * Evaluated on 3 datasets: **BUT** (750 Beijing images, $1\text{--}423\ \mu\text{g/m}^3$, 70/10/20 random split), **India–Nepal** (12,240 daylight images across 8 cities; **strict geographical split**: 5 cities train, 1 city val, 2 unseen cities test), and **NTU** (1,300 RGB-FIR pairs at Nantong Univ., 70/10/20 random split).

* **Metrics & Key Results (Sec. III-A2, III-B, III-C, III-D, Tables III–VIII):**
  * Metrics: Pearson Linear Correlation Coefficient (PLCC $\uparrow$), Spearman Rank-Order Correlation Coefficient (SRCC $\uparrow$), Root Mean Squared Error (RMSE $\downarrow$), and per-class $F_1$-score ($\uparrow$).
  * Intradataset SOTA (Table III): BUT ($\text{PLCC}=0.950, \text{SRCC}=0.912, \text{RMSE}=11.530$), India–Nepal unseen cities ($\text{PLCC}=0.734, \text{SRCC}=0.751, \text{RMSE}=62.269$, beating PE-ResNet18's $\text{PLCC}=0.269$ and PE-SwinT's $0.356$), NTU ($\text{PLCC}=0.907, \text{SRCC}=0.899, \text{RMSE}=12.554$).
  * Ablations prove that: (1) CNN+Transformer beats CNN-only ($\text{PLCC } 0.483 \to 0.734$ on India–Nepal, Table V); (2) DCM beats single-scale and raw concatenation ($\text{RMSE } 80.932 \to 71.685 \to 62.269$ on India–Nepal, Table VI); (3) Prototype classification boosts "Hazardous" tail-class $F_1$ from $0.461 \to 0.667$ (BUT) and $0.482 \to 0.616$ (India–Nepal) (Table VII); and (4) Learned prototypes 1–6 monotonically track physical image degradation—Sobel gradient contrast drops from $0.0384 \to 0.0214$ and HSV saturation drops from $0.3212 \to 0.0986$ as pollution rises (Sec. III-D, Table VIII).

* **Limitations (Sec. III-B2, III-E):**
  1. **Massive Footprint:** **115.2 million parameters (439.4 MB)** (Sec. III-E), which is prohibitive for mobile/iPhone deployment and slow for 5-fold cross-validation on a laptop.
  2. **Cross-Domain Degradation:** Training on India–Nepal and testing zero-shot on NTU drops PLCC to $0.318$ ($0.395$ with MMD, Table IV) due to regional scene, sensor, and climate shifts (Sec. III-E).
  3. **Omitted Hyperparameters & Text/Table Discrepancies:** Total training epochs, exact K-means prototype update frequency (per-epoch vs. per-batch), and how prototype features are fused into the regression head are underspecified in the text.

---

### 1.2 Paper 2: Lightweight Multiscale Dilated CNN (MDNet, IEEE TII 2024)

* **Problem Statement (Sec. I, II):**  
  High-performing vision models for air quality monitoring—such as Vision Transformers (DOViT, 58.1M params, 11.3G FLOPs) and ConvNeXt-S (49.5M params, 8.7G FLOPs)—are too large and computationally expensive to run on mobile phones or IoT edge devices (Sec. I, Table III). Furthermore, air quality estimation is a global, fine-grained perception task where haze information is distributed across the sky and building light reflections rather than localized object boundaries (Sec. I, III-A, IV-D).

* **Core Method (Sec. I, III-A, III-B):**  
  MDNet modifies the ConvNeXt architecture by replacing its standard $7 \times 7$ depthwise square convolution kernel with a **Multiscale Dilated (MD) Block** that fuses **three parallel dilated convolution kernels** at different scales via element-wise addition (Sec. I, III-A, Fig. 1). Dilated convolutions insert spacing ($d$) between kernel weights, expanding the effective kernel size to $k' = k + (k - 1)(d - 1)$ (Eq. 3a) and growing the network's cumulative receptive field to $\text{RF} = 2288$ (for MDNet) or $\text{RF} = 1136$ (for MDNet-T) while saving parameters according to $R = 1 - \lceil k/d \rceil^2 / k^2$ (Eq. 3b, 6–9, Table III).

* **Architecture (Sec. III-A, III-B, Fig. 1, Fig. 2, Table I):**
  1. **Four-Stage Hierarchical Backbone (Fig. 1a, Table I):** Accepts a $224 \times 224 \times 3$ image. A patchify stem ($4 \times 4$ Conv, stride 4) produces a $56 \times 56 \times 96$ feature map. Four stages follow with channel widths $[96, 192, 384, 768]$ and spatial sizes $[56\times 56, 28\times 28, 14\times 14, 7\times 7]$. Between stages, a `Down Conv` block (Fig. 1b) applies `LayerNorm` followed by a $2 \times 2$ convolution with stride 2.
  2. **Stage Depths & Parameter Math (Table I, Table III):**
     * **MDNet-T:** Stage block counts $[3, 3, 9, 3]$ $\implies$ **4.9M parameters**, **0.7G FLOPs**, $\text{RF} = 1136$.
     * **MDNet:** Stage block counts $[3, 3, 27, 3]$ $\implies$ **7.7M parameters**, **1.3G FLOPs**, $\text{RF} = 2288$.
     * *(Mathematical verification of parameter count: Using depthwise `groups=C` dilated branches + a single $1 \times 1$ channel-mixing conv $C \to C$ per block as drawn in Fig. 1c yields exactly $4.88\text{M}$ params for $[3,3,9,3]$ and $7.63\text{M}$ params for $[3,3,27,3]$, matching Table I's $4.9\text{M}$ and $7.7\text{M}$!)*
  3. **Inside the MD Block (Fig. 1c, Fig. 1d, Fig. 2f, Table I):**
     * Three parallel depthwise convolution branches (`groups=C`) operating on the input tensor and fused by element-wise addition (`Map_add`, Eq. 1):
       * Branch 1: $k = 3, d = 2, s = 1, p = 2 \implies$ effective kernel size $k' = 3 + (3-1)(2-1) = 5$ (denoted $d5 \times 5$ in Table I).
       * Branch 2: $k = 2, d = 2, s = 1, p = 1 \implies$ effective kernel size $k' = 2 + (2-1)(2-1) = 3$ (denoted $d3 \times 3$ in Table I; samples the 4 corners of a $3 \times 3$ neighborhood, preserving exact spatial dimensions $F_{\text{out}} = F_{\text{in}}$ under symmetric padding $p = 1$).
       * Branch 3: $k = 1, d = 1, s = 1, p = 0 \implies$ effective kernel size $k' = 1$ ($1 \times 1$ center weight).
     * The summed multi-scale feature map passes through `GELU` activation (Eq. 11–12), a pointwise $1 \times 1$ convolution ($k=1, d=1, s=1, p=0$), `LayerNorm`, and a residual shortcut connection ($\oplus$).
     * The network ends with Global Average Pooling $\to$ `LayerNorm` $\to$ Linear classification head (Fig. 1a).
  4. **Crucial Negative Result on Self-Attention Inside MD Block (Sec. III-B3, Fig. 4, Table III):**  
     When the authors replaced the element-wise addition of the three dilated branches with a QKV self-attention fusion module (`MDNet + Self-attention`, Fig. 4, Eq. 10), accuracy collapsed from **94.2% to 52.6%** (and $50.0\%$ for `MDNet-T + Self-attention`), because channel/patch attention inside the convolutional block mixed foreground/background clutter with subtle atmospheric haze cues (Sec. III-B3, IV-B).

* **Loss Function (Sec. III-A, Eq. 2):**  
  Label Smoothing Cross-Entropy over $C = 6$ AQI categories (`0–50`, `51–100`, `101–150`, `151–200`, `201–300`, `>300`, Table II) with smoothing constant $\varepsilon$ to prevent overconfidence and mitigate boundary label noise:
  $$L(p, q) = -\sum_{i=1}^C q_i \log \frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}}, \quad q_i = \begin{cases} 1 - \varepsilon, & i = y \\ \frac{\varepsilon}{C - 1}, & \text{otherwise} \end{cases} \quad \text{(Paper 2, Eq. 2)}$$

* **Training Setup, Dataset & Augmentations (Sec. III-C, IV-A):**
  * Evaluated on **GAOs-2**: 1,054 raw mobile phone daylight photos (top half sky, bottom half ground/buildings) taken in Beijing across 4 weather conditions (sunny, cloudy, haze, overcast) and 6 AQI levels (Sec. IV-A, Fig. 6, Table II).
  * Standard training uses $224 \times 224$ resizing and normalization (Sec. IV-A).
  * **Physics-Preserving Augmentation Rule (Sec. III-C, p. 14189):** Explicitly forbids `ColorJitter` and `Grayscale` *"because the AQI information is hidden in it."* Evaluates robustness under geometric/structural perturbations: perspective transform, random rotation, random masking, and Gaussian blur (Fig. 5, Table VII).

* **Metrics & Key Results (Sec. IV-B, IV-C, IV-D, Tables III–VII):**
  * Metrics: Top-1 Accuracy (%), Parameters (M), FLOPs (G), Latency (ms), Receptive Field.
  * MDNet achieves **94.2% top-1 accuracy** (7.7M params, 1.3G FLOPs, 24.6 ms latency), outperforming DOViT (92.8%, 58.1M params), EfficientNetV2-S (92.5%, 22.1M params), ConvNeXt-S (91.7%, 49.5M params), and MobileNetV3-S (90.6%, 2.5M params) (Table III).
  * Ablations show: (1) Three dilated scales ($1, 2, 3$ in MDNet, 94.2%) beat two scales (MDNet-13: 89.3%, MDNet-23: 89.3%, MDNet-33: 89.8%) and dense $3\times 3$ replacements (MDNet-133: 89.7%) (Table IV); (2) `GELU` beats `ReLU` by $+2.4\%$ ($94.2\%$ vs. $91.8\%$, Table VI); (3) Grad-CAM heatmaps (Fig. 7, Sec. IV-D) show MDNet attends equitably to both the sky and light reflections on building facades.

* **Limitations (Sec. IV-D):**
  1. **Classification Only:** Does not predict continuous AQI values (Sec. IV-D).
  2. **Single-City Small Dataset:** Trained and tested on 1,054 images from Beijing only, with no mention of station-grouped splitting (Sec. IV-A, IV-D).
  3. **Omitted Training Hyperparameters:** Does not report optimizer, learning rate, batch size, epoch count, train/test split ratio, or exact $\varepsilon$ value.

---

## 2. Synthesis: Mapping Both Papers to Our Implementation

To satisfy the hard requirement of visibly reproducing the **core ideas of both papers** while scaling to a **MacBook Pro (M5 Pro, 24 GB RAM, `mps`)** and **Core ML export**, we unify them naturally: **Paper 2's lightweight `MDNet-T` (4.9M params)** replaces Paper 1's heavy ResNet-50 (23.5M params) as the 4-stage hierarchical CNN backbone, feeding into **Paper 1's Dimension Consistency Module (DCM), lightweight Transformer encoder, Prototype classification branch, and MLP regression branch**.

| Component / Idea | Source Paper & Section | Original Paper Specification | What We Will Implement | What is Simplified / Different & Why |
| :--- | :--- | :--- | :--- | :--- |
| **4-Stage Multiscale Dilated CNN Backbone (`MDNet-T`)** | **Paper 2**, Sec. III-A, III-B1, Fig. 1a–d, Table I | 4 stages with widths $[96, 192, 384, 768]$; depths $[3, 3, 9, 3]$ (`MDNet-T`, 4.9M params) or $[3, 3, 27, 3]$ (`MDNet`, 7.7M params); `Down Conv` ($2\times 2, s=2$). | Implement configurable `MDNetBackbone` defaulting to **`MDNet-T` ($[3, 3, 9, 3]$, 4.9M params)** with option for full `MDNet` ($[3, 3, 27, 3]$). | Uses `MDNet-T` instead of Paper 1's ResNet-50 (23.5M) to cut CNN parameters by $\sim 79\%$ and speed up 5-fold CV on M5 Pro while keeping the exact $[56, 28, 14, 7]$ spatial hierarchy required by Paper 1's DCM. |
| **Multiscale Dilated (MD) Block** | **Paper 2**, Sec. III-A, III-B1, Eq. (1), (3a), (11), Fig. 1c, Table VI | 3 parallel depthwise branches ($k=3,d=2,p=2$; $k=2,d=2,p=1$; $k=1,d=1,p=0$) fused by addition (`Map_add`), `GELU`, $1\times 1$ Conv ($C\to C$), `LayerNorm`, residual shortcut. | Exact reproduction of the 3-branch depthwise dilated sum + `GELU` + $1\times 1$ Conv + `LayerNorm` + residual shortcut. | **Core ML safeguard:** For the $k=2, d=2, p=1$ branch (which samples the 4 corners of a $3\times 3$ grid), we also support an equivalent sparse $3\times 3$ kernel view at export time if `coremltools` ANE compilation flags even-sized dilated kernels. |
| **Dimension Consistency Module (DCM)** | **Paper 1**, Sec. II-B, Fig. 3, Table I | Projects $C_1, C_2, C_3, C_4$ to $D=224$ via $1\times 1$ Conv; aligns all to $14\times 14$ via MaxPool ($4\times 4, 2\times 2, 1\times 1$) and Bilinear Upsample ($\times 2$ on $C_4$); Norm + AvgPool; element-wise sum (Table I) or concat (text). | Project `MDNet-T` stages $[96, 192, 384, 768] \to D=224$; align to $14\times 14$ using fixed-stride `MaxPool2d(4,4)`, `MaxPool2d(2,2)`, `Identity`, and `Upsample(scale_factor=2, mode='bilinear')`; fuse into $[B, 224, 14, 14]$. | **Core ML safeguard:** Replaces `AdaptiveMaxPool2d` with static `MaxPool2d(kernel_size=4, stride=4)` etc. (identical for $224\times 224$ input, 100% Core ML compatible). Follows **Table I** (`Element-wise Summation` to $[B, 224, 14, 14]$) with a config toggle for concatenation + $1\times 1$ projection. |
| **CNN-Transformer Hybrid Global Encoder** | **Paper 1**, Sec. II-B, Fig. 3, Table I | Flatten $[B, 224, 14, 14] \to 196$ tokens + `[CLS]` token ($197 \times 224$) + 2D learnable positional encoding $P$; 12-layer ViT encoder ($\text{Dim}=224$); Global Average Pooling $\to \chi_i \in \mathbb{R}^{224}$. | Same $197 \times 224$ tokenization + learnable positional encoding + Transformer Encoder ($\text{Dim}=224$, 4 or 8 heads) + Global Average Pooling $\to \chi_i \in \mathbb{R}^{224}$. | **Scaled down for laptop & mobile (My Assumption):** Use **2 Transformer layers** (configurable up to 12) at $\text{Dim}=224, \text{MLP ratio}=2.0\text{--}4.0$. Captures global sky-to-ground context (Paper 1 Sec. II-B) while keeping total model size $\sim 6\text{--}7\text{M}$ params instead of 115.2M. |
| **Pollution Level Pseudo-Labels ($K=6$)** | **Paper 1**, Sec. II-C, Table II; **Paper 2**, Sec. III-A, Table II | Paper 1 Table II uses Chinese $\text{PM}_{2.5}$ concentration bins ($0\text{--}35\dots >250\ \mu\text{g/m}^3$). Paper 2 Table II uses the 6 standard **AQI** bins (`0–50`, `51–100`, `101–150`, `151–200`, `201–300`, `>300`). | Use the **6 US EPA AQI bins** (`0–50`, `51–100`, `101–150`, `151–200`, `201–300`, `301–600`) already present in `pm25_bin` and matching **Paper 2 Table II**. | PM25Vision's target column `pm25` is on the US EPA AQI scale ($1\text{--}530$), not $\mu\text{g/m}^3$. Paper 1 Sec. II-C explicitly states that the 6 thresholds *"can be flexibly adjusted to accommodate local air quality policies"* without architectural change. |
| **Prototype Feature Extraction & Distance Softmax** | **Paper 1**, Sec. II-C, II-D, Eq. (1)–(3), Fig. 4, Fig. 5 | $K=6$ prototypes $C'=\{c'_1,\dots,c'_6\} \in \mathbb{R}^{6\times D}$ initialized/updated via K-means centroids $c'_k = \frac{1}{|S_k|}\sum_{\chi_i \in S_k}\chi_i$ (Eq. 2); squared Euclidean distance $d(\chi_i, c'_k) = \|\chi_i - c'_k\|_2^2$ (Eq. 1); probability $p(y=k|\chi_i) = \text{Softmax}(-d(\chi_i, c'_k) / \tau)$ (Eq. 3). | Store $C' \in \mathbb{R}^{6\times 224}$ as a buffer/parameter. Initialize via class-conditional K-means centroids on training features and update via class-centroid averaging / EMA each epoch (**My Assumption** on update schedule + learnable temperature/scale $\tau$). | Paper 1 does not specify whether Eq. (2) runs every batch or every epoch, nor whether features $\chi_i$ are $L_2$-normalized before squaring distances (without normalization or temperature $\tau$, $-\|\chi_i - c'_k\|_2^2$ can saturate `exp` in float32). Using $\ell_2$-normalized or temperature-scaled Euclidean distance keeps gradients stable and exports cleanly to Core ML as a frozen `[6, 224]` buffer. |
| **Prototype-Guided Regression Branch $\hbar(\cdot)$** | **Paper 1**, Sec. I, II-A, II-D, III-A3, Fig. 5, Fig. 7 | 3-layer MLP ($1024 \to 256 \to 1$, Sec. III-A3). Text/Fig. 7 notes regression leverages prototype features (`prototype-weighted feature`) to improve tail estimation. | Compute prototype-weighted feature $\tilde{c}_i = \sum_{k=1}^6 p(y=k|\chi_i) c'_k$, fuse with $\chi_i$ ($\chi_i + \tilde{c}_i$ or $[\chi_i \mathbin{\Vert} \tilde{c}_i]$), and pass to MLP head ($512 \to 128 \to 1$, scaled down from $1024 \to 256 \to 1$, **My Assumption**). | Explicitly implements the prototype-to-regression bridge described in Paper 1 Sec. I & Fig. 7 (`prototype-weighted feature`) so the classification prototypes directly guide tail-class regression predictions, while also supporting direct $\hbar(\chi_i)$ (Fig. 5) for ablation. |
| **Multitask Hybrid Loss Function** | **Paper 1**, Sec. II-D, Eq. (4)–(6), Sec. III-C5; **Paper 2**, Sec. III-A, Eq. (2) | Paper 1 Eq. (6): $L = L_{\text{cro}} + \beta L_{\text{mse}}$ with $\beta = 0.5$. Paper 2 Eq. (2): Label Smoothing Cross-Entropy ($L_{\text{lsce}}$). | Implement $L = L_{\text{lsce}}(\varepsilon=0.1) + \beta L_{\text{reg}}$, supporting both **Huber / Smooth L1** and **MSE** for $L_{\text{reg}}$ (**My Assumption** on target scaling & Huber option). | Raw AQI spans $1\text{--}530$, so unscaled MSE is $\sim 10^3\text{--}10^4$ while $L_{\text{cro}}$ is $\sim 1.0\text{--}1.8$. To make $\beta=0.5$ (Paper 1 Sec. III-C5) balance both losses as intended, we normalize the regression target (e.g., divide AQI by $100$ or z-score during loss calculation) or use Smooth L1/Huber loss. |
| **Physics-Preserving Data Augmentation** | **Paper 2**, Sec. III-C, Fig. 5, Table VII | **No color jitter or grayscale** (preserves AQI color/brightness signal). Uses random rotation, perspective, masking, or Gaussian blur for robustness. | Strict ban on `ColorJitter` and `Grayscale`. Training uses random horizontal flip and subtle geometric transforms (small rotation/perspective) from Paper 2 Sec. III-C. | Directly enforces both the user constraint and Paper 2 Sec. III-C. |
| **Metadata Fusion & Shortcut Prevention** | **User Constraint** + **Paper 1**, Sec. III-A3 (Geographic evaluation) | Neither Paper 1 nor Paper 2 fuses GPS+Month metadata. However, in PM25Vision, GPS+Month is a massive shortcut. | Optional `MetadataMLP` branch (spherical 3D GPS $(x,y,z)$ + cyclic $\sin/\cos$ month $\to 32$-d) fused with vision features via **Modality Dropout** (dropping the entire metadata vector to $\mathbf{0}$ with probability $p_{\text{drop}} \in [0.3, 0.5]$ during training) + detached vision auxiliary loss (**My Assumption**). | Prevents the network from ignoring the image branch when GPS+Month is present, and enables rigorous ablation across Image-Only, Metadata-Only, Naive Fusion, and Modality-Dropout Fusion. |

---

## 3. Glossary of Essential ML & Math Concepts

Every concept below is explained in 2–3 plain sentences for a software engineer, naming the underlying math and where it appears in our project:

1. **Regression vs. Classification Loss:**  
   Regression predicts a continuous number $\hat{y} \in \mathbb{R}$ (such as exact AQI $= 142.0$) and penalizes the numerical distance $|\hat{y} - y|$ or $(\hat{y} - y)^2$ between prediction and truth. Classification predicts a discrete category $k \in \{1, \dots, K\}$ (such as EPA Band 3: "Unhealthy for Sensitive Groups") by outputting a probability distribution over classes and penalizing low probability on the true class. Combining both in **Multitask Learning (MTL)** (Paper 1, Eq. 6) uses the coarse, stable classification boundaries to prevent the continuous regression head from making wild errors on rare pollution spikes.
2. **L1 (MAE), L2 (MSE), and Huber (Smooth L1) Loss:**  
   **L1 loss** (Mean Absolute Error, $\frac{1}{N}\sum |\hat{y}_i - y_i|$) grows linearly with error, making it robust to extreme outliers (like AQI $= 500$), whereas **L2 loss** (Mean Squared Error, $\frac{1}{N}\sum (\hat{y}_i - y_i)^2$, Paper 1 Eq. 5) squares the error, which heavily penalizes large errors but can let a few extreme tail samples destabilize gradients. **Huber (Smooth L1) loss** combines the best of both: it acts like smooth quadratic L2 loss when the error $|e| < \delta$ is small (giving smooth gradients near zero) and switches to linear L1 loss when $|e| \ge \delta$ (preventing gradient explosions on outliers).
3. **Softmax & Cross-Entropy Loss (with Label Smoothing):**  
   **Softmax** converts a vector of raw scores (logits $z_k$ or negative distances $-d_k$) into a valid probability distribution summing to $1$ via $p_k = \exp(z_k) / \sum_j \exp(z_j)$ (Paper 1 Eq. 3). **Cross-Entropy** measures the information-theoretic distance (Kullback–Leibler divergence) between the true label distribution $q$ and predicted probabilities $p$ as $L = -\sum_{k=1}^K q_k \log p_k$ (Paper 1 Eq. 4). **Label Smoothing** (Paper 2 Eq. 2) replaces hard binary targets ($1$ for the true class, $0$ for others) with soft targets ($1 - \varepsilon$ and $\varepsilon/(K-1)$), preventing the network from becoming overconfident when images lie near the boundary between two AQI bins (e.g., AQI $50$ vs. $51$).
4. **Dilated (Atrous) Convolution:**  
   A standard $k \times k$ convolution samples adjacent pixels in a tight grid, whereas a **dilated convolution** (Paper 2 Sec. III-B, Eq. 3a) inserts $d - 1$ zeros (spaces) between kernel weights so a $k \times k$ filter spans an effective area of $k' = k + (k - 1)(d - 1)$ without adding any extra learnable weights. For example, a $3 \times 3$ kernel with dilation $d = 2$ has only $9$ weights per channel just like a standard $3 \times 3$ kernel, but covers a $5 \times 5$ pixel neighborhood.
5. **Receptive Field ($\text{RF}$):**  
   The **receptive field** of a neuron in layer $i$ is the side length (in input image pixels) of the region that can influence that neuron's activation, recurring via $\text{RF}_{i+1} = \text{RF}_i + (k' - 1) \times \prod_{j=1}^i \text{Stride}_j$ (Paper 2 Eq. 3b–3c). In air quality monitoring, a large receptive field is critical because haze is a global atmospheric property spanning the entire sky and distant building silhouettes rather than a small local object (Paper 2 Sec. III-A, IV-B).
6. **Depthwise Separable / Pointwise Convolution:**  
   Instead of applying a dense 3D filter across all $C_{\text{in}} \times C_{\text{out}} \times k^2$ connections at once, a **depthwise convolution** (`groups = C`) applies a separate spatial $k \times k$ filter to each channel independently ($C \times k^2$ weights), and a **pointwise ($1 \times 1$) convolution** then mixes information across channels ($C_{\text{in}} \times C_{\text{out}}$ weights). This factorization is why Paper 2's `MDNet-T` achieves a massive receptive field ($\text{RF} = 1136$) in just $4.9\text{M}$ parameters (Paper 2 Table I, III).
7. **GELU (Gaussian Error Linear Unit) vs. ReLU:**  
   **ReLU** ($\max(0, x)$) hard-zeros all negative values and has a sharp non-differentiable corner at $x = 0$, which can cause "dead neurons." **GELU** (Paper 2 Sec. IV-C3, Eq. 11–12) weights the input by the standard Gaussian cumulative distribution function, $\text{GELU}(x) = x \Phi(x) = x P(X \le x)$, creating a smooth, everywhere-differentiable curve that allows small negative activations and improves MDNet classification accuracy by $+2.4\%$ over ReLU (Paper 2 Table VI).
8. **Batch Normalization vs. Layer Normalization:**  
   **BatchNorm** normalizes activations across the mini-batch dimension $(B, H, W)$ for each channel, which can suffer from "batch confusion" when small batches contain a mix of clear and heavily polluted scenes from different cities. **LayerNorm** (Paper 2 Sec. III-A) normalizes across the channel dimension $C$ independently for each individual sample and spatial position ($\mu, \sigma$ computed per token/pixel), making behavior identical between training and single-image mobile inference.
9. **Self-Attention (Vision Transformer):**  
   **Self-Attention** (Paper 1 Sec. II-B; Paper 2 Eq. 10) projects a sequence of $M$ image patch tokens into Queries ($Q$), Keys ($K$), and Values ($V$), and computes an all-pairs similarity matrix $\text{Softmax}(Q K^T / \sqrt{d}) V$. This allows every patch in the image (e.g., upper sky) to directly attend to and compare contrast/color with every other patch (e.g., distant horizon or foreground buildings) in a single step, regardless of spatial distance.
10. **Prototype Learning & Metric Space Clustering:**  
    Rather than learning a flat linear hyperplane $W \chi_i + b$ to separate classes—which gets skewed toward majority classes in long-tailed datasets—**prototype learning** (Paper 1 Sec. II-C, Eq. 1–3) represents each pollution class $k$ by a centroid vector ("prototype") $c'_k \in \mathbb{R}^D$ in feature space and classifies samples by proximity $\|\chi_i - c'_k\|_2^2$. Minimizing cross-entropy over negative Euclidean distances acts as a clustering force that pulls rare "Hazardous" samples tightly toward their class prototype $c'_6$, preventing them from being swallowed by "Good/Moderate" classes (Paper 1 Fig. 7).
11. **GroupKFold Cross-Validation (Station-Grouped Splitting):**  
    Standard $K$-fold cross-validation randomly shuffles rows into train and validation folds, which causes severe **spatial data leakage** in environmental datasets because multiple photos from the same monitoring station share identical background buildings, camera angles, and local climate (Paper 1 Sec. III-A3). **`GroupKFold` grouped by `station_id`** guarantees that all images associated with a given `station_id` are placed entirely in either the training fold or the validation fold—never both—measuring true generalization to unseen monitoring stations.
12. **Modality Dropout (Preventing Shortcut Learning):**  
    In multimodal networks where one input branch (such as GPS + Month tabular metadata) provides an easy statistical shortcut to the target, gradient descent rapidly latches onto the tabular branch and starves the harder vision backbone of informative gradients. **Modality Dropout** randomly zeroes out the entire metadata embedding vector with probability $p_{\text{drop}}$ during training, forcing the network to rely purely on visual haze cues whenever metadata is dropped while still leveraging metadata when available.
13. **PLCC, SRCC, MAE, RMSE, and $R^2$:**  
    **MAE** ($\frac{1}{N}\sum |\hat{y}_i - y_i|$) and **RMSE** ($\sqrt{\frac{1}{N}\sum (\hat{y}_i - y_i)^2}$, Paper 1 Sec. III-A2) measure average prediction error in AQI units, with RMSE penalizing large tail errors more heavily. **PLCC** (Pearson $r$, Paper 1 Sec. III-A2) measures linear correlation $\text{Cov}(\hat{y}, y)/(\sigma_{\hat{y}}\sigma_y) \in [-1, 1]$, **SRCC** (Spearman $\rho$, Paper 1 Sec. III-A2) measures rank-order monotonicity (whether higher predicted AQI reliably corresponds to higher true AQI), and **$R^2 = 1 - \sum (y_i - \hat{y}_i)^2 / \sum (y_i - \bar{y})^2$** measures the fraction of total AQI variance explained by the model above a constant mean baseline.

---

## 4. The Experiment Ladder (Report Narrative & Hypotheses)

To make the ACIT4620 report read like a rigorous scientific investigation, we structure our experiments as a **7-step ladder**. Every rung changes **one clearly defined architectural or training variable**, uses the **exact same 5-fold `GroupKFold` station-grouped split** on cleaned training data plus final evaluation on the held-out test set, and reports both **overall metrics** (`MAE`, `RMSE`, `PLCC`, `SRCC`, `$R^2$`, `6-class Top-1 Acc`, `Macro-F1`) and **tail-class metrics** (`MAE` and `F1` on EPA Bins 5 & 6: `AQI > 200`).

| Exp # | Experiment Name | Input Modalities | Architecture & Loss | Paper Citation | Scientific Question / What This Proves in the Report |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **E0** | **Trivial Statistical & Physics Filter Baselines** | None / 9 Handcrafted Sky-Ground Filters | Median/Mean predictor; Ridge Regression on HSV Saturation, Sobel Gradient Contrast, Dark Channel Prior (top vs. bottom bands). | Paper 1 Sec. I & III-D (Table VIII); Paper 2 Sec. II | Establishes the zero-skill floor (predicting train median AQI) and proves that simple physical scattering cues (sky saturation & Sobel gradient contrast, Paper 1 Table VIII) carry real but limited linear signal. |
| **E1** | **Metadata-Only Shortcut Baseline** | Tabular Only: Spherical GPS $(x,y,z)$ + Cyclic Month $(\sin, \cos)$ | `MetadataMLP` ($5 \to 64 \to 32 \to 1$), Smooth L1 / MSE loss. | User Constraint; Paper 1 Sec. III-A3 (spatial leakage context) | Quantifies **how much of PM25Vision's test performance can be explained purely by location + season without looking at a single pixel**—and why our Data Audit (Sec. 5) shows $63\%$ of test stations lie within $10\text{ km}$ of a training station. |
| **E2** | **Image-Only Off-the-Shelf CNN Baseline** | RGB Image Only ($224 \times 224$) | Standard `ResNet-18` or `MobileNetV3-Small` backbone + single regression head ($L_{\text{reg}}$ only). | Paper 1 Table V (`MobileNetV3` / CNN ablation); Paper 2 Table III | Establishes the standard single-scale, single-task deep vision baseline on unseen stations before introducing Paper 2's dilated kernels or Paper 1's hybrid multitask framework. |
| **E3** | **Paper 2 Core: Lightweight Multiscale Dilated CNN (`MDNet-T`)** | RGB Image Only ($224 \times 224$) | `MDNet-T` backbone ($[3,3,9,3]$, 3-scale dilated MD blocks $k'=5,3,1$ fused by addition, `GELU`, `LayerNorm`, $\text{RF}=1136$). Tested in (a) 6-class classification ($L_{\text{lsce}}$, Paper 2 Eq. 2) and (b) single-task regression ($L_{\text{reg}}$). | **Paper 2** Sec. III-A, III-B, Fig. 1, Table I, III, VI | Proves **Paper 2's core claim**: multiscale dilated convolutions expand the receptive field across sky and building reflections, beating standard small-kernel CNNs (`E2`) at low parameter count ($4.9\text{M}$) and Core ML-friendly complexity. |
| **E4** | **Paper 1 + Paper 2 Backbone: `MDNet-T` + DCM + Lightweight Transformer** | RGB Image Only ($224 \times 224$) | `MDNet-T` stages $C_1\dots C_4 \to$ **Paper 1 DCM** ($[B, 224, 14, 14]$) $\to$ **2-layer ViT Encoder** ($\text{Dim}=224$) $\to$ single-task regression head ($L_{\text{reg}}$). | **Paper 1** Sec. II-B, Fig. 3, Table I, V, VI | Proves **Paper 1's first core claim**: unifying multi-stage local CNN features through the Dimension Consistency Module (DCM) and modeling global contextual relationships with Transformer self-attention improves accuracy over CNN-only (`E3`). |
| **E5** | **Full Image-Only Proposed Model: Prototype-Guided Multitask Learning (`MDNet-PFRML`)** | RGB Image Only ($224 \times 224$) | `E4` backbone + **Paper 1 Prototype Classification Branch** ($K=6$ K-means/centroid prototypes, Eq. 1–3) + **Prototype-Guided Regression MLP** + Hybrid Loss $L = L_{\text{lsce}} + \beta L_{\text{reg}}$ ($\beta=0.5$). | **Paper 1** Sec. II-C, II-D, III-C3–5, Eq. (1)–(6), Fig. 4–9, Table VII–VIII | Proves **Paper 1's second core claim**: prototype-guided multitask learning restructures the latent space (verified via t-SNE, Paper 1 Fig. 7), significantly improves **tail-class (`AQI > 200`) $F_1$ and RMSE** (Paper 1 Table VII), and produces prototypes whose assigned images monotonically degrade in Sobel contrast and HSV saturation (Paper 1 Table VIII). |
| **E6** | **Multimodal Fusion & Shortcut Mitigation (`MDNet-PFRML` + Metadata)** | RGB Image ($224\times 224$) + Metadata (GPS + Month) | Compare **E6a (Naive Concat Fusion)** vs. **E6b (Modality Dropout $p_{\text{drop}}=0.4$ + Detached Vision Auxiliary Loss)**. Evaluate on: (1) Full Test Set, (2) **Metadata-Masked Test Set** (zeroing GPS+Month at test time), (3) **Permuted-Image Test Set** (shuffling images across rows to measure vision contribution $\Delta \text{MAE}$), and (4) **Spatially Isolated Test Subset** (test stations $>25\text{ km}$ from any train station). | User Constraint; Paper 1 Sec. III-B2 / III-E | Directly measures and solves the **metadata shortcut problem**: proves that Naive Fusion (`E6a`) collapses when metadata is masked or shifted to distant regions, whereas Modality Dropout + Auxiliary Vision Loss (`E6b`) preserves strong image sensitivity ($\Delta \text{MAE}$ under image permutation) while achieving the best overall accuracy when both modalities are available. |

---

## 5. Data Audit: Verifying `AGENT_HANDOFF.md` Against the Real Repository & Dataset

We inspected the actual repository files and ran a full audit on `datasets/train/metadata.csv`, `datasets/test/metadata.csv`, `datasets/train/images/`, and `datasets/test/images/`.

### 5.1 Summary of Verified Dataset Numbers

* **Raw Rows & Image Files on Disk:**
  * `datasets/train/metadata.csv`: **8,298 rows**, **8,194 unique `image_id`s** (`filename`s), and **8,194 `.jpg` files** in `datasets/train/images/` (**0 missing images**, **0 extra files**).
  * `datasets/test/metadata.csv`: **2,921 rows**, **2,920 unique `image_id`s** (`filename`s), and **2,920 `.jpg` files** in `datasets/test/images/` (**0 missing images**, **0 extra files**).
  * Combined raw total: **11,219 metadata rows** and **11,114 image files** on disk.
* **Duplicate `image_id`s & Train/Test Image Overlap:**
  * Within `train`: **165 rows** share an `image_id` across **61 unique `image_id`s** ($165 - 61 = 104$ extra rows).
  * Within `test`: **2 rows** share an `image_id` across **1 unique `image_id`** ($2 - 1 = 1$ extra row).
  * Across `train` and `test`: **18 unique `image_id`s** appear in **both** `train/metadata.csv` and `test/metadata.csv`!
  * Combining all duplicates across `train + test`, there are **80 unique `image_id`s** (203 total rows) that appear more than once because the same Mapillary street photo was matched to $2$ or $3$ nearby monitoring stations (in 71 of those 80 cases, the nearby stations reported slightly different AQI values, with a median difference of $9.0$ AQI units and max difference of $93.0$).
  * After deduplicating per `image_id` and purging the 18 overlapping `image_id`s from `train`, the clean dataset contains **11,096 unique images**: **8,176 train** and **2,920 test**.
* **Station Counts & Station Overlap:**
  * Raw `train` has **2,532 unique `station_id`s**; raw `test` has **729 unique `station_id`s**; total unique `station_id`s $= \mathbf{3,261}$.
  * **Exact `station_id` overlap between `train` and `test` is 0** ($0 / 729 = 0.00\%$). After purging the 18 overlapping images from `train`, 2 train stations that only had overlapping images drop out, leaving **2,530 clean train stations** and **729 test stations** (**3,259 unique stations**).
* **CRITICAL NEW DISCOVERY — Why Metadata Is a Massive Shortcut Even with 0% `station_id` Overlap:**
  * Although `train` and `test` share **zero `station_id`s**, we computed the Haversine distance between every test image's `(latitude, longitude)` and the nearest training image's `(latitude, longitude)` (after purging the 18 overlapping images):
    * **8.5%** of test images are within **1 km** of a training image (minimum distance: **1.2 meters**!).
    * **63.0%** of test images are within **10 km** of a training image (median distance: **6.38 km**).
    * **75.5%** of test images are within **25 km**, and **83.5%** are within **50 km**.
    * Furthermore, **23.8% (695 / 2,921)** of test rows have a training image captured on the **exact same calendar date within 25 km**!
  * **Why?** Major cities (e.g., Delhi, Beijing, Taipei, Tokyo) have many monitoring stations only a few kilometers apart. Splitting by `station_id` puts Station A in `train` and Station B ($3\text{ km}$ away in the same city) in `test`. Consequently, a metadata model fed `(GPS, month)` memorizes each city's seasonal pollution baseline and achieves low test error without ever learning visual haze!
* **Target (`pm25`) AQI Distribution & US EPA Bin Balance:**
  * The `pm25` column contains integer-valued continuous **US EPA AQI values** (all `.0`), **not** $\mu\text{g/m}^3$ concentrations:
    * **Raw Train ($N=8,298$):** $\min = 1.0$, $P_{25} = 54.0$, $\text{median} = 112.0$, $\text{mean} = 122.96$ ($\text{SD} = 88.54$), $P_{75} = 165.0$, $P_{95} = 288.0$, $\max = 530.0$.
    * **Raw Test ($N=2,921$):** $\min = 2.0$, $P_{25} = 54.0$, $\text{median} = 114.0$, $\text{mean} = 119.33$ ($\text{SD} = 81.47$), $P_{75} = 165.0$, $P_{95} = 246.0$, $\max = 416.0$.
  * **Long-Tailed Class Imbalance Across the 6 EPA Bins (`pm25_bin`):**
    * The dataset creators capped/balanced the first 4 bins (`0–50`, `51–100`, `101–150`, `151–200`) at exactly **1,780 rows each in raw train** and **630 rows each in raw test**, while the two highest pollution bins are severely under-represented (**long-tailed**):

| EPA Level | `pm25_bin` (AQI Range) | Standard EPA Category Name | Raw Train Rows (%) | Clean Train Images (%) | Raw Test Rows (%) | Clean Test Images (%) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | `0–50` | Good | 1,780 (21.45%) | 1,778 (21.75%) | 630 (21.57%) | 629 (21.54%) |
| **2** | `51–100` | Moderate | 1,780 (21.45%) | 1,776 (21.72%) | 630 (21.57%) | 630 (21.58%) |
| **3** | `101–150` | Unhealthy for Sensitive Groups | 1,780 (21.45%) | 1,760 (21.53%) | 630 (21.57%) | 630 (21.58%) |
| **4** | `151–200` | Unhealthy | 1,780 (21.45%) | 1,776 (21.72%) | 630 (21.57%) | 630 (21.58%) |
| **5 (Tail)** | `201–300` | Very Unhealthy | 822 (9.91%) | 731 (8.94%) | 275 (9.41%) | 275 (9.42%) |
| **6 (Tail)** | `301–600` | Hazardous | 356 (4.29%) | 355 (4.34%) | 126 (4.31%) | 126 (4.32%) |
| **Total** | **1–530** | **All 6 Levels** | **8,298 (100%)** | **8,176 (100%)** | **2,921 (100%)** | **2,920 (100%)** |

  * Notice that Bins 5 and 6 (`201–300` and `301–600`) together make up only **13.3% of clean train images** ($4.34\%$ Hazardous), directly motivating **Paper 1's Prototype Feature Representation-Guided Multitask Learning** to protect the tail classes!
* **Column Nulls & Quirks:**
  * `camera_angle` and `quality_score` are **100% null (`NaN`)** across all 8,298 train rows and 2,921 test rows.
  * The column `quality` has the constant string `"good"` for 100% of rows (it denotes Mapillary image quality filtering, **not** EPA air quality—use `pm25_bin` for EPA air quality classes!).
  * Image resolutions on disk are predominantly **$1024 \times 768$** (4:3 aspect ratio, $\sim 55\%$) and **$1024 \times 576$** (16:9 aspect ratio, $\sim 40\%$).

### 5.2 Audit of Claims in `AGENT_HANDOFF.md`: What Holds vs. What Does NOT Hold

| Claim in `AGENT_HANDOFF.md` | Audit Verdict | Actual Finding on Disk / in Data |
| :--- | :---: | :--- |
| `datasets/train/` has 8,298 metadata rows and 8,194 images; `datasets/test/` has 2,921 metadata rows and 2,920 images (Sec. 2). | ✅ **Holds** | Confirmed exact counts: 8,298 / 8,194 in `train` and 2,921 / 2,920 in `test`, with 0 missing image files. |
| 80 images are associated with multiple stations, 18 images overlap between train and test, yielding 11,096 deduplicated images (8,176 train, 2,920 test) (Sec. 3). | ⚠️ **Partially Holds (Station Count Off by 2)** | The image counts (80 duplicate `image_id`s across 203 rows; 18 train/test overlap `image_id`s; 8,176 clean train + 2,920 clean test = 11,096) are exact. However, `3,261` is the **raw** station count; after purging the 18 overlapping images from train, **3,259** unique stations remain (2,530 train + 729 test). Also, only 71 of the 80 duplicate images have conflicting AQI values (9 have identical AQI across stations). |
| `features/` directory (`clean_metadata.csv`, `handcrafted_filter_features.csv`, `efficientnet_b0_embeddings.npz`, etc.), `1.eda-image-datasets.ipynb`, `2.feature-extraction.ipynb`, `3.example_train_test.ipynb`, `best_model.pt`, and `paper_taxonomy.xlsx` are present on disk (Sec. 2 & Sec. 7). | ❌ **Does NOT Hold** | **None of these files exist on the current branch (`mughil/trial-2`).** `features/`, `best_model.pt`, `paper_taxonomy.xlsx`, and the three notebooks were deleted or belong to an earlier branch. Our pipeline (`src/dataset.py`) must perform deduplication and leak purging directly from `datasets/*/metadata.csv` rather than relying on a pre-existing `features/clean_metadata.csv`. |
| Test stations are "unseen test stations" that evaluate spatial generalization (Sec. 3, Sec. 6). | ⚠️ **Misleading Without Qualification** | While `station_id` overlap is indeed $0\%$, **63.0% of test images are within $10\text{ km}$ of a training station** (and $8.5\%$ within $1\text{ km}$) inside the same cities. Thus, for a **metadata-using model**, the official test split is *not* geographically unseen (unlike Paper 1's strict 6-train / 2-unseen-city split on India–Nepal, Sec. III-A3). |
| Recommended architecture in Sec. 6: `EfficientNet-B0 / ResSE / Swin-T` + `HazeFilters` 8-channel input + simple `Smooth L1 + Cross-Entropy` heads. | ❌ **Rejected (Does Not Match Assigned Papers)** | `AGENT_HANDOFF.md`'s architecture comes from a different paper (Hasan & Shakir 2026 `ResSE`) and generic `EfficientNet-B0`. It completely omits **Paper 2's Multiscale Dilated CNN (`MDNet`)**, **Paper 1's Dimension Consistency Module (DCM)**, and **Paper 1's Prototype Feature Representation (`PFRML`)**. We replace it with our paper-traceable architecture in Section 2. |

---

## 6. Proposed Repository Layout & Build Order

### 6.1 Modular Repository Structure

```text
acit-4620-projects/
├── configs/
│   ├── default.yaml               # Default hyperparameters, paths, seed, CV folds
│   ├── e1_metadata_only.yaml      # Rung 1: Metadata MLP baseline
│   ├── e2_cnn_baseline.yaml       # Rung 2: Standard CNN image-only baseline
│   ├── e3_mdnet.yaml              # Rung 3: Paper 2 MDNet-T (dilated multiscale CNN)
│   ├── e4_mdnet_dcm_vit.yaml      # Rung 4: Paper 2 MDNet-T + Paper 1 DCM & Transformer
│   ├── e5_mdnet_pfrml.yaml        # Rung 5: Full image-only MDNet + DCM + ViT + Prototypes
│   └── e6_multimodal_fusion.yaml  # Rung 6: Multimodal fusion with Modality Dropout
├── docs/
│   └── PLAN.md                    # This architectural & experimental blueprint
├── src/
│   ├── __init__.py
│   ├── config.py                  # Dataclass/YAML config loader with CLI overrides
│   ├── data/
│   │   ├── __init__.py
│   │   ├── cleaning.py            # Deterministic deduplication (median AQI) & 18-image leak purge
│   │   ├── dataset.py             # Lazy-loading PyTorch Dataset yielding (image, meta, aqi, bin, station_id)
│   │   ├── splits.py              # Station-grouped GroupKFold / StratifiedGroupKFold + >25km spatial split
│   │   └── transforms.py          # Paper 2 Sec. III-C augmentations (NO color jitter / grayscale)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baselines.py           # MetadataMLP, Ridge/Filter baseline, ResNet18/MobileNetV3 wrapper
│   │   ├── mdnet.py               # Paper 2: MDBlock (k=3 d=2, k=2 d=2, k=1 d=1, GELU, LN) & MDNet-T/MDNet
│   │   ├── dcm_transformer.py     # Paper 1 Sec. II-B: Dimension Consistency Module (DCM) + lightweight ViT
│   │   ├── prototype_head.py      # Paper 1 Sec. II-C/D: K=6 Prototype layer, K-means/EMA update, MLP head
│   │   └── pfrml_net.py           # Unified MDNet-PFRML model + optional Modality-Dropout Metadata Fusion
│   ├── losses.py                  # Paper 1 Eq. 4–6 & Paper 2 Eq. 2: Label Smoothing CE + Huber/MSE hybrid loss
│   ├── metrics.py                 # MAE, RMSE, PLCC, SRCC, R2, 6-class Acc, Macro-F1, Tail-class (Bins 5–6) F1/MAE
│   ├── interpretability.py        # Paper 1 Table VIII (Sobel contrast & HSV saturation per prototype), t-SNE, Grad-CAM/Attention maps
│   └── export_coreml.py           # Core ML (.mlpackage) export & numerical parity verification script
├── train.py                       # CLI training & 5-fold station-grouped CV runner
├── evaluate.py                    # CLI evaluation on held-out test set + shortcut stress tests (masked meta, permuted image, >25km stations)
└── outputs/                       # Git-ignored directory for checkpoints, CSV logs, and report-ready figures
```

### 6.2 Step-by-Step Build Order (After Plan Approval)

1. **Step 1 — Data Pipeline & Verification (`src/data/`):**  
   Implement `cleaning.py` (automatically deduplicating the 80 multi-station images via median AQI and removing the 18 train/test overlap images from `train` in memory or cached CSV), `splits.py` (5-fold `StratifiedGroupKFold` by `station_id` and `pm25_bin`), `transforms.py` (strictly excluding color jitter/grayscale per Paper 2 Sec. III-C), and `dataset.py` (lazy PIL image loading on the fly).
2. **Step 2 — Metrics, Losses & Baselines (`src/metrics.py`, `src/losses.py`, `src/models/baselines.py`):**  
   Implement all evaluation metrics (`PLCC`, `SRCC`, `RMSE`, `MAE`, `$R^2$`, overall & tail-class `F1`) and run **E0** (median + physics filter Ridge baseline) and **E1** (`MetadataMLP`) to lock in the reference table within minutes.
3. **Step 3 — Paper 2 Backbone (`src/models/mdnet.py`) & CLI (`train.py`, `evaluate.py`):**  
   Build the `MDBlock` and `MDNet-T` ($4.9\text{M}$ params), verify its parameter count ($4.88\text{M}$), receptive field ($\text{RF}=1136$), and MPS forward/backward speed, and run **E2** and **E3**.
4. **Step 4 — Paper 1 DCM, Transformer & Prototype Multitask Framework (`src/models/dcm_transformer.py`, `prototype_head.py`, `pfrml_net.py`):**  
   Add the Dimension Consistency Module (DCM) and 2-layer Transformer encoder (**E4**), followed by the $K=6$ Prototype classification branch and prototype-guided regression MLP (**E5**).
5. **Step 5 — Multimodal Fusion & Shortcut Stress-Testing (`E6`):**  
   Add the `MetadataMLP` fusion branch with **Modality Dropout** and run the shortcut audit suite in `evaluate.py` (Image-Only vs. Metadata-Only vs. Naive Fusion vs. Modality-Dropout Fusion under normal test, masked-metadata test, permuted-image test, and $>25\text{ km}$ spatially isolated test stations).
6. **Step 6 — Physical Interpretability & Core ML Export (`src/interpretability.py`, `src/export_coreml.py`):**  
   Generate report figures reproducing **Paper 1 Fig. 7** (t-SNE of head vs. tail clusters), **Paper 1 Fig. 10 & Paper 2 Fig. 7** (spatial attention / Grad-CAM heatmaps on sky vs. buildings), **Paper 1 Table VIII** (mean $\pm$ SD of Sobel gradient contrast and HSV saturation across learned prototypes 1–6), and verify export of the image-only/multimodal model to Core ML (`.mlpackage`).

---

## 7. Risks, Core ML Considerations & Open Questions

### 7.1 Technical Risks & Mitigations

1. **Risk 1 — Scale Mismatch in Paper 1's Hybrid Loss ($L = L_{\text{cro}} + \beta L_{\text{mse}}$):**  
   *Issue:* In Paper 1 Eq. (6), $\beta = 0.5$. However, raw AQI values span $1\text{--}530$, so raw MSE is $\sim 2,000\text{--}5,000$ while cross-entropy $L_{\text{cro}}$ is $\sim 1.0\text{--}1.8$. Without scaling the regression target, $\beta L_{\text{mse}}$ overwhelms $L_{\text{cro}}$ by $1000\times$, contradicting Paper 1 Sec. III-C5's statement that *"the classification loss accounts for a larger proportion of the total loss"* at $\beta = 0.5$.  
   *Mitigation:* Normalize the regression target by dividing AQI by $100.0$ (so AQI $\in [0.01, 5.30]$ and squared/Huber errors are $\mathcal{O}(1)$, matching $L_{\text{cro}}$'s scale at $\beta = 0.5$), and multiply predictions by $100.0$ when computing AQI metrics.
2. **Risk 2 — K-Means Prototype Instability & Unnormalized Euclidean Distances (Paper 1 Eq. 1–3):**  
   *Issue:* If feature vectors $\chi_i \in \mathbb{R}^{224}$ have arbitrary norm, squared Euclidean distance $\|\chi_i - c'_k\|_2^2$ can grow to $50\text{--}500$, causing $\exp(-d(\chi_i, c'_k))$ in Paper 1 Eq. (3) to underflow to $0$ in float32. Furthermore, running full-dataset K-means from scratch at every step would stall training.  
   *Mitigation:* Apply $\ell_2$-normalization (or LayerNorm + learnable temperature scale $\tau$) to $\chi_i$ and $c'_k$ before computing $\|\chi_i - c'_k\|_2^2$, initialize $c'_k$ from the 6 ground-truth EPA class centroids (or K-means) after epoch 0, and update $c'_k$ via class-centroid exponential moving average (EMA) during training and exact class/K-means centroids at epoch boundaries.
3. **Risk 3 — Training Time on MacBook Pro M5 Pro (24 GB MPS):**  
   *Issue:* Running 5-fold cross-validation across 6 deep learning configurations on 8,176 images ($1024 \times 768$ JPEGs on disk) could take many hours if CPU JPEG decoding bottlenecks the M5 Pro GPU or if every ablation runs 5 folds $\times$ 50 epochs.  
   *Mitigation:* Use `DataLoader(..., num_workers=4, pin_memory=False, persistent_workers=True)` on macOS MPS, `torch.amp.autocast('mps', dtype=torch.float16/bfloat16)` where supported, and a 20–30 epoch Cosine Annealing schedule. (We can also run intermediate ablations on Fold 0 first, and run full 5-fold CV on the main ladder rungs).
4. **Risk 4 — Core ML Conversion Traps for iPhone Demo:**  
   * **Trap A (`AdaptiveMaxPool2d` / `AdaptiveAvgPool2d` in Paper 1 DCM):** Dynamic adaptive pooling to $14 \times 14$ often fails or falls back to CPU in `coremltools`. *Fix:* Use static `nn.MaxPool2d(kernel_size=4, stride=4)` and `nn.MaxPool2d(kernel_size=2, stride=2)` on fixed $224 \times 224$ inputs.
   * **Trap B (`Conv2d` with `kernel_size=2, dilation=2, padding=1` in Paper 2 MD Block):** While valid in PyTorch (`k'=3`), some Apple Neural Engine (ANE) compiler passes dislike even kernel sizes with dilation $>1$. *Fix:* In `export_coreml.py`, provide a 1-line weight expansion that embeds the $2 \times 2$ dilated-$2$ weights into an equivalent $3 \times 3$ dilation-$1$ kernel (with zeros on the middle row/column) so the exported Core ML graph uses standard $3 \times 3$ and $5 \times 5$ (or $3 \times 3, d=2$) kernels with 100% mathematical equivalence.
   * **Trap C (K-Means Clustering Ops):** K-means is only used to update prototypes during training; during inference, prototypes $C' \in \mathbb{R}^{6 \times 224}$ are a static tensor buffer.

---

### 7.2 Open Questions for Your Approval (Max 5)

Before we write any code, please let us know your preferences on these 5 design choices:

1. **Cross-Validation Strategy for the Experiment Ladder:**  
   Running 5-fold `StratifiedGroupKFold` (by `station_id`) for *every* rung (`E2`–`E6`) means training $\sim 25\text{--}30$ deep networks. Would you like us to:
   * **(Recommended)** Run **all 5 folds** for the key milestone models (`E1 Metadata`, `E3 MDNet-T`, `E5 MDNet-PFRML`, `E6b Modality-Dropout Fusion`) and use **Fold 0 + held-out Test Set** for intermediate component ablations (`E2`, `E4`, `E6a`, $\beta$ sensitivity), **or** run full 5-fold CV for every single experiment?
2. **Pretrained ImageNet Weights vs. Training `MDNet-T` from Scratch:**  
   Because Paper 2's `MDNet-T` uses custom 3-branch dilated `MDBlock` layers ($k=3,d=2$; $k=2,d=2$; $k=1,d=1$), there are no off-the-shelf ImageNet weights for `MDNet-T` (it must be trained from scratch on PM25Vision, or initialized by mapping ConvNeXt-Tiny stem/downsample weights). For `E2` (the standard CNN baseline), should we also train from scratch for a fair apple-to-apples comparison with `MDNet-T`, or compare against ImageNet-pretrained `MobileNetV3` / `ResNet-18` (or both)?
3. **Resolving Paper 1's Two Internal Ambiguities (DCM & Prototype Update):**  
   * In Paper 1 Sec. II-B, the text says the four DCM stage outputs are *"concatenated"*, while **Table I (`DCM Stage 5`)** explicitly specifies **`Element-wise Summation`** to `[B, 224, 14, 14]`.  
   * In Paper 1 Sec. II-C, prototypes are defined via K-means (Eq. 2) against 6 pollution classes.  
   Do you approve of following **Table I (`Element-wise Summation`)** so the Transformer dimension stays compact at $\text{Dim}=224$, and initializing the 6 prototypes from the 6 EPA class centroids (with K-means refinement / EMA updates per epoch)?
4. **Regression Loss Function (`MSE` vs. `Smooth L1 / Huber`):**  
   Paper 1 Eq. (5) uses MSE ($L_{\text{mse}}$), whereas `AGENT_HANDOFF.md` recommended Smooth L1 / Huber loss because AQI has extreme outliers up to $530$. Should we use **scaled MSE** as our primary loss (strictly following Paper 1 Eq. 5) and include **Huber / Smooth L1** as an ablation, or make Huber loss the default?
5. **Handling Existing Deleted/Untracked Files in `git status`:**  
   On your current branch (`mughil/trial-2`), `git status` shows the old notebooks (`1..3.ipynb`, `README.md`, `paper_taxonomy.xlsx`) staged as deleted (`D`) and helper files (`AGENT_HANDOFF.md`, `PROJECT_EXPLAINER.md`, `convert_papers_to_markdown.py`, `papers.zip`, `test.ipynb`) as untracked. When we begin the implementation phase, should we leave those existing files untouched as-is, or clean up `test.ipynb` / organize the root directory?
