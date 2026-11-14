# Image-Based Air Quality Estimation (ACIT 4620)
## Project Guide, Learning Roadmap & Research Next Steps

> **Welcome!** If you are new to Artificial Intelligence (AI) and Neural Networks (NN), this document was written specifically for you. It breaks down what your teammate built, translates the AI and machine learning jargon into intuitive concepts, outlines what you need to learn step-by-step, and highlights the best next steps based on the latest 2024–2026 research papers.

---

## Table of Contents
1. [Executive Summary: What is this Project Doing?](#1-executive-summary-what-is-this-project-doing)
2. [Deep-Dive into the Codebase (The 3 Notebooks)](#2-deep-dive-into-the-codebase-the-3-notebooks)
   - [Notebook 1: Exploratory Data Analysis & Feature Discovery (`1.eda-image-datasets.ipynb`)](#notebook-1-exploratory-data-analysis--feature-discovery-1eda-image-datasetsipynb)
   - [Notebook 2: Streamlined Feature Extraction & Tabular Models (`2.feature-extraction.ipynb`)](#notebook-2-streamlined-feature-extraction--tabular-models-2feature-extractionipynb)
   - [Notebook 3: End-to-End Deep Learning with PyTorch (`3.example_train_test.ipynb`)](#notebook-3-end-to-end-deep-learning-with-pytorch-3example_train_testipynb)
3. [Jargon Buster: Core AI & Vision Concepts Explained Simply](#3-jargon-buster-core-ai--vision-concepts-explained-simply)
4. [Step-by-Step Learning Roadmap for Beginners](#4-step-by-step-learning-roadmap-for-beginners)
5. [Key Lessons from the Research Literature (`papers/` & `paper_taxonomy.xlsx`)](#5-key-lessons-from-the-research-literature-papers--paper_taxonomyxlsx)
6. [Actionable Next Steps to Make the Project Successful](#6-actionable-next-steps-to-make-the-project-successful)
7. [How to Divide and Conquer the Work with Your Teammate](#7-how-to-divide-and-conquer-the-work-with-your-teammate)

---

## 1. Executive Summary: What is this Project Doing?

### The Core Problem
Air pollution—particularly **fine particulate matter (PM2.5)**—is a major global health hazard. Traditional government-grade air quality monitoring stations are expensive (\$10,000+ each) and scarce. 

**The Project Goal:** Can we estimate air quality (AQI / PM2.5) simply by analyzing a photograph taken from a phone, dashcam, or street camera?

```
[Street Image] ───► [Computer Vision / Neural Network] ───► [Predicted AQI Value]
```

### The Physical Intuition (Why This is Possible)
When light travels through polluted air:
1. **Light Scattering (Haze):** Particles scatter sunlight into the camera sensor, making dark objects look washed out or hazy (**Dark Channel Prior**).
2. **Color Loss (Attenuation):** Atmospheric particles absorb and scatter specific wavelengths; skies shift from crisp blue to milky grey or yellowish haze (**Blue Ratio & Saturation loss**).
3. **Contrast & Edge Blur:** Distant buildings or mountain horizons lose sharp outlines (**Edge & Local Contrast degradation**).

### What Your Teammate Has Built So Far
Your teammate used AI assistance to create a structured three-phase pipeline:
1. **Data Cleaning & Exploration (`1.eda-image-datasets.ipynb`):** Examined ~11,000 street photos across the world paired with real ground-station sensor readings. Cleaned up bugs and duplicate photos, tested 94 physics-based vision filters, and established a baseline using linear regression and a pretrained neural net (EfficientNet).
2. **Feature Extraction Pipeline (`2.feature-extraction.ipynb`):** Distilled the 94 filters down to the 6 most impactful haze filters, split images into vertical bands (sky vs ground), and tested a lightweight neural network (MLP) combining image filters with location/season metadata.
3. **End-to-End Deep Learning (`3.example_train_test.ipynb`):** Built custom PyTorch Convolutional Neural Networks (CNNs) that process raw images directly on the GPU, inject haze physics layers, and implement modern architectural techniques (Squeeze-and-Excitation attention and Residual skip connections).

---

## 2. Deep-Dive into the Codebase (The 3 Notebooks)

Here is a detailed breakdown of what each file is doing under the hood.

---

### Notebook 1: Exploratory Data Analysis & Feature Discovery (`1.eda-image-datasets.ipynb`)

#### 1. The Dataset (PM25Vision)
- **Source:** Mapillary (crowdsourced street-view photos) matched with nearby **WAQI (World Air Quality Index)** air monitoring stations.
- **Critical Detail:** Although the column is named `pm25`, its numbers (0 to 500+) are actually **US EPA Air Quality Index (AQI)** values, not raw $\mu\text{g}/\text{m}^3$ concentrations:
  - `0–50`: Good (Green)
  - `51–100`: Moderate (Yellow)
  - `101–150`: Unhealthy for Sensitive Groups (Orange)
  - `151–200`: Unhealthy (Red)
  - `201–300`: Very Unhealthy (Purple)
  - `301–500+`: Hazardous (Maroon)

#### 2. Data Cleaning (Preventing "Data Leakage")
- **Multi-station deduplication:** 80 images were linked to multiple nearby stations with conflicting readings; the script kept one median value.
- **Train/Test separation:** 18 photos were present in both train and test splits; they were removed to ensure the test set evaluates truly unseen data.
- **Final count:** 8,176 training images and 2,920 testing images.

#### 3. Data Insights (EDA)
- **Seasonality & Geography:** Pollution strongly peaks in winter (due to heating and temperature inversions). Asian stations exhibit higher median AQI (~142) than Oceania (~28).
- **Camera artifacts:** Many photos are dashcam frames containing car bonnets, reflections, and timestamp text overlays.

#### 4. The 9 Physics-Based Haze Filters
The notebook implements 9 visual filters to isolate haze:
1. **Dark Channel:** In clear outdoor images, shadow areas have near-zero intensity in at least one color channel. Haze adds scattered airlight, brightening these shadows.
2. **Transmission Map:** Mathematical model estimating how much direct light survives the journey to the camera lens without being scattered.
3. **Brightness:** Peak pixel intensity.
4. **Saturation:** Color richness (haze desaturates colors toward dull grey).
5. **Color Attenuation (Brightness minus Saturation):** Differences that highlight atmospheric depth.
6. **Blue Ratio:** Proportion of blue light (Rayleigh scattering causes blue light dispersion).
7. **Sobel & Laplacian Filters:** Edge detectors (haze obscures fine edges).
8. **Local Contrast:** Variation of pixel brightness in local patches.

#### 5. Band Slicing & Handcrafted Feature Bank
- Each image is split into 3 vertical zones: **Sky (top third)**, **Middle**, and **Ground (bottom third)**.
- **Key finding:** Haze filters correlate significantly stronger with AQI in the **sky region** ($|\rho| \approx 0.28$) than in the ground region ($|\rho| \approx 0.12$). This makes intuitive sense: the sky provides a long, unbroken air column where light scattering is purest.
- 94 summary statistics (means, 10th percentiles, 90th percentiles) were extracted per photo.

#### 6. Linear Baselines (Ridge Regression on Test Stations)
| Model / Input | Mean Absolute Error (MAE) | $R^2$ (Fit Quality) | Spearman $\rho$ (Rank Corr) |
|---|---|---|---|
| Baseline (predict median AQI, no image) | 63.6 | -0.01 | — |
| 12 Selected Handcrafted Filters | 58.3 | 0.04 | 0.41 |
| All 94 Handcrafted Filters | 53.2 | 0.18 | 0.53 |
| Pretrained Deep CNN (EfficientNet-B0) | 49.0 | 0.30 | 0.63 |
| **EfficientNet-B0 + 12 Selected Filters** | **48.6** | **0.31** | **0.64** |

*Takeaway:* Pretrained deep neural networks capture rich visual patterns that outperform raw manual filters, but combining deep features with physics-based haze filters achieves the best overall performance.

---

### Notebook 2: Streamlined Feature Extraction & Tabular Models (`2.feature-extraction.ipynb`)

Notebook 2 takes the findings from Notebook 1 and streamlines them into a fast, reusable workflow:

1. **Compact Filter Pipeline:** Reduces the code to 6 core filters (`dark_channel`, `saturation`, `brightness`, `edges`, `local_contrast`, `blue_ratio`) applied across 3 vertical image bands (top, middle, bottom).
2. **Statistical Correlation Matrix:** Heatmaps illustrate which filter/band combinations have the strongest predictive power for AQI.
3. **Greedy Forward Selection:** Sequentially adds features one by one using 5-Fold `GroupKFold` cross-validation (grouped by `station_id` to prevent leakage).
4. **Metadata Engineering:**
   - **Cyclic Month Encoding:** Converts month (1–12) into $\sin$ and $\cos$ waves:
     $$\text{month\_sin} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$
     This preserves the circular relationship that December (month 12) is adjacent to January (month 1).
   - **3D Spherical Coordinates:** Converts GPS latitude and longitude into 3D Cartesian coordinates $(x, y, z)$ on a sphere, eliminating the artificial boundary jump between $+180^\circ$ and $-180^\circ$.
5. **Multi-Layer Perceptron (MLP):** A simple neural network (64 and 32 hidden units) trained on tabular combinations of image features + metadata.

---

### Notebook 3: End-to-End Deep Learning with PyTorch (`3.example_train_test.ipynb`)

Notebook 3 transitions from pre-calculated numbers (tabular features) to training neural networks directly on **raw image pixels** using PyTorch:

```
[Raw RGB Image: 3 x 144 x 192]
              │
              ▼
   [Custom HazeFilters Layer]
   (computes dark channel, saturation, brightness, edges, contrast on GPU)
              │
              ▼
   [8-Channel Feature Tensor: 8 x 144 x 192]
              │
              ▼
   [Convolutional Neural Network Backbone]
   (learned spatial pattern extraction via Conv2D + BatchNorm + Pooling)
              │
              ▼
   [Attention & Residual Connections (ResSE)]
   (Squeeze-and-Excitation dynamically boosts haze-sensitive channels)
              │
              ▼
   [Global Average Pooling & Dense Head]
              │
              ▼
   [Predicted AQI / 100]
```

#### Key Architecture Components in Notebook 3:
1. **`HazeFilters` (GPU Layer):**
   - Automatically converts standard 3-channel RGB images into an **8-channel input** (3 RGB channels + 5 calculated haze maps).
   - Runs entirely on GPU using PyTorch pooling and 2D convolutions, avoiding slow CPU image processing.
2. **Model 1: Custom Baseline CNN:**
   - 4 convolutional stages with Batch Normalization and Max Pooling.
   - Global Average Pooling (`AdaptiveAvgPool2d(1)`) to compress feature maps into a vector.
   - Fully connected layers with Dropout (0.3) to prevent overfitting.
3. **Model 2: Modern ResSE Model (Hasan & Shakir, 2026):**
   - **Residual Connections:** Skip-connections ($x + F(x)$) that preserve earlier haze signals through deep layers.
   - **Squeeze-and-Excitation (SE):** A channel-attention mechanism. It calculates an importance weight (0 to 1) for each feature channel across the whole image, allowing the model to dynamically "turn up the volume" on channels that detect haze while muting background noise.
   - **LeakyReLU activations:** Unlike standard ReLU (which zeros out negative numbers), LeakyReLU allows small negative gradients ($0.1x$), preventing neurons from "dying."
4. **Validation Strategy:**
   - Uses `StratifiedGroupKFold` across 5 folds.
   - Grouping by `station_id` ensures that all photos from any specific monitoring station are strictly isolated into either the training fold or the validation fold.

---

## 3. Jargon Buster: Core AI & Vision Concepts Explained Simply

When reading the code, you will encounter terms that seem intimidating. Here is what they mean in plain English:

| Term | Plain English Explanation | Why it Matters in this Project |
|---|---|---|
| **Tensor** | A multi-dimensional grid of numbers (like an Excel sheet, 3D cube, or 4D stack). | In PyTorch, a batch of 64 color images is a 4D tensor: `(64, 3, 144, 192)`: (batch size, color channels, height, width). |
| **Convolution (`Conv2d`)** | Sliding a tiny magnifying glass (e.g., $3 \times 3$ grid of weights) across an image to detect specific patterns like lines, corners, or haze gradients. | The core building block of Computer Vision. The model learns which patterns matter. |
| **Batch Normalization (`BatchNorm2d`)** | Re-centering and re-scaling intermediate numbers so they don't explode or shrink to zero. | Keeps deep neural network training fast and stable. |
| **Residual Connection (Skip Connection)** | An "express elevator" that adds the input of a layer directly to its output: $y = x + F(x)$. | Solves the problem where deep networks forget subtle details from earlier layers (like fine haze). |
| **Squeeze-and-Excitation (SE)** | A mini attention mechanism that asks: *"Which channels in this image are most informative right now?"* and multiplies the useful ones by higher weights. | Allows the network to focus on haze-heavy filters and ignore irrelevant background details (like cars). |
| **Loss Function (`nn.L1Loss` / MAE)** | The "grading rubric." It measures the absolute difference between the model's prediction and the real sensor reading: $\|y_{\text{true}} - y_{\text{pred}}\|$. | In this project, an L1 loss of 0.45 means the model is off by $0.45 \times 100 = 45$ AQI points on average. |
| **Optimizer (`Adam`)** | The algorithm that tweaks the model's millions of internal dials (weights) to lower the loss. | Adam is the industry-standard optimizer; it adjusts learning steps automatically. |
| **Data Leakage** | An evaluation flaw where the model accidentally sees clues from the test set during training, creating fake high accuracy. | If the same camera station appears in both training and testing, the model might just memorize the building instead of learning real haze physics! |
| **GroupKFold** | Splitting data into $K$ chunks such that groups (here, `station_id`) never appear in both train and validation. | Guarantees that our validation score accurately reflects how well the model works at **brand-new locations**. |

---

## 4. Step-by-Step Learning Roadmap for Beginners

You do not need a computer science degree to master this project. Follow this sequenced 4-week roadmap to build practical understanding:

```
[Phase 1: Python & Data Wrangling] ──► [Phase 2: Core Machine Learning]
                 │                                      │
                 ▼                                      ▼
[Phase 3: Computer Vision & Haze]   ──► [Phase 4: PyTorch Deep Learning]
```

### Phase 1: Python Data Essentials (Days 1–5)
- **Goal:** Get comfortable manipulating tables and arrays.
- **Topics:**
  - **NumPy:** Arrays, indexing, shapes (`arr.shape`), slicing (`[:, 0:64]`), and basic statistics (`mean()`, `percentile()`).
  - **Pandas:** DataFrames, reading CSVs (`read_csv`), filtering rows (`df[df['split'] == 'train']`), grouping (`groupby('station_id')`).
  - **Matplotlib:** Plotting images (`plt.imshow`), scatter plots, and histograms.
- **Recommended Free Resource:**
  - *Kaggle Learn:* "Python" and "Pandas" micro-courses (free, interactive, 4 hours each).

### Phase 2: Machine Learning Fundamentals (Days 6–10)
- **Goal:** Understand training, validation, and evaluation metrics.
- **Topics:**
  - Supervised learning: Features ($X$) vs Targets ($y$).
  - Regression metrics: **MAE** (Mean Absolute Error), **RMSE** (Root Mean Squared Error), and **$R^2$** score.
  - Overfitting vs Generalization (learning patterns vs memorizing data).
  - Train/Validation/Test splits and why `GroupKFold` prevents cheating.
- **Recommended Free Resource:**
  - *StatQuest with Josh Starmer (YouTube):* "Machine Learning Fundamentals", "Cross Validation", "Ridge Regression".

### Phase 3: Digital Image Basics & Haze Physics (Days 11–16)
- **Goal:** Understand how computers see images and how atmospheric physics affects pixels.
- **Topics:**
  - Images as 3D matrices: Height $\times$ Width $\times$ 3 Color Channels (Red, Green, Blue).
  - Normalization: Scaling pixel values from $[0, 255]$ to $[0.0, 1.0]$.
  - The Dark Channel Prior (DCP) by He et al. (2009): Why clear shadows are dark and hazy shadows are bright.
  - Image gradients and edge detection (Sobel filters).
- **Recommended Free Resource:**
  - *OpenCV Python Tutorials:* Image basics, color spaces, and filtering.
  - Read Section 4 of Notebook 1: visually inspect the generated filter maps!

### Phase 4: Neural Networks with PyTorch (Days 17–25)
- **Goal:** Understand how PyTorch models are defined, trained, and evaluated.
- **Topics:**
  - PyTorch Tensors and automatic differentiation (`loss.backward()`).
  - The standard training loop: `model.train()` $\to$ `optimizer.zero_grad()` $\to$ `loss.backward()` $\to$ `optimizer.step()`.
  - Convolutions (`Conv2d`), pooling (`MaxPool2d`), and linear layers (`Linear`).
  - DataLoaders: Batching images together for GPU processing.
- **Recommended Free Resource:**
  - *PyTorch Official Beginner Tutorials:* "Deep Learning with PyTorch: A 60 Minute Blitz".
  - *3Blue1Brown (YouTube):* "Neural Networks" series (the best visual explanation of how neural networks learn).

---

## 5. Key Lessons from the Research Literature (`papers/` & `paper_taxonomy.xlsx`)

The `papers/` folder and `paper_taxonomy.xlsx` review 16 recent IEEE papers (2024–2026). Here is what the literature teaches us—including common pitfalls and winning techniques.

### 1. The Red Flag: Beware of "Fake" 95%+ Accuracy
Several published papers claim 90%–97% accuracy on image air quality classification. However, a critical look at their methodologies reveals a major flaw:
- **The Stationary Webcam Trap:** Many papers used a single fixed camera on a rooftop, taking photos every 15 minutes. When they performed a standard random 80/20 train/test split, photos taken at 10:00 AM went into training and photos from 10:15 AM went into testing!
- The models simply memorized static background buildings or lighting conditions rather than learning true haze physics. When tested on a new city, these models completely collapsed.
- **Our Advantage:** Our dataset uses **PM25Vision** across thousands of worldwide cameras, and our code strictly uses **`GroupKFold` by station**. Our results reflect honest, real-world generalizability.

### 2. What Works in the Literature
1. **Multi-Task Learning (Classification + Regression):**
   - *Example:* *Deep Multi-Task Learning for AQI Classification and PM2.5 Estimation* (2026).
   - Training a network to predict the exact continuous AQI number **AND** classify the broad AQI category (e.g., "Good", "Moderate", "Unhealthy") simultaneously improves performance. The categorical loss helps regularize the continuous prediction.
2. **Attention & Transformers:**
   - *Example:* *SkyFuseNet* (2026) uses a Swin Transformer with Cross-Attention between ground-level photos and meteorological readings.
   - Channel attention (like the Squeeze-and-Excitation in our Notebook 3) consistently boosts performance by 5%–10% over plain CNNs.
3. **Multimodal Fusion (Images + Metadata):**
   - *Example:* *Location Generalizability of Image-Based Air Quality Models* (2025).
   - Fusing image embeddings with location embeddings (GeoCLIP/SatCLIP) and seasonal sin/cos time features significantly reduces error at unseen locations.
4. **Targeted Region Analysis (Sky & Horizon):**
   - *Example:* *Can an Image Tell the Tale* (2024).
   - Haze models perform best when focusing on regions with long lines of sight (the sky and distant horizon), while ignoring near-field clutter (road asphalt, cars, trees).

### 3. What Hurts Performance (Do Not Do This)
- **Aggressive Color Jitter Data Augmentation:** In standard vision tasks (like dog vs. cat classification), randomly shifting image hues and brightness is good practice. In air quality estimation, **color and brightness ARE the physical signal**. Randomly shifting colors destroys the subtle blue-ratio and atmospheric light cues.

---

## 6. Actionable Next Steps to Make the Project Successful

Here is a structured roadmap of concrete enhancements to elevate this project from a good baseline to an outstanding, publication-ready submission:

```
[Level 1: Quick Wins] ──► [Level 2: Multimodal Architecture] ──► [Level 3: Multi-Task & Ensembling]
```

### Level 1: Quick Wins (Can be implemented immediately)
1. **Target Transformation (Log Scaling):**
   - AQI has a heavy right tail (most values are 20–100, but some reach 400+). In Notebook 1, training on $\log(1 + \text{AQI})$ stabilized linear models.
   - In Notebook 3, try training the PyTorch model on $\log(1 + \text{AQI})$ instead of linear $\text{AQI} / 100$. This penalizes percentage errors fairly and stabilizes gradients.
2. **Smooth L1 Loss (Huber Loss):**
   - Replace `nn.L1Loss()` with `nn.SmoothL1Loss()`. Huber loss behaves like L2 (squared) loss for small errors and L1 for large errors, combining fast convergence with robustness to noisy outliers.
3. **Sky-Focused Cropping:**
   - As proven in Notebook 1, the top third of the image carries more than double the correlation signal of the bottom third. Experiment with feeding both the full image AND a dedicated crop of the top half into the model.

### Level 2: Multimodal Architecture (Fusing Image + Metadata)
Currently, Notebook 2 tests metadata with an MLP, and Notebook 3 tests images with a CNN, but they are not combined end-to-end.
- **Action:** Build a **Two-Branch Multimodal Network**:
  - **Branch A (CNN):** Processes the image into a 128-dimensional visual vector.
  - **Branch B (Dense MLP):** Processes the metadata (`month_sin`, `month_cos`, `geo_x`, `geo_y`, `geo_z`) into a 32-dimensional environmental vector.
  - **Fusion Head:** Concatenates both vectors ($128 + 32 = 160$ dimensions) and passes them to the final prediction layer.
  - *Why this works:* The metadata provides strong prior knowledge (e.g., "It is December in New Delhi, so baseline pollution is high"), while the image provides the real-time local observation (e.g., "The sky is clear today due to rain").

### Level 3: Multi-Task Learning (The "Hasan & Shakir" Strategy)
- Modify the model head in Notebook 3 to produce **two outputs**:
  1. A continuous regression value (predicted AQI).
  2. A 6-class classification probability distribution (AQI category: Good, Moderate, Unhealthy, etc.).
- Train with a combined loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{regression}} + \lambda \mathcal{L}_{\text{classification}}$$
- This forces the neural network representations to be both fine-grained and category-aware.

### Level 4: Pretrained Backbone Fine-Tuning (Transfer Learning)
- Instead of training the CNN entirely from scratch on 8,000 images, use a pretrained **EfficientNet-B0** or **ResNet-18** (which already know how to detect textures, edges, and skies from ImageNet).
- Replace the final classification layer with an AQI regression head, freeze early layers, and fine-tune the later layers with a small learning rate ($10^{-4}$).

---

## 7. How to Divide and Conquer the Work with Your Teammate

Since your teammate has strong neural network experience and you are building your AI skills, here is how you can divide responsibilities productively without feeling left behind:

| Project Area | Beginner-Friendly / High-Value Contributions (Your Focus) | Advanced Deep Learning (Teammate's Focus) |
|---|---|---|
| **Data Analysis & Preprocessing** | - Inspect prediction error outliers (find which images the model failed on: rain, night, lens glare?).<br>- Visualize sky crops vs ground crops.<br>- Validate station metadata distributions. | - Implementing complex tensor batch augmentations.<br>- GPU dataloader optimization. |
| **Model Experiments** | - Run tabular feature experiments in Notebook 2 with different scikit-learn models (Random Forest, Gradient Boosting / XGBoost).<br>- Test the effect of adding temporal/cyclic features. | - Coding custom PyTorch attention layers and multi-task loss functions in Notebook 3. |
| **Literature & Benchmarking** | - Map your experimental results against the metrics in `paper_taxonomy.xlsx`.<br>- Write the methodology and results sections of the final report explaining why your station-grouped split avoids the data leakage seen in published papers. | - Tuning hyperparameters, learning rate schedules, and gradient clipping. |
| **Presentation & Interpretation** | - Create before-and-after scatter plots comparing true AQI vs predicted AQI.<br>- Formulate physical explanations for why certain filters succeed or fail. | - Model checkpointing, export, and inference benchmarks. |

---

## Summary Checklist to Get Started Today

1. [ ] **Read this guide carefully** to internalize the big picture.
2. [ ] Open [`1.eda-image-datasets.ipynb`](1.eda-image-datasets.ipynb) and look at the images and filter visualizations in Section 4.
3. [ ] Complete the free Kaggle "Pandas" micro-course to understand tabular data filtering.
4. [ ] Watch the 3Blue1Brown video on Neural Networks.
5. [ ] Discuss the Level 1 and Level 2 next steps with your teammate!
