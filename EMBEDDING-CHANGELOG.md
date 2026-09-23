# Embedding Change Plan: Border-Relative Instance Embeddings

## Objective

Extend the current embedding-based instance segmentation formulation into a richer mode system that supports both the existing center-relative formulation and a new border-relative formulation, and further supports two instance-separation strategies:

- seed-based instance differentiation;
- embedding-clustering-based instance differentiation.

The implementation must be exposed as a parameter of the main training function, and invalid values must raise a `ValueError`.

The supported values are:

- `center-seed`: current approach with center-relative vectors and seed-based instance differentiation;
- `border-seed`: vectors pointing toward the nearest border, but using seed-based instance differentiation;
- `center-cluster`: vectors pointing toward the center, but using embedding clustering for instance differentiation;
- `border-cluster`: vectors pointing toward the nearest border, but using embedding clustering for instance differentiation;
- `combined-center`: two vectors are predicted, one toward the center and one toward the border, while using seed-based instance differentiation;
- `combined-cluster`: two vectors are predicted, one toward the center and one toward the border, while using embedding clustering for instance differentiation.

---

## 1. High-Level Design

The current approach assumes that each pixel encodes a relative displacement to the instance center. The proposed extension preserves the same overall pipeline, but changes the reference target from a center anchor to a border anchor.

### Center-relative behavior

For each pixel $i$ belonging to an instance $k$:

- a displacement vector is learned toward the instance centroid $c_k$;
- the model predicts a coordinate embedding relative to that centroid;
- the pixel classifier learns a representation of instance shape relative to the center.

### Border-relative behavior

For each pixel $i$ belonging to an instance $k$:

- a displacement vector is learned toward the nearest border point of that instance;
- the model predicts a coordinate embedding relative to that border reference;
- the pixel classifier learns a representation of instance shape relative to the nearest border.

### Combined behavior

For each pixel $i$ belonging to an instance $k$:

- two displacement vectors are predicted simultaneously:
  - one toward the centroid;
  - one toward the nearest border;
- the model can be trained to support both geometric references jointly;
- the downstream instance differentiation strategy decides whether the representation is consumed through seeds or through clustering.

This preserves the existing distance-to-nearest-border supervision paradigm, but extends the geometric reference from a single centroid-based target to a richer set of target formulations.

---

## 2. Theoretical Requirements

### 2.1 Semantic definition of the new targets

The supported embedding targets must be defined as one or two vector fields:

- center-relative target:

$$
\mathbf{v}_i^{(center)} = \mathbf{c}_k - \mathbf{g}_i
$$

- border-relative target:

$$
\mathbf{v}_i^{(border)} = \mathbf{b}_i - \mathbf{g}_i
$$

where:

- $\mathbf{g}_i$ is the spatial coordinate of pixel $i$;
- $\mathbf{c}_k$ is the centroid of instance $k$;
- $\mathbf{b}_i$ is the coordinate of the nearest point on the instance border for that pixel.

The target is therefore either a vector from the current pixel toward the centroid or a vector from the current pixel toward the nearest border of the same instance.

For the combined modes, the supervision must contain both vector targets jointly.

### 2.2 Relationship to the existing paradigm

The current center-relative formulation is preserved as one of the supported modes. The border-relative formulation extends it by replacing the centroid with a border anchor while keeping the same general embedding logic. The combined modes add a second target field so the model can learn both geometric references simultaneously.

### 2.3 Geometric meaning

The border-relative embedding should satisfy the following properties:

- pixels near the border should have small displacement vectors;
- pixels far from the border should have larger displacement vectors;
- the field should be consistent across elongated, touching, and irregular objects;
- the learned representation should support instance separation by local boundary geometry rather than by center geometry.

### 2.4 Training supervision requirements

The coordinate branch should still be supervised with the same type of regression target as before, but the target values must be generated from the border reference instead of the centroid reference.

The seed branch may remain conceptually similar, but its interpretation should be validated so that it still provides a stable set of candidate instance centers or anchors for the new embedding mode.

### 2.5 Inference and instance separation requirements

During inference:

- for the seed-based modes (`*-seed`), candidate centers or anchor points must still be extracted from the seed map;
- for the clustering modes (`*-cluster`), instance differentiation must be performed by grouping pixels according to the learned embedding field rather than by using explicit seeds;
- local feature extraction around each candidate must use the same embedding mode that was used during training;
- the pixel classifier must distinguish foreground pixels from background pixels using the selected embedding features.

The new modes must be evaluated for cases where the center-relative approach may be less reliable, especially:

- touching objects;
- elongated objects;
- objects with irregular boundaries;
- crowded scenes with ambiguous centers.

### 2.6 Embedding clustering procedure for the `*-cluster` modes

The clustering-based modes replace the explicit seed map with a seedless grouping procedure operating directly on the dense embedding field. The goal is to partition foreground pixels into instance-level clusters by using both the learned embedding vectors and the local spatial context.

#### 2.6.1 Inputs

The clustering procedure consumes:

- a dense embedding tensor $\mathbf{E} \in \mathbb{R}^{H \times W \times D}$ predicted by the model;
- a foreground confidence or mask $M \in \{0,1\}^{H \times W}$, either from the model output or from a thresholded foreground probability map;
- optional boundary evidence $B \in [0,1]^{H \times W}$, which may be derived from the model or from the predicted embedding field;
- optional coordinate maps $\mathbf{g}_i$ for spatial regularization.

For the combined modes, the embedding tensor must contain both the center-relative and border-relative vector channels, either by concatenation or by storing them in a structured representation that preserves both components.

#### 2.6.2 Feature construction

For each foreground pixel $i$, define a feature vector:

$$
\mathbf{z}_i = \left[ \mathbf{e}_i, \mathbf{g}_i \right]
$$

where:

- $\mathbf{e}_i$ is the learned embedding vector for pixel $i$;
- $\mathbf{g}_i$ is the pixel’s spatial coordinate (or a normalized coordinate embedding).

In the combined modes, $\mathbf{e}_i$ should be the concatenation of the center-relative and border-relative embedding components.

#### 2.6.3 Affinity definition

The clustering procedure must define an affinity between neighboring pixels $i$ and $j$ that is high when the pixels are likely to belong to the same instance and low when they are likely to lie on different instances. A suitable affinity is:

$$
A_{ij} = \exp\left(-\frac{\|\mathbf{e}_i - \mathbf{e}_j\|^2}{2\sigma_e^2}\right)
\cdot
\exp\left(-\frac{\|\mathbf{g}_i - \mathbf{g}_j\|^2}{2\sigma_g^2}\right)
\cdot
\left(1 - B_{ij}\right)
$$

where:

- $B_{ij}$ is the boundary penalty between pixels $i$ and $j$;
- $\sigma_e$ controls how strongly embedding similarity is enforced;
- $\sigma_g$ controls the strength of spatial locality.

The implementation may use an 8-neighborhood graph, a small local window, or a sparse graph built from the nearest neighbors in feature space.

#### 2.6.4 Graph construction

The procedure must construct a graph over all foreground pixels:

- each foreground pixel becomes one node;
- edges connect neighboring pixels when the affinity is above a configurable threshold;
- edges that cross strong boundary evidence should be weakened or removed.

This graph should be sparse enough to be efficient, but dense enough to preserve local connectivity for objects with irregular shapes.

#### 2.6.5 Grouping strategy

The clustering step must assign each foreground pixel to one instance cluster. Two equivalent implementation strategies are acceptable:

1. Connected-components style grouping
   - build the graph and group connected components after thresholding the affinity graph;
   - optionally merge small components or split large ones using a refinement pass.

2. Watershed or agglomerative clustering style grouping
   - initialize candidate basins from high-confidence foreground regions;
   - merge neighboring regions according to the learned affinity;
   - stop merging when the boundary penalty becomes too strong or when the affinity falls below threshold.

The implementation should be deterministic and should not require explicit seed points.

#### 2.6.6 Cluster refinement

After the initial clustering, the procedure should apply a small refinement pass:

- remove clusters smaller than a minimum size threshold;
- merge tiny fragments into the nearest neighboring cluster if the affinity is high;
- split clusters that are too large or that contain internally disconnected high-confidence subregions;
- enforce spatial compactness or local consistency where appropriate.

This refinement stage is important for avoiding over-fragmentation and for preventing a single object from being split into many clusters.

#### 2.6.7 Output format

The final output of the clustering procedure must be a dense label map:

$$
\mathbf{L} \in \{0, 1, 2, \dots, K\}^{H \times W}
$$

where:

- $0$ denotes background;
- each positive integer denotes one instance cluster.

The procedure should also be able to return auxiliary information such as:

- cluster size;
- cluster centroid or medoid;
- per-pixel confidence values;
- the number of clusters detected.

#### 2.6.8 Training-time interpretation

For the clustering modes, the embedding field must be trained so that pixels belonging to the same instance become close in embedding space while pixels from different instances become more separated. The supervision should therefore encourage:

- consistency of the learned embedding with the selected geometric target (center, border, or combined);
- local smoothness of the embedding field;
- strong separation across instance boundaries.

This means the clustering procedure is not merely a postprocessing trick; it is an integral part of the model’s inference behavior and should be treated as a first-class part of the implementation plan.

---

## 3. Technical Requirements

### 3.1 Training parameter

The main training function must accept a parameter such as:

```python
embedding_mode: str = "center-seed"
```

The accepted values must be:

- `"center-seed"`: center-relative vectors with seed-based differentiation;
- `"border-seed"`: border-relative vectors with seed-based differentiation;
- `"center-cluster"`: center-relative vectors with clustering-based differentiation;
- `"border-cluster"`: border-relative vectors with clustering-based differentiation;
- `"combined-center"`: both center and border vectors with seed-based differentiation;
- `"combined-cluster"`: both center and border vectors with clustering-based differentiation.

Any other value must raise:

```python
ValueError("embedding_mode must be one of: 'center-seed', 'border-seed', 'center-cluster', 'border-cluster', 'combined-center', 'combined-cluster'")
```

This validation must happen before training begins and before target generation is executed.

### 3.2 Configuration integration

The same parameter should be accessible through the training configuration and overrides mechanism so that it can be specified from:

- the main training function call;
- config overrides passed to training;
- YAML configuration if that is supported by the project’s training pipeline.

### 3.3 Target-generation utilities

A new utility or set of utilities must be added to compute the border-relative embedding target from ground-truth masks.

Required behavior:

1. For each instance mask, identify the instance boundary.
2. For each foreground pixel in that instance, determine the nearest border point of the same instance.
3. Generate a displacement vector from the pixel to that border point.
4. Store the result in the same tensor shape expected by the existing coordinate branch.

The implementation should be deterministic and differentiable in the sense that the target generation is fixed and does not depend on learnable parameters.

### 3.4 Training loop integration

The training loop must branch on the selected mode:

- if `embedding_mode == "center-seed"`, use the current center-relative target-generation path and the seed-based postprocessing path;
- if `embedding_mode == "border-seed"`, use the border-relative target-generation path and the seed-based postprocessing path;
- if `embedding_mode == "center-cluster"`, use the center-relative target-generation path and the clustering-based postprocessing path;
- if `embedding_mode == "border-cluster"`, use the border-relative target-generation path and the clustering-based postprocessing path;
- if `embedding_mode == "combined-center"`, use both vector targets and the seed-based postprocessing path;
- if `embedding_mode == "combined-cluster"`, use both vector targets and the clustering-based postprocessing path.

No architecture change is required unless the implementation later proves that a slightly different head or feature formulation improves stability.

### 3.5 Feature engineering and inference

The postprocessing and feature-construction path must be updated so that the same reference convention is used at inference time as during training.

That means:

- center-relative modes use the current centroid-based feature engineering;
- border-relative modes use a border-based feature engineering path;
- combined modes use both geometric references in the feature construction path;
- seed-based modes use the current seed-driven proposal logic;
- clustering-based modes use a graph or watershed-like grouping strategy over the dense embedding field.

This is essential for consistency between training and inference.

### 3.6 Seed-map handling

The seed map logic should remain compatible with the seed-based modes. However, the behavior should be reviewed to ensure that:

- seed extraction remains stable;
- local maxima still correspond to candidate object anchors;
- the classifier can separate instances correctly for the border-based embedding field.

For clustering-based modes, the seed map is not required as the main instance-separation mechanism. Instead, the dense embedding field must be grouped directly using boundary-aware affinity and spatial continuity.

If needed, the seed-map supervision should be parameterized by the same mode to preserve semantic consistency.

### 3.7 Error handling and guardrails

The implementation must include clear checks for:

- unsupported modes;
- empty or malformed instance masks;
- shape mismatches between masks and predicted tensors;
- invalid values during target generation.

---

## 4. Implementation Steps

### Step 1: Define the new mode contract

- Introduce a new parameter in the main training function.
- Name it clearly, for example `embedding_mode`.
- Validate that only the six supported values are accepted.
- Add a helpful error message for invalid values.

### Step 2: Add border-based target generation

- Implement a utility that computes, for each foreground pixel, the nearest border point of its instance.
- Ensure the computation is performed per instance and uses the instance mask geometry.
- Generate the resulting displacement vectors in the expected coordinate tensor format.

### Step 3: Route training targets by mode

- Update the training pipeline so that the coordinate branch uses:
  - center targets for the center-based modes;
  - border targets for the border-based modes;
  - both targets for the combined modes.
- Keep the rest of the model unchanged unless required by testing.

### Step 4: Update feature engineering for inference

- Ensure the same reference convention is used during postprocessing and local classification.
- Make the feature extraction logic branch on the selected mode.
- For the clustering-based modes, implement a dense grouping strategy that consumes the embedding field without relying on explicit seeds.

### Step 5: Add regression and stability tests

Implement tests for:

- parameter validation for invalid values;
- correct target generation for each supported mode;
- shape correctness of the generated embedding tensors;
- consistency of feature construction between training and inference;
- correctness of the seed-based postprocessing path for the `*-seed` modes;
- correctness of the clustering-based grouping path for the `*-cluster` modes;
- a small end-to-end smoke test for the new modes.

### Step 6: Verify documentation and configuration

- Update the training documentation and example configs.
- Ensure the new parameter is discoverable in the repository documentation.

---

## 5. Testing Plan

### 5.1 Unit tests

Add unit tests for:

1. `embedding_mode="center-seed"` is accepted.
2. `embedding_mode="border-seed"` is accepted.
3. `embedding_mode="center-cluster"` is accepted.
4. `embedding_mode="border-cluster"` is accepted.
5. `embedding_mode="combined-center"` is accepted.
6. `embedding_mode="combined-cluster"` is accepted.
7. Any other string raises `ValueError`.
8. Border target generation returns the expected displacement direction for a simple synthetic mask.
9. Center target generation remains correct for the center-based modes.
10. Combined target generation produces both vector fields in the expected shape.
11. Clustering-based grouping produces the expected number of groups for a simple synthetic example.
12. Seed-based postprocessing remains compatible with the existing path.

### 5.2 Integration tests

Add a short integration test to confirm that:

- training can start with each supported mode;
- the training loop completes for a tiny synthetic dataset;
- the produced outputs have the expected shapes;
- the model can run inference in each supported mode without crashing.

### 5.3 Regression tests

Re-run the relevant existing tests to ensure that:

- the existing behavior remains unchanged for the legacy center-seed path;
- the old seed-based postprocessing remains compatible;
- no regressions appear in postprocessing or evaluation.

---

## 6. Acceptance Criteria

The change is complete when all of the following are true:

- the main training function accepts a mode parameter with the six supported values;
- invalid values raise `ValueError`;
- center-relative and border-relative targets are generated correctly depending on the selected mode;
- combined modes produce both vector fields in the expected shape and format;
- the training and inference pipelines use the same embedding convention for each selected mode;
- seed-based modes remain compatible with the existing postprocessing path;
- clustering-based modes implement a dense grouping strategy without explicit seeds;
- tests cover validation, target generation, grouping behavior, and basic training/inference behavior;
- the new modes are documented and configurable.
