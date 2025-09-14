# Serial Dependence Model Mathematical Formula

## Complete Multi-Dimensional Serial Dependence Model

The logistic regression model for predicting choice behavior with serial dependence effects is:

$$\text{logit}(P(\text{action} = 1)) = \beta_0 + \beta_1 \cdot \text{angle} + \sum_{m=1}^{3} \beta_{m+1} \cdot \mathbf{1}[\text{mod} = m] + \sum_{i=1}^{k} \left[ \beta_{\text{choice},i} \cdot \text{action}_{n-i} + \beta_{\text{perceptual},i} \cdot \text{angle}_{n-i} + \beta_{\text{outcome},i} \cdot \text{hitmiss}_{n-i} + \beta_{\text{difficulty},i} \cdot \text{difficulty}_{n-i} \right]$$

Where:
- $P(\text{action} = 1)$ is the probability of choosing "turn right" (action = 1)
- $k$ is the history depth (typically $k = 3$)
- $\mathbf{1}[\cdot]$ is the indicator function

## Expanded Form (for k=3)

$$\begin{align}
\text{logit}(P(\text{action} = 1)) = &\beta_0 + \beta_1 \cdot \text{angle} \\
&+ \beta_2 \cdot \mathbf{1}[\text{mod} = 1] + \beta_3 \cdot \mathbf{1}[\text{mod} = 2] + \beta_4 \cdot \mathbf{1}[\text{mod} = 3] \\
&+ \beta_5 \cdot \text{action}_{n-1} + \beta_6 \cdot \text{action}_{n-2} + \beta_7 \cdot \text{action}_{n-3} \\
&+ \beta_8 \cdot \text{angle}_{n-1} + \beta_9 \cdot \text{angle}_{n-2} + \beta_{10} \cdot \text{angle}_{n-3} \\
&+ \beta_{11} \cdot \text{hitmiss}_{n-1} + \beta_{12} \cdot \text{hitmiss}_{n-2} + \beta_{13} \cdot \text{hitmiss}_{n-3} \\
&+ \beta_{14} \cdot \text{difficulty}_{n-1} + \beta_{15} \cdot \text{difficulty}_{n-2} + \beta_{16} \cdot \text{difficulty}_{n-3}
\end{align}$$

## Variable Definitions

### Current Trial Variables
- $\text{angle}$: Current stimulus angle (0° - 90°)
- $\text{mod}$: Current sensory modality (1=Touch, 2=Vision, 3=Visual-Tactile)

### Historical Variables (for trial n-i)
- $\text{action}_{n-i}$: Previous choice (0=left, 1=right)
- $\text{angle}_{n-i}$: Previous stimulus angle (0° - 90°)
- $\text{hitmiss}_{n-i}$: Previous trial outcome (0=miss, 1=hit)
- $\text{difficulty}_{n-i}$: Previous trial difficulty = $|\text{angle}_{n-i} - 45°|$

## Coefficient Interpretation

### Four Types of Serial Dependence Effects

1. **Choice Effects** ($\beta_{\text{choice},i}$): 
   - Motor memory and decision bias
   - Positive values indicate attractive effects (repeat previous choice)
   - Negative values indicate repulsive effects (avoid previous choice)

2. **Perceptual Effects** ($\beta_{\text{perceptual},i}$):
   - Sensory adaptation and expectation
   - Negative values indicate repulsive effects (bias away from previous stimulus)
   - Positive values indicate attractive effects (bias toward previous stimulus)

3. **Outcome Effects** ($\beta_{\text{outcome},i}$):
   - Reinforcement learning
   - Positive values indicate win-stay behavior
   - Negative values indicate lose-shift behavior

4. **Difficulty Effects** ($\beta_{\text{difficulty},i}$):
   - Confidence-weighted serial dependence
   - Negative values indicate that difficult (uncertain) trials have reduced influence
   - Represents confidence modulation of history effects

## Model Properties

- **Algorithm**: Logistic Regression with L2 regularization
- **Feature Standardization**: StandardScaler preprocessing applied to all features
- **Total Features**: 4 + 4k (where k is history depth)
  - 1 current angle + 3 modality indicators + 4k historical features
  - For k=3: 16 total features
- **Regularization**: $C = 1.0$ (inverse regularization strength)
- **Solver**: Limited-memory BFGS (lbfgs)

## Probability Transformation

The final choice probability is obtained via the logistic function:

$$P(\text{action} = 1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \cdot \text{angle} + \ldots)}}$$

## Model Variants

### Modality-Specific Models

For analyzing cross-modal effects, simplified models are used:

**Perceptual History Model:**
$$\text{logit}(P(\text{action} = 1)) = \beta_0 + \beta_1 \cdot \text{angle} + \beta_2 \cdot \text{angle}_{n-1} + \beta_3 \cdot \mathbf{1}[\text{vision}]$$

**Choice History Model:**
$$\text{logit}(P(\text{action} = 1)) = \beta_0 + \beta_1 \cdot \text{angle} + \beta_2 \cdot \text{action}_{n-1} + \beta_3 \cdot \mathbf{1}[\text{vision}]$$

Where $\mathbf{1}[\text{vision}] = \mathbf{1}[\text{mod} = 2]$ is an indicator for vision modality.

---

*This mathematical formulation captures the complete multi-dimensional serial dependence model implemented in the `SerialDependenceAnalyzer` class.*
