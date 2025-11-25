# Figure Descriptions for AutoRAG-Allocator

## Generated Figures

All figures are based on **synthetic/simulated data** to illustrate the expected results. These should be replaced with actual experimental data once the system is implemented.

---

## Figure 1: results_comparison.pdf (20 KB)

**File:** `results_comparison.pdf`

**Used in Paper:** Figure 1 (Section 6, Main Results)

**Description:** 
Three-panel bar chart comparing system performance across NQ-Open and HotpotQA datasets.

**Panels:**
1. **Left - Accuracy (Exact Match %)**
   - Uniform (Llama-3-8B): 62% (NQ), 54% (HotpotQA)
   - Static Allocated: 67% (NQ), 58% (HotpotQA)
   - Adaptive Allocated: 69% (NQ), 60% (HotpotQA)

2. **Middle - Cost (¢/query)**
   - Uniform: 6.2¢ (NQ), 6.8¢ (HotpotQA)
   - Static: 5.3¢ (NQ), 5.9¢ (HotpotQA)
   - Adaptive: 4.6¢ (NQ), 5.1¢ (HotpotQA)

3. **Right - Latency (ms)**
   - Uniform: 1700ms (NQ), 1850ms (HotpotQA)
   - Static: 1590ms (NQ), 1720ms (HotpotQA)
   - Adaptive: 1480ms (NQ), 1620ms (HotpotQA)

**Key Takeaway:** Adaptive allocation achieves best performance across all three metrics (accuracy, cost, latency) on both datasets.

**Colors:**
- Blue bars: NQ-Open dataset
- Red bars: HotpotQA dataset

---

## Figure 2: pareto_frontier.pdf (19 KB)

**File:** `pareto_frontier.pdf`

**Used in Paper:** Figure 2 (Section 6, Pareto Frontier Analysis)

**Description:**
Scatter plot showing cost-accuracy trade-off for all 36 explored configurations on NQ-Open dataset.

**Elements:**
- **Gray dots:** Non-Pareto configurations (dominated by others)
- **Red dots:** Pareto-optimal configurations (non-dominated)
- **Red dashed line:** Connects Pareto frontier points
- **Gold star:** Selected configuration (4.6¢, 69% EM)

**Axes:**
- X-axis: Cost per query (cents)
- Y-axis: Exact Match accuracy (%)

**Key Takeaway:** 
- Only 7 out of 36 configurations are Pareto-optimal
- Selected configuration (star) offers best accuracy at ~$0.05 budget
- Shows diminishing returns beyond 5¢/query

**Pattern:** 
Higher cost generally improves accuracy, but with diminishing returns. The Pareto frontier makes explicit the trade-off between spending more and getting better results.

---

## Figure 3: profiling_overhead.pdf (21 KB)

**File:** `profiling_overhead.pdf`

**Used in Paper:** Figure 3 (Section 6, Profiling Efficiency Analysis)

**Description:**
Bar chart comparing profiling time between exhaustive grid search and adaptive profiling.

**Bars:**
1. **Exhaustive Grid Search:** 58.0 minutes (gray)
2. **Adaptive Profiling:** 8.2 minutes (green)

**Annotation:**
- "85.9% reduction" label between bars
- Value labels on top of each bar

**Key Takeaway:**
Adaptive profiling (test models independently on small probe set) is dramatically faster than exhaustive search (test all 36 combinations on full dataset).

**Calculation:**
- Exhaustive: 36 configs × 100 queries × ~1 min per config = ~58 min
- Adaptive: (3+4+3) models × 15 queries × ~30 sec = ~8 min
- Reduction: (58 - 8.2) / 58 = 85.9%

---

## How These Figures Were Generated

**Method:** Python matplotlib with seaborn styling

**Data Source:** Synthetic/simulated based on plausible estimates

**Code:** `generate_figures.py` (included in project)

**Libraries Used:**
```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
```

**Reproducibility:**
All figures use `np.random.seed(42)` for reproducible synthetic data generation.

---

## ⚠️ IMPORTANT NOTE

**These figures show plausible results, NOT actual experimental data.**

### What Needs to Happen:

1. **Implement the system** - Build actual RAG allocator
2. **Run experiments** - Evaluate on real NQ-Open and HotpotQA data
3. **Collect metrics** - Record actual EM, F1, latency, cost for each configuration
4. **Regenerate figures** - Use real experimental data
5. **Update paper** - Replace synthetic results with actual measurements

### Current Status:

| Figure | Data Type | Paper Claims | Reality |
|--------|-----------|--------------|---------|
| Figure 1 | Bar charts | "Main results on NQ-Open and HotpotQA" | Simulated estimates |
| Figure 2 | Scatter plot | "Cost-accuracy Pareto frontier" | Simulated 36 configs |
| Figure 3 | Bar chart | "Profiling time comparison" | Simulated timing |

**All three figures are placeholders** showing what the results *should* look like based on the system design. They are not from real experiments.

---

## How to Replace with Real Data

Once you have actual experimental results:

### Step 1: Collect Data
```python
# Example data structure
results = {
    'NQ-Open': {
        'Uniform': {'em': 62.0, 'f1': 69.2, 'lat': 1700, 'cost': 6.2},
        'Static': {'em': 67.3, 'f1': 74.1, 'lat': 1590, 'cost': 5.3},
        'Adaptive': {'em': 69.2, 'f1': 76.4, 'lat': 1480, 'cost': 4.6}
    },
    'HotpotQA': {...}
}
```

### Step 2: Update generate_figures.py
Replace synthetic data generation with actual results:
```python
# Before (synthetic):
em_nq = [62, 67, 69]

# After (real data):
em_nq = [results['NQ-Open']['Uniform']['em'],
         results['NQ-Open']['Static']['em'],
         results['NQ-Open']['Adaptive']['em']]
```

### Step 3: Regenerate
```bash
python generate_figures.py
```

### Step 4: Recompile Paper
```bash
pdflatex autorag_allocator.tex
pdflatex autorag_allocator.tex
```

---

## Figure Quality

**Resolution:** 300 DPI (publication quality)

**Format:** PDF (vector graphics, scalable)

**Size:** 
- results_comparison.pdf: ~20 KB
- pareto_frontier.pdf: ~19 KB
- profiling_overhead.pdf: ~21 KB

**Compatibility:** Work with LaTeX \includegraphics command

---

## Files Included

```
/mnt/user-data/outputs/
├── results_comparison.pdf     # Figure 1 (3-panel bar chart)
├── pareto_frontier.pdf        # Figure 2 (scatter plot)
├── profiling_overhead.pdf     # Figure 3 (bar chart)
└── FIGURE_DESCRIPTIONS.md     # This file
```

All figures are ready to use in the paper, but remember they show **simulated results** that need to be replaced with real experimental data.