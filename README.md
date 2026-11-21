# FATE-YOLO

**Fourier–Taylor Convolution and Entropy-Gated Dual-Affinity for Brain Tumor Detection**  
*ICASSP 2026* · [Paper](https://doi.org/10.1109/ICASSP55912.2026.11463467)

FATE-YOLO is a lightweight YOLO-based architecture for brain tumor detection in MRI images. This repository provides the **Ultralytics model implementation and configuration files** only; training scripts, datasets, and trained weights are not included.

## Contents

```text
ultralytics/
├── nn/
│   ├── modules/
│   │   └── fate_yolo.py
│   └── tasks.py
└── cfg/
    ├── models/
    │   └── fate-yolo/
    │       └── fate-yolo.yaml
    └── datasets/
        └── br35h.yaml
```

## Components

- **FTConv:** Fourier–Taylor kernel reconstruction combined with spatial convolution.
- **GLAA:** Window-based local affinity and pooled global affinity.
- **EGAA:** Entropy-guided fusion with optional edge-aware modulation.
- **DAABlock / GLDBlock:** Global–local aggregation with residual connections and normalization.
- **FATEHead:** Lightweight shared-feature detection head with boundary-aware task modulation.

## Integration

Place these files in a compatible Ultralytics source tree. Ensure `FTConv`, `DAABlock`, `GLDBlock`, and `FATEHead` are imported and handled by `parse_model()` in `ultralytics/nn/tasks.py`, including the three input feature channels for `FATEHead`.

The architecture is specified by `ultralytics/cfg/models/fate-yolo/fate-yolo.yaml`. The `br35h.yaml` file is a dataset configuration **template**: update its local paths before use. The dataset itself is not distributed here.

## Note

The supplied `fate_yolo.py` includes a paper-guided GLAA/EGAA implementation and an **experimental FATEHead**. It has not been verified as the exact code or weights used to obtain the published ICASSP results.

## Citation

Zhengnan Yin *et al.*, “FATE-YOLO: Fourier-Taylor Convolution and Entropy-Gated Dual-Affinity for Brain Tumor Detection,” *IEEE ICASSP*, 2026. [DOI: 10.1109/ICASSP55912.2026.11463467](https://doi.org/10.1109/ICASSP55912.2026.11463467).
