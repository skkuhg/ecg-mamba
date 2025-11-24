---
license: mit
tags:
- ecg
- mamba
- cardiac
- classification
- medical
- ptb-xl
- state-space-model
datasets:
- PTB-XL
language:
- en
library_name: pytorch
pipeline_tag: image-classification
---

# ECG-Mamba: Cardiac Abnormality Classification

## Model Description

ECG-Mamba is a deep learning model that leverages the Mamba state space architecture for classifying cardiac abnormalities from 12-lead ECG signals. The model is trained on the PTB-XL dataset from PhysioNet.

## Model Architecture

- **Base Architecture**: Mamba (Selective State Space Model)
- **Input**: 12-lead ECG signals (1000 timesteps × 12 channels at 100Hz)
- **Output**: 5-class classification (NORM, MI, STTC, CD, HYP)
- **Parameters**:
  - Model dimension (d_model): 64
  - State space dimension (d_state): 16
  - Number of Mamba layers: 2
  - Convolution kernel size (d_conv): 4
  - Expansion factor: 2

## Intended Use

This model is designed for:
- Research purposes in cardiac abnormality detection
- Educational demonstrations of Mamba architecture on medical signals
- Baseline comparison for ECG classification tasks

**Note**: This model is NOT intended for clinical diagnosis or medical decision-making.

## Training Data

- **Dataset**: PTB-XL (PhysioNet)
- **Training samples**: ~400 records (80% of 500 record subset)
- **Validation samples**: ~100 records (20% of 500 record subset)
- **Sampling rate**: 100 Hz (low resolution)
- **Signal length**: 10 seconds (1000 samples)
- **Preprocessing**: Standardization (zero mean, unit variance per channel)

## Performance

On the test subset (500 records):
- **Training Accuracy**: ~75%
- **Test Accuracy**: ~70%

**Important**: These metrics are from a small-scale demonstration. For production use, train on the full PTB-XL dataset (21,837 records).

## Diagnostic Classes

| Class | Description |
|-------|-------------|
| NORM | Normal ECG |
| MI | Myocardial Infarction |
| STTC | ST/T Change |
| CD | Conduction Disturbance |
| HYP | Hypertrophy |

## Usage

```python
import torch
import numpy as np
from mamba_ssm import Mamba

# Load model (you'll need to save/load weights separately)
model = ECGMambaClassifier(n_classes=5)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Prepare your ECG data
# ecg_signal: numpy array of shape (1000, 12)
ecg_tensor = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0)

# Inference
with torch.no_grad():
    logits = model(ecg_tensor)
    predicted_class = torch.argmax(logits, dim=1)
```

## Limitations

1. **Small training set**: Model trained on only 500 records for demonstration
2. **Simplified classification**: Single-label classification (many ECGs have multiple conditions)
3. **Class imbalance**: Not addressed in this implementation
4. **No clinical validation**: Not validated on independent clinical datasets
5. **Research use only**: Not approved for medical diagnosis

## Ethical Considerations

- This model should NOT be used for clinical diagnosis
- Medical decisions should only be made by qualified healthcare professionals
- The model may exhibit biases present in the PTB-XL dataset
- Performance may vary across different patient populations

## Training Procedure

### Preprocessing
1. Download PTB-XL records from PhysioNet
2. Extract low-resolution (100Hz) 12-lead ECG signals
3. Filter for single-label diagnostic superclass
4. Standardize signals (zero mean, unit variance)

### Training Hyperparameters
- **Optimizer**: AdamW
- **Learning rate**: 1e-3
- **Batch size**: 32
- **Epochs**: 10
- **Loss function**: CrossEntropyLoss
- **Hardware**: NVIDIA T4 GPU

### Data Augmentation
None applied in this implementation.

## Environmental Impact

- **Hardware**: NVIDIA T4 GPU (Google Colab)
- **Training time**: ~2-3 minutes
- **Carbon footprint**: Minimal due to short training time

## Citation

### This Model
```bibtex
@software{ecg_mamba_2024,
  title={ECG-Mamba: Cardiac Abnormality Classification using Mamba Architecture},
  year={2024},
  url={https://huggingface.co/your-username/ecg-mamba}
}
```

### PTB-XL Dataset
```bibtex
@article{wagner2020ptbxl,
  title={PTB-XL, a large publicly available electrocardiography dataset},
  author={Wagner, Patrick and Strodthoff, Nils and Bousseljot, Ralf-Dieter and Kreiseler, Dieter and Lunze, Fatima I and Samek, Wojciech and Schaeffter, Tobias},
  journal={Scientific Data},
  volume={7},
  number={1},
  pages={154},
  year={2020}
}
```

### Mamba
```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

## Model Card Authors

This model card was created as part of the ECG-Mamba project.

## Model Card Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/skkuhg/ecg-mamba).
