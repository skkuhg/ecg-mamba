# ECG-Mamba Cardiac Abnormality Classification

A deep learning project implementing the Mamba architecture for ECG cardiac abnormality classification using the PTB-XL dataset from PhysioNet.

## Overview

This project demonstrates the application of the Mamba state space model for multi-lead ECG signal classification. The model is trained to classify cardiac abnormalities into diagnostic superclasses including NORM, MI, STTC, CD, and HYP.

## Features

- **Mamba Architecture**: Utilizes the efficient Mamba state space model for sequence modeling
- **PTB-XL Dataset**: Automated downloading and preprocessing of ECG data from PhysioNet
- **Multi-lead ECG**: Processes all 12 standard ECG leads
- **GPU Accelerated**: Optimized for GPU training with CUDA support
- **Visualization**: Includes prediction visualization for model interpretation

## Model Architecture

The ECGMambaClassifier consists of:
- Linear embedding layer (12 channels → 64 dimensions)
- 2 Mamba layers with state space modeling
- Layer normalization
- Classification head for multi-class prediction

Key hyperparameters:
- `d_model`: 64 (model dimension)
- `d_state`: 16 (state space dimension)
- `d_conv`: 4 (convolution kernel size)
- `expand`: 2 (expansion factor)

## Requirements

- Python 3.7+
- PyTorch
- mamba-ssm
- causal-conv1d >= 1.2.0
- wfdb
- pandas
- numpy
- scikit-learn
- matplotlib
- requests

## Installation

```bash
pip install torch mamba-ssm causal-conv1d wfdb pandas numpy scikit-learn matplotlib requests
```

**Note**: This project requires a GPU to run the Mamba implementation efficiently. For Google Colab, ensure Runtime type is set to T4 GPU or better.

## Usage

### Running in Google Colab

1. Open the notebook in Google Colab
2. Set Runtime type to GPU (Runtime → Change runtime type → T4 GPU)
3. Run all cells sequentially

The notebook will automatically:
1. Install all dependencies
2. Download a subset of PTB-XL dataset (500 records)
3. Preprocess and normalize ECG signals
4. Train the Mamba model for 10 epochs
5. Evaluate on test set and visualize predictions

### Dataset

The notebook uses the PTB-XL dataset, a large publicly available electrocardiography dataset containing:
- 21,837 clinical 12-lead ECGs from 18,885 patients
- 10 second recordings at 100Hz (low resolution) or 500Hz (high resolution)
- Multiple diagnostic statements by cardiologists

For this implementation, we use:
- 500 records (configurable via `NUM_RECORDS`)
- Low resolution (100Hz) for faster processing
- Single diagnostic superclass per record for simplified classification

Dataset citation:
```
Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2020).
PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3).
PhysioNet. https://doi.org/10.13026/x4td-x982
```

## Training Results

Typical training performance (10 epochs, 500 records):
- Training Accuracy: ~75%
- Test Accuracy: ~70%
- Training time: ~2-3 minutes on T4 GPU

## Model Performance

The model achieves competitive performance on cardiac abnormality classification:
- Fast inference time thanks to Mamba's efficient architecture
- Good generalization on multi-class ECG classification
- Potential for improvement with larger datasets and longer training

## Visualization

The notebook includes visualization of:
- Multiple ECG leads with offset for clarity
- True vs. predicted diagnostic classes
- Sample predictions from the test set

## Project Structure

```
.
├── ECG_Mamba_Colab_Test.ipynb  # Main Jupyter notebook
├── README.md                     # This file
├── LICENSE                       # MIT License
└── ptb_xl_data/                 # Downloaded dataset (created at runtime)
```

## Diagnostic Classes

The model classifies ECG signals into the following diagnostic superclasses:
- **NORM**: Normal ECG
- **MI**: Myocardial Infarction
- **STTC**: ST/T Change
- **CD**: Conduction Disturbance
- **HYP**: Hypertrophy

## Future Improvements

- [ ] Increase dataset size for better generalization
- [ ] Implement cross-validation for robust performance metrics
- [ ] Add data augmentation techniques
- [ ] Experiment with deeper Mamba architectures
- [ ] Support for multi-label classification
- [ ] Model deployment pipeline
- [ ] Real-time ECG inference API

## References

1. **Mamba**: Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
2. **PTB-XL**: Wagner et al. (2020). PTB-XL, a large publicly available electrocardiography dataset.
3. **PhysioNet**: Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PhysioNet for providing the PTB-XL dataset
- The Mamba-SSM team for the efficient state space model implementation
- Google Colab for providing free GPU resources

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ecg_mamba_2024,
  title={ECG-Mamba: Cardiac Abnormality Classification using Mamba Architecture},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ecg-mamba}
}
```

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.
