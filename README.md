
# Uncertainty in Deep Learning Models for Building Detection in Satellite Images

This project investigates the uncertainties in deep learning models, specifically the U-Net architecture, for building detection in satellite images. The focus is on both data and model uncertainties, achieved through hyperparameter tuning and analysis of model performance across noisy and non-noisy datasets. This work was conducted at the National Remote Sensing Centre (NRSC), ISRO.

## Introduction
- All models are wrong, but models that know when they are   wrong, are useful.
- All DLmodels comes with its own advantages and limitations.
- Understanding about uncertainties will enable fine tuning of
  models, make it more robust and improve its performance in   blind spots.
- Present study focuses on the uncertainties and quantification  of its influence on the final results.
- Addressing uncertainties in DL models is an active area of research for Deep learning domain.

## Project Overview

Satellite imagery is invaluable for monitoring urban environments and detecting buildings. However, challenges like data quality and model uncertainties often impact the accuracy of deep learning models for such tasks. This project aims to address these issues through:

- **Uncertainty Quantification**: Analyzing both data (aleatoric) and model (epistemic) uncertainty.
- **Hyperparameter Tuning**: Using Bayesian optimization with Weights & Biases (W&B) sweeps.
- **Evaluation Metrics**: Assessing model performance using various metrics, including accuracy, AUC, precision, mean IoU, and confidence masks.

## Background

The study explores two types of uncertainty:
1. **Data Uncertainty**: Originating from noise in satellite images.
2. **Model Uncertainty**: Addressed through hyperparameter tuning to reduce the variability in model predictions.

For building detection, a U-Net model is used due to its strength in image segmentation tasks, and experiments are conducted with multiple configurations to observe the effects of different hyperparameters.

## Methodology

### Data Collection and Preprocessing
- **Dataset**: Satellite images of Hyderabad with building annotations, divided into noisy and non-noisy sets.
- **Normalization**: Both normalized and non-normalized versions of data are used for comparison.

### Model Architecture
- **U-Net**: Used for segmentation with a contracting-expansive path architecture, making it suitable for precise detection tasks.
- **Hyperparameters**: Key hyperparameters include optimizer (SGD, Adam), activation function (ReLU, Tanh), and loss function (Binary Cross-Entropy, Binary Focal Cross-Entropy).

### Uncertainty Analysis
- **Data Uncertainty**: Assessed by predictions on noisy and non-noisy datasets, with metrics such as accuracy, loss, and confusion matrix.
- **Model Uncertainty**: Addressed via tuning hyperparameters with Bayesian optimization using W&B sweeps.

## Implementation

The project is implemented using TensorFlow and involves:
1. **Experiment Setup**: Model training with various configurations on noisy and non-noisy datasets.
2. **Loss Functions**: Comparison of Binary Cross-Entropy and Binary Focal Cross-Entropy.
3. **Activation Functions**: ReLU and Tanh activations evaluated for optimal performance.
4. **Optimizers**: Adam and SGD optimizers compared for performance on building detection tasks.

## Results

### Model Performance
- **Best Configuration**: The Adam optimizer with ReLU activation and Binary Cross-Entropy loss function yielded the best performance in terms of accuracy, AUC, and IoU.
- **Confusion Matrix Insights**: Analysis revealed a high number of False Negatives, highlighting potential areas for model improvement.
- **Confidence Masks**: Visualizations of confidence scores provided insights into model certainty across predictions.

  ## Hyperparameter Tuning Results

-To evaluate the impact of different hyperparameters on model performance, multiple configurations were tested. The table below summarizes the accuracy, AUC, mean IoU, precision, and loss achieved for each configuration, highlighting the best-performing model settings.

![run](https://github.com/user-attachments/assets/9c72fbc9-2737-4975-bb5e-5de2fa9af10e)

*Figure: Comparison of different optimizer, activation, and loss function configurations.*


## Conclusion

The project demonstrates the importance of uncertainty quantification in enhancing model reliability. The Adam optimizer with ReLU activation and Binary Cross-Entropy loss was found to be optimal for this task. High False Negative rates suggest that further fine-tuning could improve recall.

## References

1. Alsabhan, W., et al. "Detecting Buildings from Satellite Images Using U-Net." *Computational Intelligence and Neuroscience*, 2022.
2. Weights & Biases documentation.

## Acknowledgments

Special thanks to my mentors Miss. Reedhi Shukla and Mr. Sampath Kumar at NRSC, ISRO, and the support from NRSC staff. This work was carried out under their valuable guidance.

