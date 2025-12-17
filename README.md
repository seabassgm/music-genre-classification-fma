# Music Genre Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Librosa-Feature_Extraction-orange)
![Library](https://img.shields.io/badge/TensorFlow-Deep_Learning-red)

> ** [Click here to read the full Project Report (PDF)](./docs/STMAE_Report_Sebas_Ceren_Mateo.pdf)**

## Project Overview
An end-to-end audio classification pipeline designed to categorize musical tracks from the **FMA-small dataset** into 8 distinct genres (Hip-Hop, Pop, Folk, Experimental, Rock, International, Electronic, Instrumental).

This project benchmarks traditional Machine Learning approaches against Deep Learning architectures to evaluate feature robustness and generalization.

## Audio Processing & Feature Engineering
Using **Librosa**, we engineered a multidimensional feature set to capture the timbral, rhythmic, and harmonic content of the audio:
* **Spectral Features:** MFCCs (13 coefficients), Spectral Contrast, Spectral Bandwidth, Rolloff, and Centroid.
* **Rhythmic Features:** Tempo (BPM) and Zero-Crossing Rate (ZCR).
* **Harmonic Features:** Chroma Vectors.

## Models Benchmarked
We implemented and compared four distinct architectures:
1.  **Random Forest Classifier:** Baseline ensemble model.
2.  **SVM (RBF Kernel):** To handle non-linear feature separation.
3.  **Multilayer Perceptron (MLP):** A fully connected neural network.
4.  **1D-CNN (Convolutional Neural Network):** Processing raw sequential feature data.

## Performance & Observations
* [cite_start]**Best Performers:** The **SVM (RBF)** and **1D-CNN** achieved the most balanced performance[cite: 1806].
* **Genre Specifics:**
    * **High Accuracy:** *Folk*, *Hip-Hop*, and *International* genres showed high F1-scores due to distinct spectral signatures.
    * [cite_start]**Challenges:** *Pop* and *Experimental* genres showed significant spectral overlap, leading to higher misclassification rates[cite: 1808].

## How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn librosa tensorflow matplotlib
