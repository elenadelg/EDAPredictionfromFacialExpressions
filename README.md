# Predicting Electrodermal Activity from Facial Expresssion Analysis

## 📝 Introduction

This project **investigates ElectroDermal Activity (EDA) prediction using video-based facial expression analysis**. The study evaluates and compares several **deep learning architectures**, including **PhysNet, DeepPhys, and CNN-based baselines**, to explore their effectiveness in predicting EDA under real-world conditions using the UBFC-Phys dataset. It explores:
- Deep learning models performance for predicting EDA from facial videos.
- Data augmentation and face cropping effectiveness in reducing overfitting and improving generalization.
- Assess generalization capabilities and explore limitations in handling small, noisy datasets.


---

## 📊 Dataset: UBFC-Phys

The UBFC-Phys Dataset ([Sabour et al., 2021][1]) was used, which contains:
- **Participants:** 56 individuals aged 19 to 38.
- **Data streams:** 
  - Synchronized video frames (4 FPS)
  - EDA signals (4 Hz)
  - Blood Volume Pulse (BVP) signals (64 Hz)
- **Tasks:**  
  Each participant was recorded during three tasks simulating different physiological states:
  - Resting
  - Speech
  - Arithmetic tests

---

### ⚙️  Models 

#### 1. **Sequential CNNs**
- A **Sequential CNN** learns temporal patterns from facial video frames and integrates **LSTMs** to capture dependencies over time.
- **Sequential CNN + BVP:** Incorporates BVP signals alongside video data to enhance prediction accuracy.

#### 2. **PhysNet** ([Yu et al., 2019][2])
- Employs **spatio-temporal 3D convolutions** to capture blood volume variations across facial frames.
- Enhanced to handle BVP signals for better prediction of EDA.

#### 3. **DeepPhys**  ([Chen & McDuff, 2018][3])
- Incorporates **attention mechanisms** to focus on physiologically relevant regions, such as skin areas.
- Processes appearance and motion tensors** via a dual-design **CNN architecture**.

---

### 📏 Evaluation Metrics
- **Mean Squared Error (MSE):** Measures absolute accuracy by quantifying the difference between actual and predicted EDA signals.
- **Negative Pearson Correlation (NPC):** Evaluates the linear correlation between actual and predicted EDA signals.  
- **Grad-CAM:** Visualizes model attention, highlighting the facial regions that contribute the most to EDA predictions. ([Selvaraju et al., 2017][4]).

---

## 🔬 Experiments

- **Data Augmentation**: Random transformations (e.g., rotations, color jitter, flips) help models learn generalized patterns and reduce overfitting.
- **Face Cropping**:  Face regions are isolated using the **face_recognition library** ([Ageitgey, 2018][5]), allowing the models to focus on meaningful features and reduce noise.

---

## ⚠️ Challenges and Limitations

### 1. **Data Quality and Diversity**
- The small dataset (56 participants) and **gender imbalance** limited model generalization.
- **Systematic bias** was introduced due to homogeneous recording environments.

### 2. **Limited Temporal Resolution**
- Downsampling video from **35 FPS to 4 FPS** reduced the available temporal context, limiting the ability to capture subtle variations in facial expressions.

---
## 🔍 **References**

[1] Sabour, R. M., Benezeth, Y., De Oliveira, P., Chappe, J., & Yang, F. (2021). "UBFC-Phys: A multimodal database for psychophysiological studies of social stress." IEEE Transactions on Affective Computing.

[2] Yu, Z., Li, X., Zhao, G., & Pietikäinen, M. (2019). "Remote Photoplethysmograph Signal Measurement From Facial Videos Using Spatio-Temporal Convolutional Networks." Proceedings of the AAAI Conference on Artificial Intelligence.

[3] Chen, W., & McDuff, D. (2018). "DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks." European Conference on Computer Vision (ECCV).

[4] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). "Grad-CAM: Visual explanations from deep networks via gradient-based localization." Proceedings of the IEEE International Conference on Computer Vision (ICCV).

[5] Ageitgey, A. (2018). "face_recognition: The world’s simplest facial recognition API for Python and the command line." GitHub repository: https://github.com/ageitgey/face_recognition.



---

## 🛠️ Repository Structure

```plaintext
/CV
    ├── notebooks/                        # Jupyter notebooks for exploratory data analysis
    │   ├── exploratorydataanalysis.ipynb  
    │   └── summary_patients.ipynb         
    │
    ├── src/                              # Source code for the project
    │   ├── data/                         
    │   │   ├── datacollection/            
    │   │   ├── dataloader/                
    │   │   └── dataprocessing/            
    │   │
    │   ├── utils.py                      # Utility functions for eda tasks
    │
    │   ├── evaluation/                   
    │   │   ├── gradcam/                   # Grad-CAM visualizations
    │   │   └── loss/                      # Loss computation and NPC evaluation
    │   │
    │   ├── models/                       # Core model architectures
    │   │   ├── DeepPhys.py                
    │   │   ├── PhysNet.py                 
    │   │   ├── seqCNN_BVP.py              
    │   │   └── seqCNN.py                  
    │   │
    │   ├── evaluate.py                   
    │   └── training.py                   
    │
    ├── .python-version                  
    ├── pyproject.toml                   
    ├── README.md                         
    ├── requirements.txt                  
    └── JobParameters.json                
