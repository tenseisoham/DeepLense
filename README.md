# Strong Lensing Classification & Super-Resolution with Masked Autoencoders  

This repository contains experiments and implementations for classifying strong gravitational lensing images and enhancing their resolution using deep learning techniques. The project explores multiple approaches, including ResNet-18-based classification and Masked Autoencoder (MAE) pretraining, followed by ESRGAN-based super-resolution. Additionally, interpretability experiments are conducted to analyze model behavior.

![image](https://github.com/user-attachments/assets/efe2e73b-9755-4ff1-8e24-ff777d7ff5dc)
![image](https://github.com/user-attachments/assets/7d65af60-46a8-4977-b5ac-97de2ac45667)
![image](https://github.com/user-attachments/assets/5fcfb040-fe6f-4cc2-9611-7efde923f1b4)

- Some interesting results attached here!

### 1️⃣ Installation  

First, install the required dependencies using the provided requirements file:  

```bash
pip install -r requirements.txt
```

### 2️⃣ Download Datasets  

Download the datasets from the provided links and place them in the appropriate directories:  

- **General Task Dataset (Multi-Class Classification):**  
  [Download Here](https://drive.google.com/file/d/1doUhVoq1-c9pamZVLpvjW1YRDMkKO1Q5/view?usp=drive_link)  
- **Lens Finding Dataset:**  
  [Download Here](https://drive.google.com/file/d/1doUhVoq1-c9pamZVLpvjW1YRDMkKO1Q5/view?usp=drive_link)  
- **Masked Autoencoder Dataset:**  
  [Download Here](https://drive.google.com/file/d/1znqUeFzYz-DeAE3dYXD17qoMPK82Whji/view?usp=sharing)  
- **Super-Resolution Dataset:**  
  [Download Here](https://drive.google.com/file/d/1uJmDZw649XS-r-dYs9WD-OPwF_TIroVw/view?usp=sharing)  

---

## 📂 Repository Structure  

```
.
├───Foundation Model
│   ├───Task VI A
│   │   ├── MAE_pretraining.ipynb
│   │   ├── ResNet_finetuning_classification.ipynb
│   │   ├── MAE_finetuning_classification.ipynb
│   ├───Task VI B
│   │   ├── ESRGAN_super_resolution.ipynb
│
├───General_Task
│   ├── general_common_task.ipynb
│
└───TASK II - Lens Finding
│   ├── lens_finding.ipynb
```

---

## 🧪 Tasks & Experiments  

### 🔹 **1. General Task: Multi-Class Classification**  
**Notebook:** [general_common_task.ipynb](General_Task/general_common_task.ipynb)  

**Objective:**  
- Train a model to classify images into three categories:  
  1. **No Substructure**  
  2. **Cold Dark Matter (CDM) Substructure**  
  3. **Axion-like Particle Substructure**  

---

### 🔹 **2. Lens Finding Task (Binary Classification)**  
**Notebook:** [lens_finding.ipynb](TASK%20II%20-%20Lens%20Finding/lens_finding.ipynb)  

**Objective:**  
- Train a model to classify images into **lenses** and **non-lenses**.  

**Dataset Structure:**  
- **train_lenses/** – Images of strong gravitational lenses.  
- **train_nonlenses/** – Images of galaxies without strong lensing.  
- **test_lenses/** – Test set of lensed galaxies.  
- **test_nonlenses/** – Test set of non-lensed galaxies.  

**Challenges:**  
- The dataset is highly imbalanced, with significantly more non-lensed images.  
- Addressed class imbalance through re-weighting strategies.  

---

### 🔹 **3. Masked Autoencoder (MAE) for Feature Learning**  

**Pretraining Phase:**  
**Notebook:** [MAE_pretraining.ipynb](Foundation%20Model/Task%20VI%20A/MAE_pretraining.ipynb)  

**Objective:**  
- Train a simple CNN-based MAE to reconstruct masked portions of strong lensing images.  
- Learn meaningful feature representations of strong lensing structures.  

**Fine-Tuning on Classification Task:**  
**Notebook:** [MAE_finetuning_classification.ipynb](Foundation%20Model/Task%20VI%20A/MAE_finetuning_classification.ipynb)  

**Objective:**  
- Fine-tune the MAE encoder on the full dataset for multi-class classification.  
- Compare its performance against ResNet-18.  

**Additional Experiment:**  
**Notebook:** [ResNet_finetuning_classification.ipynb](Foundation%20Model/Task%20VI%20A/ResNet_finetuning_classification.ipynb)  

- An experimental ResNet-18 classification model trained on the same dataset for comparison.  

---

### 🔹 **4. Super-Resolution Using ESRGAN**  
**Notebook:** [ESRGAN_super_resolution.ipynb](Foundation%20Model/Task%20VI%20B/ESRGAN_super_resolution.ipynb)  

**Objective:**  
- Use a pre-trained MAE encoder from Task VI.A to fine-tune an **ESRGAN** model for super-resolution.  
- Upscale low-resolution strong lensing images using high-resolution samples as ground truths.  

**Evaluation Metrics:**  
- **MSE (Mean Squared Error)**  
- **SSIM (Structural Similarity Index Measure)**  
- **PSNR (Peak Signal-to-Noise Ratio)**  

---

## 🧐 Model Interpretability & Explainability  

To ensure the models are making decisions based on meaningful features and not artifacts in the simulated data, we perform **interpretability experiments** using **Grad-CAM**:  

- Analyze which parts of the images contribute most to the model's classification decision.  
- Identify potential cases where the model "cheats" by relying on background noise patterns instead of lensing features.  
- Implement physics-informed augmentations to remove such biases in future iterations. 

---

## Future Work  

- **Physics-Informed Augmentations:** Improve generalization by ensuring models don’t exploit spurious dataset artifacts.  
- **Contrastive Learning for Better Representations:** Utilize contrastive loss to enhance feature learning in the MAE pretraining phase.  
- **Uncertainty Quantification:** Evaluate model confidence and calibration in real-world applications.  

---

## 📊 Evaluation Metrics  

| Task                | Evaluation Metric(s) |
|--------------------|--------------------|
| Multi-Class Classification | ROC Curve, AUC Score |
| Lens Finding | ROC Curve, AUC Score |
| Super-Resolution | MSE, SSIM, PSNR |

---

## 📌 Citation  

If you use this repository in your research, please cite it appropriately.  

---

## 👨‍💻 Author  

Developed and maintained by **[Your Name]**.  

---

## 🤝 Contributions  

Contributions are welcome! Feel free to submit pull requests or open issues for discussions.  

---

## 📝 License  

This project is licensed under the **MIT License**.
```

---

### Key Features of This README:
✅ **Clear structure with well-defined sections**  
✅ **Markdown-friendly links to Jupyter notebooks**  
✅ **Dataset links for easy access**  
✅ **Explains model interpretability efforts**  
✅ **Summarizes evaluation metrics in a table**  
✅ **Future work section for improvements**  

Let me know if you'd like any modifications! 🚀
