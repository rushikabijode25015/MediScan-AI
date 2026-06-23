# MediScan AI: Explainable Medical Diagnosis Platform

MediScan AI is an explainable medical imaging and decision support platform designed to detect radiographic anomalies (such as opacity, tumors, consolidations, or fractures) in medical imagery while maintaining clinical trust.

By combining deep transfer learning (ResNet-50) with Grad-CAM (Gradient-weighted Class Activation Mapping), MediScan AI addresses the **"black-box" challenge** of AI in healthcare. It does not just classify scans—it visually overlays and highlights the exact regions in the scans that influenced the model's classification, giving radiologists and clinical staff verification proof for model decisions.

---

## 🌟 Key Features

1. **Explainable AI (Grad-CAM Integration)**: Computes class-specific activation maps from the last convolutional layer (`layer4`) of ResNet-50. This reveals the spatial focus of the model, allowing doctors to audit the decision pathway.
2. **Transfer Learning Pipeline**: Built on PyTorch using ResNet-50 pre-trained on ImageNet, adapted for clinical classification with frozen convolutional backbones for rapid training/deployment.
3. **Interactive Clinical Dashboard**:
   - Drag-and-drop file uploader for patient images.
   - Clickable clinical sample presets (Normal vs Anomaly Chest X-Rays and Brain MRIs).
   - Real-time side-by-side visualization with an adjustable heatmap transparency slider.
   - Clinical explanation cards highlighting the diagnostic rationale and triage status (Urgent Audit vs Routine Queue).
4. **Performance Validation Analytics**: Real-time interactive graphs showing training history logs, Confusion Matrix (N=1,000 cases), and ROC Analysis curves.

---

## ⚙️ Core Architecture & Methodology

### Transfer Learning with ResNet-50
ResNet-50 utilizes deep residual learning to prevent vanishing gradients, making it excellent at extracting fine visual textures (like soft-tissue density variations in X-Rays or lesions in MRIs). The model freezes the feature extraction layers and exposes the fully connected layer (`fc`) to learn clinical binary classifications (Normal vs Anomaly).

### The Math Behind Grad-CAM
Grad-CAM computes the gradients of the output score $Y^c$ for class $c$ with respect to the activation maps $A^k$ of the target convolutional layer:

$$\alpha_c^k = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A_{i,j}^k}$$

Where $Z$ is the height $\times$ width of the activation map. The weights $\alpha_c^k$ represent the importance of feature map $k$ for the target class $c$. We then perform a weighted sum of forward activation maps and apply a Rectified Linear Unit (ReLU) to isolate features that positively contribute to the class:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k} \alpha_c^k A^k\right)$$

The resulting 2D heatmap is resized to the original image dimensions, colorized (Jet colormap), and superimposed.

---

## 🚀 Setup & Execution Guide

### Prerequisites
- Python 3.8 or higher
- `git` CLI

### 1. Clone & Set Up Directory
```bash
git clone https://github.com/rushikabijode25015/MediScan-AI.git
cd MediScan-AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Clinical Samples
Generate the mock Chest X-rays and Brain MRIs used by the dashboard for quick testing:
```bash
python generate_samples.py
```
This writes four realistic-looking grayscale images to `static/samples/`.

### 4. Train the Model (Optional)
Run a quick CPU-based training script to fine-tune the classification head on synthetic medical patterns and save the model weights:
```bash
python train.py
```
This generates `model.pth`, saving the state dictionary of the fine-tuned ResNet-50 classification head.

### 5. Launch the Web API & Dashboard
Start the Flask server:
```bash
python app.py
```
By default, the server runs on `http://localhost:5000`. Open this address in your web browser to explore the dashboard!

---

## 📂 Project Structure

```text
MediScan AI/
├── app.py                  # Flask Web Server & API
├── model_helper.py         # PyTorch Model loader & Grad-CAM engine
├── train.py                # Synthetic dataset generator & model training
├── generate_samples.py     # Script to generate realistic mock medical images for testing
├── requirements.txt        # Python dependency list
├── .gitignore              # Git ignore file
├── README.md               # Extensive project documentation
├── static/                 # Frontend assets (served by Flask)
│   ├── css/
│   │   └── style.css       # Premium responsive design CSS
│   ├── js/
│   │   └── main.js         # Frontend interactive logic
│   └── samples/            # Pre-generated sample scans (X-Ray, MRI)
└── templates/              # HTML templates
    └── index.html          # Dashboard HTML
```

---

## 🩺 Clinical Triage Recommendation Protocol

| Predicted Class | Probability Threshold | Triage Status | Recommendation |
| :--- | :--- | :--- | :--- |
| **Normal** | $\ge 90\%$ | Normal Triage | Routine queue review. No immediate action required. |
| **Anomaly Detected** | $\ge 90\%$ | Urgent Audit | High-priority flag. Push to the front of the radiologist audit queue. |
| **Inconclusive** | $< 90\%$ | Standard Review | Standard processing queue. |

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
