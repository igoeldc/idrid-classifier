# 🧠 Retinal Disease Classification with CNN (IDRiD Dataset)

This project implements a convolutional neural network (CNN) in **PyTorch** to classify retinal fundus images from the [IDRiD dataset](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) based on diabetic retinopathy grades. It includes data preprocessing, model training, evaluation, and visualization.

---

## 📂 Project Structure
```
.
├── datasets/
│   └── B-Disease_Grading/
│       ├── 1-Original_Images/
│       │   ├── a-Train_Set/
│       │   └── b-Test_Set/
│       └── 2-Groundtruths/
│           ├── a-Train_Labels.csv
│           └── b-Test_Labels.csv
├── notifyme.py
├── model_training.ipynb
├── README.md
```

---

## 🧪 Dataset

- **Source:** [IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)
- **Task:** Predict the `Retinopathy grade` for each image
- **Classes:** 0 (No DR) to 4 (Proliferative DR)
- **Format:** Images in `.jpg`, labels in `.csv`

---

## 🧰 Features

- Custom PyTorch `Dataset` class
- Data augmentation using `torchvision.transforms`
- Automatic dataset normalization (mean and std computed per channel)
- CNN model definition, training, validation, and testing pipeline
- Visualizations: loss/accuracy plots, confusion matrix
- Mac desktop notifications (`notifyme.py`) for training/testing completion
- Supports Apple Silicon (`mps` backend)

---

## 🚀 Model Architecture

- 2 convolutional layers with ReLU and pooling
- Flattened and passed through fully connected layers with dropout
- `CrossEntropyLoss` and `Adam` optimizer (`lr=0.00025`, `weight_decay=1e-4`)
- Trained for **25 epochs** on ~413 labeled images

---

## 📊 Sample Results

- **Overall Accuracy:** 37%
- **Best Performance:** Class 0 (Precision 0.55, Recall 0.50)
- **Worst Performance:** Class 1 (Precision 0.00, Recall 0.00)

The confusion matrix and classification report are available in the notebook output.

---

## 💡 Suggested Improvements

- Use pretrained models (e.g. ResNet, EfficientNet) for better feature extraction
- Apply class balancing techniques (oversampling, weighted loss)
- Visual error analysis with Grad-CAM or saliency maps
- Expand the dataset with external augmentation or synthetic data

---

## 🛠 Requirements

Install the necessary packages:

```bash
pip install torch torchvision pandas matplotlib seaborn scikit-learn tqdm pillow
```
