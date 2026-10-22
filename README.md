# Deep Learning Image Classifier — Cats vs Dogs

A deep learning image classification project built with **PyTorch, MobileNetV2, transfer learning, and Streamlit**. Upload an image to predict whether it contains a cat or a dog.

## Overview

This project uses a pretrained MobileNetV2 convolutional neural network (CNN). The pretrained feature extractor is frozen, and the final classification layer is trained to distinguish between cats and dogs.

The model was trained on the Oxford-IIIT Pet dataset, grouping 12 cat breeds into the `Cat` class and 25 dog breeds into the `Dog` class.

## Features

- Cat vs Dog image classification
- Transfer learning with pretrained MobileNetV2
- Interactive Streamlit web interface
- Prediction confidence display
- Automated unit tests
- Reproducible training script
- CPU-compatible inference

## Tech Stack

- Python
- PyTorch
- Torchvision
- MobileNetV2
- Streamlit
- Pillow
- Unittest

## Dataset

**Oxford-IIIT Pet Dataset**

- Training and validation source: `trainval` split
- Original categories: 37 pet breeds
- Binary classes: Cat and Dog
- Training/validation split: 80% / 20%

The resulting split contains 2,944 training images and 736 validation images.

## Model Performance

| Metric | Result |
|---|---:|
| Architecture | MobileNetV2 |
| Training epochs | 5 |
| Final training loss | 0.0835 |
| Validation accuracy | 98.51% |

*Note: The reported accuracy is measured on a randomly selected validation split. It is not an independent test-set result, and performance may vary on new images.*

## Project Structure

```text
Deep-Learning-Image-Classifier/
├── models/
│   └── cats_dogs_mobilenet_v2.pth
├── src/
│   ├── __init__.py
│   ├── model.py
│   └── predict.py
├── tests/
│   └── test_model.py
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/ahsanhabibabir04/Deep-Learning-Image-Classifier.git
cd Deep-Learning-Image-Classifier
```

### 2. Create and activate a virtual environment

Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Launch the app

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal.

## Run Tests

```bash
python -m unittest discover -s tests -v
```

## Retrain the Model

```bash
python train.py
```

The trained checkpoint is saved to `models/cats_dogs_mobilenet_v2.pth`.

## Limitations

- The classifier predicts only cats or dogs.
- Validation accuracy does not guarantee performance on every image.
- Confidence is the model's predicted probability, not a guarantee of correctness.
- This project is intended for educational and portfolio purposes.

## Author

**M.A. Ahsan Habib Abir**

GitHub: [ahsanhabibabir04](https://github.com/ahsanhabibabir04)

---

*Built as part of an AI/ML portfolio project.*
