# 🌿 Plant Disease Detection

A deep learning web application that identifies plant leaf diseases from a photo. Upload an image of a leaf, and the app returns the most likely disease along with a confidence score and the top-3 possible predictions.

---

## Overview

This project uses a trained deep learning model to look at a photograph of a plant leaf and predict which disease (if any) it shows. Think of it like a first-pass visual check: you take a picture, upload it, and within seconds you get:

- The **predicted disease name**
- A **confidence score** (how sure the model is)
- The **top-3 most likely diseases**, in case the top guess is wrong

Under the hood, the "guessing" is done by a convolutional neural network (a type of deep learning model built for recognizing patterns in images) that has been trained on thousands of labeled leaf photos across 112 different disease categories.

This is **not a diagnostic tool for farmers or agronomists to rely on** — it's a demonstration of how a modern image classification model can be trained, evaluated honestly, and shipped as a working web application, end to end.

---

## Why This Project?

Manually identifying plant diseases usually requires trained expertise, and that expertise isn't always available quickly or cheaply — especially for hobbyist growers, students, or anyone without easy access to an agricultural specialist. An automated first-pass check can be a useful starting point for narrowing down what might be wrong with a plant.

This project **does not** claim to replace plant pathologists, agronomists, or professional diagnostic services. It's an applied machine learning project that explores how far an image classifier can get on a hard, imbalanced, real-world dataset — and it's transparent about where the model struggles.

---

## How It Works

```
Leaf Image Upload
      │
      ▼
Image Preprocessing (resize, normalize)
      │
      ▼
EfficientNet-B0 Model (trained, fine-tuned)
      │
      ▼
Disease Classification (112 possible classes)
      │
      ▼
Confidence Calculation (softmax probability)
      │
      ▼
Top-3 Predictions Selected
      │
      ▼
Result Displayed in Web Interface
```

1. **Upload** — the user selects/uploads a leaf photo through the web page.
2. **Preprocessing** — the image is resized and normalized to match what the model expects.
3. **Model inference** — the EfficientNet-B0 model processes the image and outputs a probability for each of the 112 disease classes.
4. **Classification** — the class with the highest probability becomes the main prediction.
5. **Confidence** — that probability is shown as a percentage.
6. **Top-3** — the three highest-probability classes are shown, in case the correct answer isn't the top guess.
7. **Display** — the frontend renders the result, including a low-confidence warning when the model isn't very sure.

---

## Demo / Application

The web interface lets a user:

- Upload a leaf image from their device
- View the predicted disease name
- View the confidence percentage
- View the top-3 predictions
- See a simple visualization of confidence
- See a warning if the model's confidence is low

There is no live hosted demo — the app is designed to be run locally (see [Running the Application](#running-the-application)).

**[Add application screenshot here]**

---

## Key Features

- 🧠 Deep learning–based image classification
- 🔁 Transfer learning using a pretrained EfficientNet-B0 backbone
- 🎯 Fine-tuning on the target dataset with class-weighted loss to address imbalance
- 🌱 112 plant disease classes
- 🏆 Top-3 prediction output, not just a single guess
- 📊 Confidence scores for every prediction
- ⚡ FastAPI backend for serving predictions
- 🖥️ Simple HTML/CSS/JS web frontend
- 📈 Full model evaluation: classification report, per-class metrics, confusion matrix
- 🔍 Confusion-pair analysis to understand what the model mixes up
- 🔦 Grad-CAM visualizations for interpretability
- 🕵️ An OCR-based audit checking whether the model could be relying on embedded text as a shortcut
- 💻 Automatic CPU/GPU device selection during training and inference

---

## Dataset

- **112 disease classes** are used in the final trained model.
- The **test set contains 1,148 images**.
- The dataset is **significantly imbalanced** — some classes have far more images than others, and a number of classes have very few test samples, which makes their individual metrics less statistically reliable.
- During data cleaning, a minimum-image-per-class threshold was applied, and a small number of classes with too few images were excluded from training entirely rather than kept with insufficient data.
- The data was split into **train / validation / test sets on a per-class basis** (each class contributes its own train/validation/test images), which helps keep the class distribution consistent across splits.
- The raw dataset image files are **not included in this repository**. Only the trained model checkpoint and metadata (e.g. `data/classes.txt`) are included.

⚠️ Because of the class imbalance and the sheer number of visually similar diseases, this model should **not** be assumed to generalize well to every plant or disease in the real world — see [Limitations](#limitations).

---

## Deep Learning Approach

**Why a CNN?** Convolutional neural networks are well suited to image tasks because they learn to recognize visual patterns — edges, textures, spots, discoloration — directly from pixel data, without those patterns needing to be manually defined.

**EfficientNet-B0** is a convolutional neural network architecture designed to get strong accuracy while staying relatively lightweight and efficient compared to larger, older architectures.

**Transfer learning** means starting from a model that has already learned general visual features (from a large, unrelated image dataset) instead of training from scratch. This dramatically reduces the amount of data and time needed to get a useful model.

The project follows a two-stage transfer learning approach:

1. **Feature extraction** — the pretrained EfficientNet-B0 backbone is frozen, and only a new classification head (sized for 112 classes) is trained. This is a fast, low-risk way to see how well the pretrained features transfer to this dataset.
2. **Fine-tuning** — more of the network's layers are unfrozen and trained at a lower learning rate, allowing the model to adapt its internal features more specifically to plant disease imagery. This stage also uses **class-weighted loss** to reduce the impact of class imbalance.

**Preprocessing/transforms** applied to every image include resizing to the model's expected input size and normalizing pixel values using standard ImageNet mean/std statistics (since the pretrained backbone was originally trained on ImageNet-normalized images).

**Softmax probabilities**: the model's final layer outputs a probability distribution over all 112 classes. The highest probability becomes the main prediction, and the confidence score shown to the user is simply that probability.

**Top-3 predictions** are generated by taking the three classes with the highest softmax probabilities, rather than only the single best guess.

---

## Model Training

The training pipeline (see `notebook/dataset_training_edited.ipynb`) covers:

1. **Data auditing & cleaning** — walking the image directory, checking for missing/corrupted files and duplicates, and excluding classes with too few images.
2. **Train/validation/test splitting** — done per class to preserve class balance across splits, with the resulting CSVs saved for reproducibility.
3. **Dataloaders** — a custom PyTorch `Dataset`/`DataLoader` pipeline handles reading images, applying transforms, and batching.
4. **Baseline model** — a simple CNN trained from scratch first, to establish a reference point before using transfer learning.
5. **Feature extraction** — the pretrained EfficientNet-B0 backbone (frozen) with a new classification head, trained for several epochs.
6. **Class imbalance handling** — class weights computed from the training label distribution and used in a weighted cross-entropy loss.
7. **Fine-tuning** — a further round of training with more layers unfrozen, using the weighted loss.
8. **Extended fine-tuning** — additional training with a learning-rate scheduler (reducing the learning rate when validation loss plateaus) and early stopping (stopping training if validation performance stops improving).
9. **Checkpointing** — the best-performing model (by validation macro-F1) is saved to `model/efficientnet_finetuned_best.pth` during training.
10. **Evaluation** — the saved checkpoint is reloaded and evaluated on the held-out test set.

---

## Results

Final test-set performance of the fine-tuned EfficientNet-B0 model:

| Metric | Score |
|---|---|
| Test Accuracy | 60.54% |
| Macro Precision | 56.97% |
| Macro Recall | 58.32% |
| Macro F1-Score | 55.43% |

**What these mean, in plain terms:**

- **Accuracy** — out of all test images, the percentage the model classified correctly.
- **Precision (macro)** — of the images the model predicted as a given class, how often it was actually right, averaged equally across all 112 classes (regardless of how many test images each class had).
- **Recall (macro)** — of all the images that truly belong to a given class, how many the model correctly found, again averaged equally across all classes.
- **F1-Score (macro)** — a balance between precision and recall, again averaged equally across all classes.

**Why macro-F1 matters here:** with 112 imbalanced classes, plain accuracy can look reasonable while the model is still performing poorly on rare classes — a model that's great at the 10 most common diseases and bad at the rest could still post a decent overall accuracy. Macro-averaged metrics treat every class equally, which gives a more honest picture of how the model performs across the *entire* set of diseases, not just the most common ones.

---

## Error Analysis

Performance is not uniform across the 112 classes. Several factors contribute to this:

- **Class imbalance** — classes with more training images tend to be learned better than classes with very few.
- **Low-support classes** — classes with very few test images produce metrics that can swing heavily on just one or two predictions, making them less statistically stable.
- **Visually similar diseases** — some diseases look alike, especially across closely related crops or diseases with overlapping symptoms (leaf spots, mildews, rusts, blights).
- **Cross-species similarity** — some confusions occur between the *same disease type* on different plants (e.g. two different powdery mildews), suggesting the model has partly learned the disease *pattern* rather than the plant *species*.

A confusion matrix and a ranked list of the most frequent misclassification pairs were generated. Some of the most notable confusions include:

| True Class | Frequently Confused With |
|---|---|
| Soybean frog eye leaf spot | Soybean downy mildew |
| Squash powdery mildew | Cucumber powdery mildew |
| Wheat septoria blotch | Wheat stripe rust |
| Wheat head scab | Wheat bacterial leaf streak (black chaff) |
| Squash powdery mildew | Zucchini powdery mildew |
| Corn northern leaf blight | Corn gray leaf spot |
| Cucumber bacterial wilt | Zucchini bacterial wilt |
| Bean rust | Soybean rust |
| Tomato septoria leaf spot | Tomato bacterial leaf spot |
| Wheat stripe rust | Wheat bacterial leaf streak (black chaff) |

These confusions make intuitive sense — many of these pairs share very similar visual symptoms (spotting patterns, powdery textures, rust-colored lesions) and, in several cases, are the same disease category affecting related crops.

---

## Model Interpretability

**Grad-CAM** (Gradient-weighted Class Activation Mapping) is a technique that highlights which regions of an input image most influenced the model's prediction. In simple terms: it produces a heatmap over the leaf image showing *where the model was "looking"* when it made its decision.

Grad-CAM was applied to both correctly and incorrectly classified test images to visually inspect whether the model was focusing on relevant parts of the leaf (e.g. lesions, spots, discoloration) rather than irrelevant background details.

⚠️ Grad-CAM visualizations are a diagnostic aid, not proof of correctness. A heatmap that looks reasonable does not guarantee the underlying prediction is right, and a wrong prediction can still have a "plausible-looking" heatmap.

---

## Model Audit / OCR Experiment

One risk in image classification datasets is that a model can accidentally learn to rely on **shortcuts** unrelated to the actual subject — for example, embedded text, watermarks, or labels present in certain classes of images.

To check for one specific version of this risk, an OCR (optical character recognition) pass was run across **all 1,148 test images** to detect any visible text.

**Result:** OCR detected text in **0 of the 1,148 test images**.

**What this means:** this experiment did not find evidence that the model could be exploiting visible embedded text as a shortcut for classification, since no such text was detected in the test set.

**What this does not mean:** this is a narrow, single-signal audit. It does **not** rule out other kinds of shortcuts or biases the model might be relying on (such as background patterns, image source artifacts, or lighting conditions correlated with a particular class). It should be read as one small, honest data point — not proof that the model is free of shortcut learning.

---

## Technology Stack

**Machine Learning**
- Python
- PyTorch
- EfficientNet-B0 architecture (used via transfer learning)

**Data / Evaluation**
- NumPy
- Pandas
- Scikit-learn (metrics, class-weight computation)
- Matplotlib
- Pillow (PIL)
- Pytesseract (OCR audit)

**Backend**
- FastAPI
- Uvicorn

**Frontend**
- HTML
- CSS
- JavaScript

**Development**
- Jupyter Notebook
- Git / GitHub

---

## Project Structure

```
Plant disease detection_1/
│
├── app/
│   ├── main.py              # FastAPI application (routes, prediction endpoint)
│   └── static/
│       ├── index.html       # Web page for image upload and results
│       ├── style.css        # Styling for the frontend
│       └── script.js        # Handles upload, API calls, and rendering results
│
├── data/
│   ├── classes.txt          # List of the 112 disease class names
│   └── evaluation/          # Saved evaluation results (reports, metrics, etc.)
│
├── model/
│   └── efficientnet_finetuned_best.pth   # Final trained model checkpoint
│
├── notebook/
│   └── dataset_training_edited.ipynb     # Full data prep, training, and evaluation pipeline
│
├── src/
│   ├── models/               # Model architectures (baseline CNN, EfficientNet, Grad-CAM)
│   ├── preprocessing/        # Transforms and dataloader creation
│   └── predict.py            # Inference wrapper used by the API
│
├── .gitignore
├── requirements.txt
├── run_server.py             # Entry point to launch the FastAPI server
└── README.md
```

> Note: this structure reflects the project layout as described by the author. If your local repository differs, treat this as a guide rather than a guarantee.

---

## Installation

```bash
git clone <repository-url>

cd plant-disease-detection

python -m venv venv
```

**Activate the virtual environment (Windows PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

> The raw training dataset is **not included** in this repository — only the trained model checkpoint and supporting metadata are provided.

---

## Running the Application

```bash
python run_server.py
```

Then open your browser to:

```
http://127.0.0.1:8000
```

---

## API Documentation

### `GET /`
Serves the web frontend (the upload page).

### `POST /predict`
Accepts an uploaded leaf image and returns the model's prediction.

- **Method:** `POST`
- **Endpoint:** `/predict`
- **Input:** an image file, sent as multipart form data
- **Output:** a JSON response containing the predicted class, confidence score, and top-3 predictions

**Example response (illustrative — see `app/main.py` for the exact response schema):**

```json
{
  "prediction": "blueberry anthracnose",
  "confidence": 0.9112,
  "top_3": [
    { "class": "blueberry anthracnose", "confidence": 0.9112 },
    { "class": "blueberry botrytis blight", "confidence": 0.0421 },
    { "class": "blueberry rust", "confidence": 0.0187 }
  ]
}
```

---

## Example Prediction

*(Illustrative example, based on the underlying prediction interface — not necessarily an exact captured output.)*

**Input:** a photo of a blueberry leaf with dark, sunken lesions
**Output:**
- Prediction: `blueberry anthracnose`
- Confidence: `91.1%`
- Top-3: `blueberry anthracnose`, `blueberry botrytis blight`, `blueberry rust`

---

## Reproducing Training

The full training pipeline — data auditing, cleaning, splitting, baseline training, feature extraction, fine-tuning, and evaluation — is documented step by step in:

```
notebook/dataset_training_edited.ipynb
```

Running this notebook end to end (with the raw dataset present locally) reproduces the same overall workflow used to train the checkpoint included in this repository. Exact numeric reproducibility is **not guaranteed**, since neural network training involves some inherent randomness (e.g. weight initialization, data shuffling) that isn't fully fixed across every step of the pipeline.

---

## Limitations

- This is a **112-class classification problem** — a genuinely hard task, especially with limited data per class.
- The dataset is **imbalanced**, and several classes have very few images, making the model's performance on those classes less reliable and harder to trust.
- Some diseases are **visually very similar** to one another (see [Error Analysis](#error-analysis)), which limits how confidently the model can separate them.
- Overall test accuracy is around **60.54%** — solid for a 112-class imbalanced problem, but far from perfect, and performance varies significantly by class.
- The training images may not fully represent the **variety of real-world conditions** (lighting, camera quality, plant growth stage, backgrounds) a user's photo might have.
- A **high confidence score does not guarantee a correct prediction** — the model can be confidently wrong, particularly on classes it has seen little data for.
- This tool is an **educational/portfolio project**, not a substitute for professional agricultural or plant pathology diagnosis.

---

## Future Improvements

*(Listed as potential future work — none of the following are currently implemented.)*

- Collecting more, and more balanced, training data — especially for low-support classes
- Stronger and more varied data augmentation
- Techniques specifically aimed at improving rare-class performance (e.g. oversampling, focal loss)
- Experimenting with other pretrained architectures for comparison
- More systematic hyperparameter tuning
- Calibrating confidence scores so they better reflect true prediction reliability
- Broader real-world validation using images outside the training distribution
- Deployment optimization (e.g. containerization, cloud hosting)
- Deeper explainability work beyond Grad-CAM

---

## What I Learned

- Practical image preprocessing and dataset auditing for a real, messy image dataset
- Building and training CNNs and transfer-learning pipelines in PyTorch
- Fine-tuning pretrained models and managing frozen vs. trainable layers
- Handling class imbalance in a multi-class setting using class-weighted loss
- Choosing and interpreting evaluation metrics beyond plain accuracy (macro precision/recall/F1)
- Reading and acting on a confusion matrix and per-class error analysis
- Applying Grad-CAM for basic model interpretability
- Auditing a model for a specific type of shortcut learning (embedded text via OCR)
- Serving a trained PyTorch model through a FastAPI backend
- Connecting a simple frontend to a backend inference API end to end

---

## Disclaimer

This project is an **educational and research demonstration**. Predictions produced by this application should **not** be treated as a professional agricultural or plant pathology diagnosis. For real plant health concerns, consult a qualified agricultural expert.

---

## Author

**Author:** Hitexa
**GitHub:** [Add GitHub profile link]
**LinkedIn:** [Add LinkedIn profile link]
