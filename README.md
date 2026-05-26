# Face Mask Detection

A deep learning project that detects whether a person is wearing a face mask or not, using a CNN model built with Keras/TensorFlow.

## Project Structure

```
face-mask-/
├── data/                  # Dataset (with_mask / without_mask)
├── train.py               # Model training script
├── app.py                 # Real-time detection app
├── mask_detector.h5       # Trained model
└── accuracy_plot.png      # Training accuracy/loss graph
```

## How to Run

**1. Install dependencies**
```bash
pip install tensorflow keras opencv-python numpy
```

**2. Train the model**
```bash
python train.py
```

**3. Run the app**
```bash
python app.py
```

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
