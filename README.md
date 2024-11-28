# CNN Visualization on 4x4 Matrices

This project demonstrates training a Convolutional Neural Network (CNN) on 4x4 matrices with real-time visualization of the training process. You can run it either as a Python script or view it as a web application on GitHub Pages.

## Features
- Simple CNN architecture designed for 4x4 input matrices
- Training visualization including loss and accuracy plots
- Matrix visualization using heatmaps
- Binary classification based on matrix sum
- Real-time visualization of:
  - Input matrices
  - Convolutional filters
  - Feature maps
  - Pooling layer outputs
  - Training and validation metrics

## Web Version
Visit the [GitHub Pages site](https://YOUR_USERNAME.github.io/cnn_visual/) to see the live visualization running in your browser.

## Local Python Version

### Requirements
Install the required packages using:
```bash
pip install -r requirements.txt
```

### Usage
Run the main script:
```bash
python cnn_4x4.py
```

The script will:
1. Generate random 4x4 matrices for training
2. Train the CNN model
3. Display training progress (loss and accuracy)
4. Visualize sample input matrices and predictions in real-time

## Model Architecture
- Input: 4x4 matrix (1 channel)
- Conv2d layer: 4 filters, 2x2 kernel, stride 1, padding 1
- ReLU activation
- MaxPool2d layer: 2x2 kernel, stride 1
- Fully connected layer to 2 output classes

## Repository Structure
```
cnn_visual/
├── README.md
├── requirements.txt
├── cnn_4x4.py          # Local Python version
├── web_cnn.py          # Web version of CNN visualization
└── index.html          # GitHub Pages entry point
```

## Setting up GitHub Pages
1. Create a new repository on GitHub
2. Push this code to the repository:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/cnn_visual.git
git push -u origin main
```
3. Go to repository Settings > Pages
4. Select 'main' branch as source
5. Save and wait for the page to be published

## License
MIT License
