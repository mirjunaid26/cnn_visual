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
Visit the [GitHub Pages site](https://mirjunaid26.github.io/cnn_visual/) to see the live visualization running in your browser.

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

## Web Version (TensorFlow.js)

The web version uses TensorFlow.js to run the CNN directly in your browser. This version provides real-time visualization of the training process.

### Running the Web Version

You can run the web version in two ways:

1. **Using Python's HTTP Server**:
   ```bash
   # Navigate to the project directory
   cd /path/to/cnn_visual
   
   # Start the server
   python -m http.server 8000
   ```
   Then open your browser and visit: `http://localhost:8000`

2. **Using GitHub Pages**:
   - Push the code to a GitHub repository
   - Enable GitHub Pages in your repository settings
   - Visit `https://<your-username>.github.io/<repository-name>`

### Features
- Real-time visualization of:
  - Input 4x4 matrices
  - Convolutional filters
  - Feature maps after convolution
  - Pooling layer outputs
  - Training and validation loss curves
  - Accuracy metrics

### Technical Details
- Built with TensorFlow.js for neural network operations
- Uses Plotly.js for interactive visualizations
- Runs entirely in the browser (no backend required)
- Responsive design that works on various screen sizes

## Files
- `index.html`: Main web interface
- `web_cnn.js`: TensorFlow.js implementation of the CNN and visualizations
- `README.md`: This documentation
- `.gitignore`: Git ignore file for the project

## Browser Compatibility
The web version has been tested and works on:
- Chrome (recommended)
- Firefox
- Safari

## Contributing
Feel free to open issues or submit pull requests for improvements!

## License
MIT License
