# CNN Visualization - 4x4 Matrix Training

This project provides an interactive web-based visualization of a Convolutional Neural Network (CNN) training process using 4x4 matrices. It demonstrates how CNNs process and learn from input data in real-time, making it an excellent educational tool for understanding deep learning concepts.

## Live Demo

Visit [https://mirjunaid26.github.io/cnn_visual/](https://mirjunaid26.github.io/cnn_visual/) to see the visualization in action.

## Project Overview

The visualization shows how a CNN processes 4x4 matrices through its various layers:

1. **Input Layer**: Displays randomly generated 4x4 matrices
2. **Convolutional Layer**: Shows the learned convolutional filters and resulting feature maps
3. **Pooling Layer**: Demonstrates the max pooling operation
4. **Training Metrics**: Real-time plots of loss and accuracy during training

## Features

### Data Generation
- Generates random 4x4 matrices with values between 0 and 1
- Creates binary labels based on matrix sum (1 if sum > 8, 0 otherwise)
- Automatically batches data for training

### CNN Architecture
- Input Shape: [4, 4, 1] (4x4 grayscale images)
- Convolutional Layer: 4 filters of size 2x2 with ReLU activation
- Max Pooling Layer: 2x2 pooling with stride 1
- Dense Layer: 2 units with softmax activation for binary classification

### Visualizations
1. **Input Matrix**
   - Shows the current 4x4 input matrix
   - Updates every few epochs during training

2. **Convolutional Filters**
   - Displays learned 2x2 filters
   - Shows how filters evolve during training

3. **Feature Maps**
   - Visualizes the output after applying convolution
   - Demonstrates how the network "sees" the input

4. **Training Progress**
   - Real-time loss plot
   - Real-time accuracy plot
   - Training status updates

## Technology Stack

- **TensorFlow.js**: For building and training the CNN in the browser
- **Plotly.js**: For interactive data visualization
- **HTML/CSS/JavaScript**: For the web interface
- **GitHub Pages**: For hosting the live demo

## Implementation Details

### JavaScript (web_cnn.js)
- `generateData()`: Creates training data and labels
- `createModel()`: Defines the CNN architecture
- `plotMatrix()`: Handles matrix visualization using Plotly
- `trainModel()`: Manages the training process and updates visualizations

### HTML (index.html)
- Responsive grid layout for visualizations
- Status display for training progress
- Container divs for all plots and visualizations

## Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/mirjunaid26/cnn_visual.git
   ```

2. Navigate to the project directory:
   ```bash
   cd cnn_visual
   ```

3. Serve the files using any HTTP server. For example, using Python:
   ```bash
   python -m http.server
   ```

4. Open your browser and visit `http://localhost:8000`

## Browser Requirements

- Modern web browser with JavaScript enabled
- WebGL support for TensorFlow.js
- Sufficient RAM for training (recommended: 4GB+)

## Understanding the Visualization

1. **Initial State**
   - The page loads with empty visualization boxes
   - TensorFlow.js and Plotly are initialized
   - Status messages appear in the status bar

2. **Training Process**
   - Random 4x4 matrices are generated
   - The CNN processes these through its layers
   - Visualizations update every few epochs
   - Training metrics are plotted in real-time

3. **Interpreting Results**
   - Darker colors in heatmaps indicate higher values
   - Loss should decrease over time
   - Accuracy should increase over time
   - Feature maps show what patterns the CNN detects

## Contributing

Feel free to open issues or submit pull requests for improvements. Some areas for potential enhancement:

- Additional visualization types
- More complex CNN architectures
- Interactive parameter adjustment
- Support for different input sizes
- Performance optimizations

## License

This project is open source and available under the MIT License.

## Acknowledgments

- TensorFlow.js team for the excellent deep learning library
- Plotly team for the visualization tools
- The deep learning community for inspiration and knowledge sharing

## DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14267207.svg)](https://doi.org/10.5281/zenodo.14267207)

## DOI_2" [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14267244.svg)](https://doi.org/10.5281/zenodo.14267244)
