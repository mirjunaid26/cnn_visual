// Initialize arrays for storing training history
let trainLosses = [];
let trainAccs = [];
let valLosses = [];
let valAccs = [];

// Helper function to generate random 4x4 matrices
function generateData(numSamples) {
    try {
        const data = [];
        const labels = [];
        
        for (let i = 0; i < numSamples; i++) {
            // Create a 4x4 matrix with random values between 0 and 1
            const matrix = Array.from({ length: 4 }, () => 
                Array.from({ length: 4 }, () => Math.random())
            );
            
            // Calculate sum and create label (1 if sum > 8, 0 otherwise)
            const sum = matrix.flat().reduce((a, b) => a + b, 0);
            const label = sum > 8 ? 1 : 0;
            
            data.push(matrix);
            labels.push(label);
        }
        
        // Convert to tensors with proper shapes
        const xTensor = tf.tensor4d(data.map(m => m.map(row => row.map(val => [val]))));
        const yTensor = tf.oneHot(tf.tensor1d(labels, 'int32'), 2);
        
        return [xTensor, yTensor];
    } catch (error) {
        console.error('Error in generateData:', error);
        throw error;
    }
}

// Create CNN model
function createModel() {
    try {
        const model = tf.sequential();
        
        // Add convolutional layer
        model.add(tf.layers.conv2d({
            inputShape: [4, 4, 1],
            filters: 4,
            kernelSize: [2, 2],
            padding: 'same',
            activation: 'relu',
            kernelInitializer: 'glorotNormal'
        }));
        
        // Add max pooling layer
        model.add(tf.layers.maxPooling2d({
            poolSize: [2, 2],
            strides: [1, 1]
        }));
        
        // Flatten the output
        model.add(tf.layers.flatten());
        
        // Add dense layer for classification
        model.add(tf.layers.dense({
            units: 2,
            activation: 'softmax',
            kernelInitializer: 'glorotNormal'
        }));

        // Compile the model
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        return model;
    } catch (error) {
        console.error('Error in createModel:', error);
        throw error;
    }
}

// Visualization functions
function plotMatrix(elementId, data, title = '') {
    try {
        const layout = {
            title: title,
            height: 250,
            width: 250,
            margin: { t: 30, b: 30, l: 30, r: 30 },
            xaxis: {
                showticklabels: false,
                showgrid: false,
                zeroline: false
            },
            yaxis: {
                showticklabels: false,
                showgrid: false,
                zeroline: false
            }
        };
        
        const trace = {
            z: data,
            type: 'heatmap',
            colorscale: 'Viridis'
        };
        
        Plotly.newPlot(elementId, [trace], layout);
    } catch (error) {
        console.error(`Error plotting matrix for ${elementId}:`, error);
    }
}

function plotTrainingProgress() {
    try {
        // Plot loss
        const lossTrace = {
            y: trainLosses,
            type: 'scatter',
            mode: 'lines',
            name: 'Training Loss'
        };
        
        const lossLayout = {
            title: 'Training Loss',
            xaxis: { title: 'Epoch' },
            yaxis: { title: 'Loss' }
        };
        
        Plotly.newPlot('loss-plot', [lossTrace], lossLayout);
        
        // Plot accuracy
        const accTrace = {
            y: trainAccs,
            type: 'scatter',
            mode: 'lines',
            name: 'Training Accuracy'
        };
        
        const accLayout = {
            title: 'Training Accuracy',
            xaxis: { title: 'Epoch' },
            yaxis: { title: 'Accuracy' }
        };
        
        Plotly.newPlot('accuracy-plot', [accTrace], accLayout);
    } catch (error) {
        console.error('Error plotting training progress:', error);
    }
}

// Training function
async function trainModel() {
    try {
        const status = document.getElementById('status');
        status.textContent = 'Generating training data...';
        
        // Generate training data
        const [xTrain, yTrain] = generateData(100);
        const [xVal, yVal] = generateData(20);
        
        status.textContent = 'Creating model...';
        const model = createModel();
        
        // Display model summary
        model.summary();
        
        status.textContent = 'Training model...';
        
        // Train the model
        await model.fit(xTrain, yTrain, {
            epochs: 50,
            validationData: [xVal, yVal],
            batchSize: 32,
            shuffle: true,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    trainLosses.push(logs.loss);
                    trainAccs.push(logs.acc);
                    
                    if (logs.val_loss) valLosses.push(logs.val_loss);
                    if (logs.val_acc) valAccs.push(logs.val_acc);
                    
                    status.textContent = `Training... Epoch ${epoch + 1}/50`;
                    plotTrainingProgress();
                    
                    // Visualize current state
                    if (epoch % 5 === 0) {
                        // Get sample input
                        const sampleInput = xTrain.slice([0, 0, 0, 0], [1, 4, 4, 1]);
                        
                        // Get conv layer output
                        const convLayer = model.layers[0];
                        const convOutput = tf.tidy(() => {
                            const layerOutput = convLayer.apply(sampleInput);
                            return layerOutput.squeeze();
                        });
                        
                        // Plot visualizations
                        const inputMatrix = await sampleInput.squeeze().array();
                        plotMatrix('input-matrix', inputMatrix, 'Input');
                        
                        const convWeights = await convLayer.getWeights()[0].squeeze().array();
                        plotMatrix('conv-filters', convWeights[0], 'Conv Filter 1');
                        
                        const convOutputArr = await convOutput.array();
                        plotMatrix('feature-maps', convOutputArr[0], 'Feature Map 1');
                        
                        // Clean up tensors
                        convOutput.dispose();
                    }
                    
                    await tf.nextFrame();
                }
            }
        });
        
        status.textContent = 'Training complete!';
        
    } catch (error) {
        console.error('Error during training:', error);
        status.textContent = 'Error during training: ' + error.message;
    }
}

// Start training when page loads
window.addEventListener('load', async () => {
    console.log('Page loaded, checking TensorFlow.js...');
    try {
        if (typeof tf === 'undefined') {
            throw new Error('TensorFlow.js not loaded');
        }
        console.log('TensorFlow.js version:', tf.version.tfjs);
        
        if (typeof Plotly === 'undefined') {
            throw new Error('Plotly not loaded');
        }
        console.log('Plotly loaded successfully');
        
        await trainModel();
    } catch (error) {
        console.error('Initialization error:', error);
        document.getElementById('status').textContent = 'Error: ' + error.message;
    }
});
