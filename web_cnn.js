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
            const matrix = Array(4).fill().map(() => 
                Array(4).fill().map(() => Math.random())
            );
            const sum = matrix.flat().reduce((a, b) => a + b, 0);
            const label = sum > 8 ? 1 : 0;
            
            data.push(matrix);
            labels.push(label);
        }
        
        // Convert to tensors
        const xTensor = tf.tensor4d(data, [numSamples, 4, 4, 1]);
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
            kernelSize: 2,
            padding: 'same',
            activation: 'relu',
            kernelInitializer: 'glorotUniform'
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
            kernelInitializer: 'glorotUniform'
        }));
        
        return model;
    } catch (error) {
        console.error('Error in createModel:', error);
        throw error;
    }
}

// Visualization functions
function plotMatrix(elementId, data, title) {
    try {
        const layout = {
            title: title || '',
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
            colorscale: 'Viridis',
            showscale: false
        };
        
        Plotly.newPlot(elementId, [trace], layout, {displayModeBar: false});
    } catch (error) {
        console.error(`Error plotting matrix for ${elementId}:`, error);
        console.error('Data:', data);
        throw error;
    }
}

function plotTrainingProgress() {
    try {
        // Loss plot
        const lossLayout = {
            title: 'Training and Validation Loss',
            xaxis: { title: 'Epoch' },
            yaxis: { title: 'Loss' },
            height: 300,
            margin: { t: 30, b: 50, l: 50, r: 30 }
        };
        
        Plotly.newPlot('loss-plot', [
            { y: trainLosses, name: 'Training Loss', mode: 'lines', line: { color: 'blue' } },
            { y: valLosses, name: 'Validation Loss', mode: 'lines', line: { color: 'red' } }
        ], lossLayout);
        
        // Accuracy plot
        const accLayout = {
            title: 'Training and Validation Accuracy',
            xaxis: { title: 'Epoch' },
            yaxis: { title: 'Accuracy' },
            height: 300,
            margin: { t: 30, b: 50, l: 50, r: 30 }
        };
        
        Plotly.newPlot('accuracy-plot', [
            { y: trainAccs, name: 'Training Accuracy', mode: 'lines', line: { color: 'blue' } },
            { y: valAccs, name: 'Validation Accuracy', mode: 'lines', line: { color: 'red' } }
        ], accLayout);
    } catch (error) {
        console.error('Error in plotTrainingProgress:', error);
        throw error;
    }
}

// Training function
async function trainModel() {
    const statusElement = document.getElementById('status');
    statusElement.textContent = 'Generating training data...';
    
    try {
        console.log('Starting training process...');
        
        // Generate data
        const [X_train, y_train] = generateData(100);
        console.log('Training data generated:', {
            xShape: X_train.shape,
            yShape: y_train.shape
        });
        
        const [X_val, y_val] = generateData(30);
        console.log('Validation data generated');
        
        // Create and compile model
        console.log('Creating model...');
        const model = createModel();
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        console.log('Model compiled');
        
        // Initial visualizations
        console.log('Setting up initial visualizations...');
        const initialMatrix = X_train.slice([0], [1]).squeeze().arraySync();
        plotMatrix('input-matrix', initialMatrix, 'Input Matrix');
        
        // Training loop
        const numEpochs = 50;
        statusElement.textContent = 'Training...';
        
        for (let epoch = 0; epoch < numEpochs; epoch++) {
            console.log(`Starting epoch ${epoch + 1}/${numEpochs}`);
            
            // Train on batch
            const history = await model.fit(X_train, y_train, {
                epochs: 1,
                validationData: [X_val, y_val],
                verbose: 1
            });
            
            // Update metrics
            trainLosses.push(history.history.loss[0]);
            trainAccs.push(history.history.acc[0]);
            valLosses.push(history.history.val_loss[0]);
            valAccs.push(history.history.val_acc[0]);
            
            // Update visualizations every few epochs
            if (epoch % 2 === 0) {
                tf.tidy(() => {
                    // Get sample input
                    const sample = X_train.slice([0], [1]);
                    
                    // Get conv layer weights
                    const conv = model.layers[0];
                    const convWeights = conv.getWeights()[0].squeeze().arraySync();
                    plotMatrix('conv-filters', convWeights[0], 'Convolution Filter');
                    
                    // Get feature maps
                    const convOutput = tf.model({
                        inputs: model.input,
                        outputs: conv.output
                    }).predict(sample);
                    plotMatrix('feature-maps', convOutput.squeeze().arraySync()[0], 'Feature Map');
                    
                    // Get pooling output
                    const pooling = model.layers[1];
                    const poolOutput = tf.model({
                        inputs: model.input,
                        outputs: pooling.output
                    }).predict(sample);
                    plotMatrix('pooling-output', poolOutput.squeeze().arraySync()[0], 'Pooling Output');
                    
                    // Update training plots
                    plotTrainingProgress();
                });
            }
            
            // Update status
            statusElement.textContent = `Training... Epoch ${epoch + 1}/${numEpochs}`;
            await tf.nextFrame();
        }
        
        statusElement.textContent = 'Training complete!';
        console.log('Training completed successfully');
        
    } catch (error) {
        console.error('Detailed error in training:', error);
        statusElement.textContent = `Error: ${error.message}. Check console for details.`;
        throw error;
    } finally {
        // Clean up tensors
        tf.disposeVariables();
    }
}

// Start training when page loads
window.addEventListener('load', () => {
    console.log('Page loaded, checking TensorFlow.js...');
    if (typeof tf === 'undefined') {
        console.error('TensorFlow.js not loaded!');
        document.getElementById('status').textContent = 'Error: TensorFlow.js not loaded';
        return;
    }
    
    console.log('TensorFlow.js version:', tf.version.tfjs);
    tf.ready().then(() => {
        console.log('TensorFlow.js initialized, starting training...');
        trainModel().catch(error => {
            console.error('Training failed:', error);
        });
    });
});
