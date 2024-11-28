// Initialize arrays for storing training history
let trainLosses = [];
let trainAccs = [];
let valLosses = [];
let valAccs = [];

// Helper function to generate random 4x4 matrices
function generateData(numSamples) {
    const data = [];
    const labels = [];
    
    for (let i = 0; i < numSamples; i++) {
        const matrix = tf.randomUniform([4, 4]).arraySync();
        const sum = matrix.flat().reduce((a, b) => a + b, 0);
        const label = sum > 8 ? 1 : 0;
        
        data.push(matrix);
        labels.push(label);
    }
    
    return [tf.tensor4d(data, [numSamples, 4, 4, 1]), tf.oneHot(labels, 2)];
}

// Create CNN model
function createModel() {
    const model = tf.sequential();
    
    model.add(tf.layers.conv2d({
        inputShape: [4, 4, 1],
        filters: 4,
        kernelSize: 2,
        padding: 'same',
        activation: 'relu'
    }));
    
    model.add(tf.layers.maxPooling2d({
        poolSize: 2,
        strides: 1
    }));
    
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 2, activation: 'softmax' }));
    
    return model;
}

// Visualization functions
function plotMatrix(elementId, data, title) {
    const layout = {
        title: title,
        height: 200,
        margin: { t: 30, b: 30, l: 30, r: 30 }
    };
    
    Plotly.newPlot(elementId, [{
        z: data,
        type: 'heatmap',
        colorscale: 'Viridis'
    }], layout);
}

function plotTrainingProgress() {
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
}

// Training function
async function trainModel() {
    const statusElement = document.getElementById('status');
    statusElement.textContent = 'Generating training data...';
    
    // Generate data
    const [X_train, y_train] = generateData(100);
    const [X_val, y_val] = generateData(30);
    
    statusElement.textContent = 'Creating model...';
    const model = createModel();
    
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
    
    // Initial visualizations
    plotMatrix('input-matrix', X_train.slice([0], [1]).squeeze().arraySync());
    
    // Training loop
    const numEpochs = 50;
    statusElement.textContent = 'Training...';
    
    for (let epoch = 0; epoch < numEpochs; epoch++) {
        // Train on batch
        const history = await model.fit(X_train, y_train, {
            epochs: 1,
            validationData: [X_val, y_val],
            verbose: 0
        });
        
        // Update metrics
        trainLosses.push(history.history.loss[0]);
        trainAccs.push(history.history.acc[0]);
        valLosses.push(history.history.val_loss[0]);
        valAccs.push(history.history.val_acc[0]);
        
        // Update visualizations
        if (epoch % 2 === 0) {
            // Get intermediate activations
            const sample = X_train.slice([0], [1]);
            
            // Get conv layer weights and activations
            const conv = model.layers[0];
            const convWeights = conv.getWeights()[0].squeeze().arraySync();
            
            // Visualize filters and feature maps
            plotMatrix('conv-filters', convWeights[0]);
            
            const features = tf.tidy(() => {
                const activation = tf.model({
                    inputs: model.input,
                    outputs: conv.output
                });
                return activation.predict(sample);
            });
            
            plotMatrix('feature-maps', features.squeeze().arraySync()[0]);
            
            // Get pooling layer output
            const pooling = model.layers[1];
            const poolingModel = tf.model({
                inputs: model.input,
                outputs: pooling.output
            });
            const poolOutput = poolingModel.predict(sample);
            plotMatrix('pooling-output', poolOutput.squeeze().arraySync()[0]);
            
            // Update training plots
            plotTrainingProgress();
            
            // Update status
            statusElement.textContent = `Training... Epoch ${epoch + 1}/${numEpochs}`;
            
            // Clean up tensors
            features.dispose();
            poolOutput.dispose();
        }
        
        // Allow UI to update
        await tf.nextFrame();
    }
    
    statusElement.textContent = 'Training complete!';
}

// Start training when page loads
window.addEventListener('load', () => {
    tf.ready().then(() => {
        trainModel();
    });
});
