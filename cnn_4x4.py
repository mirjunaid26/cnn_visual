import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('default')

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Input: 1 channel 4x4 matrix
        self.conv1 = nn.Conv2d(1, 4, kernel_size=2, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 2)
        
    def forward(self, x):
        conv_out = self.conv1(x)
        relu_out = self.relu(conv_out)
        pool_out = self.maxpool(relu_out)
        flat_out = self.flatten(pool_out)
        final_out = self.fc(flat_out)
        return final_out, conv_out, pool_out

def generate_data(num_samples=100):
    data = []
    labels = []
    
    for _ in range(num_samples):
        matrix = np.random.rand(4, 4)
        label = 1 if np.sum(matrix) > 8 else 0
        data.append(matrix)
        labels.append(label)
    
    return np.array(data), np.array(labels)

class NetworkVisualizer:
    def __init__(self, model, X_train, y_train, X_val, y_val):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        # Setup the figure
        plt.ion()  # Enable interactive mode
        self.fig = plt.figure(figsize=(15, 8))
        self.setup_subplots()
        
        # Storage for metrics
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        
    def setup_subplots(self):
        gs = self.fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        
        # Input matrix
        self.ax_input = self.fig.add_subplot(gs[0, 0])
        # Convolution filters
        self.ax_filters = self.fig.add_subplot(gs[0, 1])
        # Feature maps
        self.ax_features = self.fig.add_subplot(gs[0, 2])
        # Pooling output
        self.ax_pooling = self.fig.add_subplot(gs[0, 3])
        # Training curves
        self.ax_loss = self.fig.add_subplot(gs[1, :2])
        self.ax_acc = self.fig.add_subplot(gs[1, 2:])
        
    def update(self, frame):
        self.fig.suptitle(f'Training Epoch {frame+1}', fontsize=16)
        
        # Clear all axes
        for ax in [self.ax_input, self.ax_filters, self.ax_features, 
                  self.ax_pooling, self.ax_loss, self.ax_acc]:
            ax.clear()
            
        # Get sample input and its features
        sample_input = self.X_train[0]
        outputs, conv_out, pool_out = self.model(sample_input.unsqueeze(0))
        
        # Plot input matrix
        im1 = self.ax_input.imshow(sample_input.squeeze(), cmap='viridis')
        self.ax_input.set_title('Input Matrix')
        self.fig.colorbar(im1, ax=self.ax_input)
        
        # Plot convolution filters
        filters = self.model.conv1.weight.detach().numpy()
        n_filters = filters.shape[0]
        filter_grid = np.hstack([filters[i, 0] for i in range(n_filters)])
        im2 = self.ax_filters.imshow(filter_grid, cmap='viridis')
        self.ax_filters.set_title('Conv Filters')
        self.fig.colorbar(im2, ax=self.ax_filters)
        
        # Plot feature maps
        feature_maps = conv_out.detach().squeeze().numpy()
        feature_grid = np.hstack([feature_maps[i] for i in range(feature_maps.shape[0])])
        im3 = self.ax_features.imshow(feature_grid, cmap='viridis')
        self.ax_features.set_title('Feature Maps')
        self.fig.colorbar(im3, ax=self.ax_features)
        
        # Plot pooling output
        pool_maps = pool_out.detach().squeeze().numpy()
        pool_grid = np.hstack([pool_maps[i] for i in range(pool_maps.shape[0])])
        im4 = self.ax_pooling.imshow(pool_grid, cmap='viridis')
        self.ax_pooling.set_title('After Pooling')
        self.fig.colorbar(im4, ax=self.ax_pooling)
        
        # Plot training curves
        if len(self.train_losses) > 0:
            self.ax_loss.plot(self.train_losses, 'b-', label='Train Loss')
            self.ax_loss.plot(self.val_losses, 'r-', label='Val Loss')
            self.ax_loss.set_title('Loss Over Time')
            self.ax_loss.set_xlabel('Epoch')
            self.ax_loss.set_ylabel('Loss')
            self.ax_loss.legend()
            self.ax_loss.grid(True)
            
            self.ax_acc.plot(self.train_accs, 'b-', label='Train Acc')
            self.ax_acc.plot(self.val_accs, 'r-', label='Val Acc')
            self.ax_acc.set_title('Accuracy Over Time')
            self.ax_acc.set_xlabel('Epoch')
            self.ax_acc.set_ylabel('Accuracy')
            self.ax_acc.legend()
            self.ax_acc.grid(True)
        
        plt.draw()
        plt.pause(0.1)  # Add small pause to ensure display updates
        
    def update_metrics(self, train_loss, train_acc, val_loss, val_acc):
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)

def train_and_visualize():
    # Generate data
    X_train, y_train = generate_data(100)
    X_val, y_val = generate_data(30)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).unsqueeze(1)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val).unsqueeze(1)
    y_val = torch.LongTensor(y_val)
    
    # Initialize model and training components
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize visualizer
    visualizer = NetworkVisualizer(model, X_train, y_train, X_val, y_val)
    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs, _, _ = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        _, predicted = torch.max(outputs.data, 1)
        train_acc = (predicted == y_train).sum().item() / len(y_train)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs, _, _ = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_acc = (val_predicted == y_val).sum().item() / len(y_val)
        
        # Update visualizer
        visualizer.update_metrics(loss.item(), train_acc, val_loss.item(), val_acc)
        visualizer.update(epoch)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Training - Loss: {loss.item():.4f}, Accuracy: {train_acc:.4f}')
            print(f'Validation - Loss: {val_loss.item():.4f}, Accuracy: {val_acc:.4f}\n')
    
    plt.ioff()  # Disable interactive mode
    plt.show()

if __name__ == "__main__":
    train_and_visualize()
