# MNIST Digit Classification with PyTorch

This project implements a convolutional neural network (CNN) for classifying handwritten digits from the MNIST dataset. The model achieves >99.4% test accuracy while keeping the parameter count under 8k.

## Project Structure

- `src/model.py`: Defines the CNN model architecture.
- `src/train.py`: Contains the training loop and evaluation metrics.
- `src/README.md`: This README file.

## Model Architecture

The model (SimpleMNISTNet) uses:
- Multiple convolutional layers with batch normalization and ReLU activation
- MaxPooling and AveragePooling for spatial reduction
- No fully connected layers (except final 1x1 convolution)
- Parameter count: ~8k

Key features:
- Uses batch normalization after each convolution
- Employs skip connections
- Uses 1x1 convolution for final classification
- Implements dropout for regularization

The training script:
- Uses SGD optimizer with momentum
- Implements learning rate scheduling
- Applies data augmentation (random rotation)
- Saves the best model based on validation accuracy
- Generates training metrics plots

## Results

The model achieves:
- Test Accuracy: >99.4%
- Parameter Count: ~8k parameters
- Training Time: ~15 epochs

Training metrics are plotted and saved as 'training_metrics.png', showing:
- Loss progression
- Accuracy progression

## Model Features

1. **Efficient Architecture**:
   - Uses batch normalization for faster convergence
   - Implements skip connections for better gradient flow
   - Employs pooling layers for spatial reduction

2. **Training Optimizations**:
   - Learning rate scheduling
   - Data augmentation
   - Momentum optimizer
   - Batch normalization

3. **Monitoring and Validation**:
   - Tracks training and test metrics
   - Saves best model checkpoint
   - Generates performance plots
   - Validates parameter count constraints

## Requirements

Install the required packages:

  ```bash
  pip install -r requirements.txt
  ```

  Main dependencies:
  - PyTorch >= 1.9.0
  - torchvision >= 0.10.0
  - numpy >= 1.19.0
  - tqdm >= 4.65.0

  ## Training

  To train the model:

  ```bash
  python src/train.py
  ```

  The training process includes:
  - SGD optimizer with momentum (0.9)
  - Learning rate scheduling (StepLR)
  - Data augmentation (random rotation)
  - Batch normalization after each convolution
  - Model validation and parameter count checking

  ## Results

  The model achieves:
  - Test Accuracy: >99.4%
  - Parameter Count: ~14k parameters
  - Training Time: ~15 epochs

  Training progress is monitored with:
  - Loss tracking for both training and test sets
  - Accuracy metrics for training and testing
  - Progress bars showing batch-wise progress
  - Best model checkpoint saving

  ## Model Details

  The CNN architecture:
  ```python
  # Example model structure
  layer1 = Conv2d(1, 8) -> BatchNorm -> ReLU
  layer2 = Conv2d(8, 12) -> BatchNorm -> ReLU
  layer3 = Conv2d(12, 16) -> BatchNorm -> ReLU
  MaxPool2d
  ...
  Final 1x1 Conv2d for classification
  ```

  ## Usage

  ```python
  from model import SimpleMNISTNet

  # Initialize model
  model = SimpleMNISTNet()

  # View model summary
  model.print_model_summary()

  # Check parameter count
  num_params = model.count_parameters()
  ```

  ## Training Visualization

  The training script automatically generates plots showing:
  - Training and test loss curves
  - Training and test accuracy progression
  - Saved as 'training_metrics.png'

  ## Contributing

  Feel free to:
  - Open issues for bugs or enhancements
  - Submit pull requests with improvements
  - Share feedback on model architecture

  ## License

  This project is open-source and available under the MIT License.
