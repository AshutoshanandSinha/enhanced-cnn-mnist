#Model 3

# Using batch normalization, RELU at end of each convolution layer, except last layer
#  parameter count: 8,000
#  Max accuracy > 99.40%
#  Max accuracy staying consistent for multiple epochs
# Problem:
#.  1. parameter above required limit

#Steps taken to improve model:
# Reduced number of parameters by reducing the number of filters in each layer


import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMNISTNet(nn.Module):
    def __init__(self):
        super(SimpleMNISTNet, self).__init__()

        # Define the combined layers with input and output channels
        self.layer1 = self._conv_relu_bn(in_channels=1, out_channels=8, kernel_size=3)   # Output: [-1, 8, 26, 26]  # Receptive field: 3
        self.layer2 = self._conv_relu_bn(in_channels=8, out_channels=12, kernel_size=3)  # Output: [-1, 12, 24, 24] # Receptive field: 5
        self.layer3 = self._conv_relu_bn(in_channels=12, out_channels=16, kernel_size=3) # Output: [-1, 16, 24, 24] # Receptive field: 7
        self.layer4 = self._conv_relu_bn(in_channels=16, out_channels=16, kernel_size=3) # Output: [-1, 16, 24, 24] # Receptive field: 9

        self.pool = nn.MaxPool2d(2)  # Output: [-1, 16, 12, 12]  # Receptive field: 10

        self.layer5 = self._conv_relu_bn(in_channels=16, out_channels=10, kernel_size=3)  # Output: [-1, 10, 12, 12] # Receptive field: 14
        self.layer6 = self._conv_relu_bn(in_channels=10, out_channels=20, kernel_size=3)  # Output: [-1, 20, 12, 12] # Receptive field: 18
        self.layer7 = self._conv_relu_bn(in_channels=20, out_channels=16, kernel_size=3)  # Output: [-1, 16, 10, 10] # Receptive field: 22
        self.layer8 = self._conv_relu_bn(in_channels=16, out_channels=10, kernel_size=3)  # Output: [-1, 10, 8, 8]   # Receptive field: 26

        self.avgpool = nn.AvgPool2d(2)  # Output: [-1, 10, 4, 4]  # Receptive field: 26

        self.layer9 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=1, bias=False)  # Output: [-1, 10, 1, 1] # Receptive field: 26

    def _conv_relu_bn(self, in_channels, out_channels, kernel_size):
        """Helper function to create a Conv -> ReLU -> BatchNorm block with bias=False."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Forward pass through the network
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)

        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        x = self.avgpool(x)

        x = self.layer9(x)

        x = x.view(x.size(0), -1)  # Flatten the output
        return F.log_softmax(x, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def print_model_summary(self):
        """Prints a summary of the model architecture."""
        print("Model Summary:")
        print("Layer (type)               Output Shape         Param #")
        print("=" * 70)
        total_params = 0
        input_tensor = torch.zeros(1, 1, 28, 28)  # Assuming input size is (1, 1, 28, 28)

        for name, layer in self.named_children():
            output_tensor = layer(input_tensor)  # Pass the input tensor through the layer
            output_shape = output_tensor.shape
            num_params = sum(p.numel() for p in layer.parameters())
            total_params += num_params
            print(f"{name:<25} {str(output_shape):<20} {num_params}")
            input_tensor = output_tensor  # Update the input tensor for the next layer

        print("=" * 70)
        print(f"Total params: {total_params}")
        print("=" * 70)

# Example of how to instantiate and print the model summary
if __name__ == "__main__":
    model = SimpleMNISTNet()
    model.print_model_summary()  # Call the summary method
    print(f"Total Parameters: {model.count_parameters()}")
