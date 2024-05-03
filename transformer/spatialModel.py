import torch

class MLP(torch.nn.Module):
    def __init__(self, nIn, nOut, Hidlayer, withReLU):
        """
        Initializes the MLP model.

        Args:
            nIn (int): The number of input features.
            nOut (int): The number of output features.
            Hidlayer (list of int): A list containing the sizes of each hidden layer.
            withReLU (bool): Whether to include ReLU activation functions after each hidden layer.
        """
        super(MLP, self).__init__()  # Initialize the superclass torch.nn.Module
        
        numHidlayer = len(Hidlayer)  # The number of hidden layers
        net = []  # This list will hold the layers of the MLP

        # Add the first layer from input to the first hidden layer
        net.append(torch.nn.Linear(nIn, Hidlayer[0]))
        if withReLU:
            # Optionally add a ReLU activation function after the first hidden layer
            net.append(torch.nn.ReLU())
        
        # Dynamically add the remaining hidden layers
        for i in range(0, numHidlayer-1):
            # Add a linear layer from the i-th hidden layer to the (i+1)-th
            net.append(torch.nn.Linear(Hidlayer[i], Hidlayer[i+1]))
            if withReLU:
                # Optionally add a ReLU activation function after each hidden layer
                net.append(torch.nn.ReLU())
        
        # Add the final layer from the last hidden layer to the output
        net.append(torch.nn.Linear(Hidlayer[-1], nOut))
        
        # Create the sequential model from the list of layers
        self.mlp = torch.nn.Sequential(*net)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor produced by the MLP.
        """
        # Pass the input through the MLP to get the output
        return self.mlp(x)