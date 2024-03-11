import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Define the training function
def train_encoder(autoencoder, data, num_epochs, batch_size, lr):
    autoencoder.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr)
    autoencoder.to(device)  # Move the model to GPU
    data = data.to(device)  # Move the data to GPU
    for epoch in range(num_epochs):
        dataloader = DataLoader(data, batch_size, shuffle=True)
        for batch in dataloader:
            recon_loss = 0.0
            # Forward pass
            output = autoencoder(batch)
            loss = criterion(output, batch)
            recon_loss += loss.item()

            # Backward pass and parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print the loss every 100 iterations
        if (epoch + 1) % 10 == 0:
            print('Epoch {}/{} training, Loss:{}, Learning Rate:{}'.format(epoch + 1, num_epochs, loss.item(), lr))

    return autoencoder


def encode(autoencoder, data):
    data = data.to(device)
    return autoencoder.encoder(data).detach().cpu()  # Return the computation result to CPU
