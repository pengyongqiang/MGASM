import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

device = 'cpu'


class SimilarityModel_bak(nn.Module):
    def __init__(self, input_size):
        super(SimilarityModel_bak, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        return out


def z_score_normalization(tensor):
    mean = tensor.mean()
    std = tensor.std()
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor


def train_model(positive_example, negative_example, num_epochs=100, lr=0.01, batch_size=32):
    merged_tensor = torch.cat((positive_example, negative_example), dim=0).to(device)
    merged_tensor = z_score_normalization(merged_tensor)

    labels = torch.cat((torch.ones(len(positive_example)), torch.zeros(len(negative_example))))

    dataset = TensorDataset(merged_tensor, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model and define the loss function and optimizer
    model = SimilarityModel_bak(merged_tensor.size(-1)).to(device)

    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Set up the Cosine Annealing Learning Rate Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-8)

    # Train the model
    for epoch in range(num_epochs):
        total_loss = 0.0  # Variable to accumulate the total loss for this epoch

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        average_loss = total_loss / len(dataloader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss}, LR: {optimizer.param_groups[0]['lr']}")

        scheduler.step()

    return model


def predict_score(model, input_vector):
    input_vector = input_vector.to(device)
    with torch.no_grad():
        output = model(input_vector)
    return output.detach()
