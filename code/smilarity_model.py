import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

device = 'cpu'


class SimilarityModel(nn.Module):
    def __init__(self, input_size):
        super(SimilarityModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 196)  # Increase the number of units
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(196, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


def z_score_normalization(tensor):
    mean = tensor.mean()
    std = tensor.std()
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor


def train_model(positive_example, negative_example, dataset, num_epochs=100, batch_size=64):
    if dataset == 'wd':
        lr = 0.004
    else:
        lr = 0.002
    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)

    merged_tensor = torch.cat((positive_example, negative_example), dim=0).to(device)
    # merged_tensor = z_score_normalization(merged_tensor)

    labels = torch.cat((torch.ones(len(positive_example)), torch.zeros(len(negative_example))))

    dataset = TensorDataset(merged_tensor, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimilarityModel(merged_tensor.size(-1)).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # Adjusted weight decay
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    for epoch in range(num_epochs):
        total_loss = 0.0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        average_loss = total_loss / len(dataloader)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss}, LR: {optimizer.param_groups[0]['lr']}")

        scheduler.step()

    return model


def predict_score(model, input_vector):
    input_vector = input_vector.to(device)
    with torch.no_grad():
        output = model(input_vector)
    return output.detach()
