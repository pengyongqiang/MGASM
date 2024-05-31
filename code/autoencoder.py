import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# 定义训练函数
def AE(emb_dim, data, epochs, batch_size=1024, lr=0.01):
    data = torch.tensor(data, dtype=torch.float32).to(device)
    autoencoder = Autoencoder(data.shape[-1], emb_dim)
    autoencoder.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr)
    for epoch in range(epochs):
        dataloader = DataLoader(data, batch_size, shuffle=True)
        for batch in dataloader:
            recon_loss = 0.0
            # 前向传递
            output = autoencoder(batch)
            loss = criterion(output, batch)
            recon_loss += loss.item()

            # 反向传播和参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 每100个迭代打印一次损失
        if (epoch + 1) % 50 == 0:
            print('第{}/{}轮训练，损失值:{},学习率:{}'.format(epoch + 1, epochs, loss.item(), lr))

    return autoencoder.encoder(data).detach().cpu()
