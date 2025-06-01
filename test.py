import torch
import torch.nn as nn
import torch.optim as optim

# Data: x from 0 to 9, y = 2x + 1
x = torch.arange(10, dtype=torch.float32).unsqueeze(1)  # shape (10,1)
y = 2 * x + 1

# Model: input 1 → hidden 5 → output 1
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 5)
        self.relu = nn.ReLU()
        self.output = nn.Linear(5, 1)
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        return self.output(x)

model = SimpleNet()

# Loss & optimizer
loss_fn = nn.MSELoss()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


x = x.to(device)
y = y.to(device)
model = SimpleNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
for epoch in range(10000):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

# Test
model.eval()
print("Predictions:", model(x).detach().squeeze())
print("Ground truth:", y.squeeze())

x1 = (torch.arange(10, dtype=torch.float32)+5).unsqueeze(1).to(device)  # shape (10,1)
y1 = (2 * x1 + 1).to(device)

print("\n\n")
print("Predictions:", model(x1).detach().squeeze())
print("Ground truth:", y1.squeeze())