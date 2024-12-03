import torch
import torch.nn as nn
import torch.optim as optim

class ImprovedDLinear(nn.Module):
    def __init__(self, input_size):
        super(ImprovedDLinear, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.output(x)
        return x

# 初始化模型
model = ImprovedDLinear(input_size=2)
criterion = nn.MSELoss()  # 可以考虑换成 nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练和测试代码保持不变，只需替换模型初始化部分
