# =============================================================================
# PyTorch — Redes Neurais na Resolução de Equações Diferenciais
# =============================================================================

# -----------------------------------------------------------------------------
# Bibliotecas
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
 
# -----------------------------------------------------------------------------
# Exemplo 1: Ajuste de Curva / Regressão
# -----------------------------------------------------------------------------
 
# Amostragem de um seno
x = np.linspace(0, 2 * np.pi, 1000)
y = 2 * np.sin(x)
plt.scatter(x, y)
plt.show()
 
# Hold-out: Divisão do conjunto de dados entre treino e teste
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42, shuffle=True
)
 
# Construção dos Tensores
x_train, y_train = torch.Tensor(x_train).float(), torch.Tensor(y_train).float()
x_test, y_test = torch.Tensor(x_test).float(), torch.Tensor(y_test).float()
 
plt.scatter(x_test, y_test)
plt.show()
 
# -----------------------------------------------------------------------------
# Criação da rede neural com 1 camada escondida
# -----------------------------------------------------------------------------
class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)    # Entrada → Camada escondida
        self.Hidden1F = nn.Sigmoid()                         # Função de ativação
        self.layer2 = nn.Linear(hidden_size, output_size)    # Camada escondida → Saída
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.Hidden1F(x)
        x = self.layer2(x)
        return x
 
# -----------------------------------------------------------------------------
# Funções de ativação (referência):
#   - Sigmoid, Tanh, ReLU, SiLU, Softmax, etc.
#
# Funções de perda (referência):
#   - Regressão:     MSELoss (L2), L1Loss (MAE)
#   - Classificação: BCELoss (binária), CrossEntropyLoss (multiclasse)
# -----------------------------------------------------------------------------
 
# Definições e inicialização da rede
input_size = 1
hidden_size = 15
output_size = 1
model = NN(input_size, hidden_size, output_size)
 
print("Estrutura da RNA:")
print(model, "\n")
 
total_params = sum(p.numel() for p in model.parameters())
print("Numero total de parametros do modelo:", total_params)
 
# -----------------------------------------------------------------------------
# Função de perda e otimizador
# -----------------------------------------------------------------------------
LossF = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
 
# -----------------------------------------------------------------------------
# Operações com tensores (referência):
#   - Reshape:      .view() ou .reshape()
#   - Slicing:      tensor[0:2, :]
#   - Concatenar:   torch.cat()
#   - Matemáticas:  a + b, a * b, torch.matmul(a, b)
# -----------------------------------------------------------------------------
 
# Preparação dos dados com DataLoader
train_dataset = TensorDataset(x_train.view(-1, 1), y_train.view(-1, 1))
test_dataset = TensorDataset(x_test.view(-1, 1), y_test.view(-1, 1))
 
train_loader = DataLoader(train_dataset, batch_size=900, shuffle=False)
 
# -----------------------------------------------------------------------------
# Treinamento por Backpropagation
# -----------------------------------------------------------------------------
num_epochs = 2000
losses = []
 
model.train()
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()       # Zera o gradiente
        outputs = model(inputs)     # Feedforward
        loss = LossF(outputs, labels)  # Calcula o erro
        loss.backward()             # Retropropagação
        optimizer.step()            # Atualiza pesos e vieses
    losses.append(loss.item())
 
plt.title("Loss")
plt.plot(losses)
plt.xlabel("Época")
plt.ylabel("MSE")
plt.show()
 
# -----------------------------------------------------------------------------
# Avaliação do modelo
# -----------------------------------------------------------------------------
# Salvar:   torch.save(model, 'model.pth')
# Carregar: model = torch.load('model.pth', weights_only=False)
 
model.eval()
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
 
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
 
print("Funcionou")
plt.plot(x, y, label="Exata")
plt.scatter(x_test.numpy(), outputs.numpy(), label="Predição", color="red")
plt.legend()
plt.show()