import torch

# _____________________________________
# Criando Tensores
# A partir de listas
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# _____________________________________
# Propriedades importantes
# Tensores especiais
zeros = torch.zeros(3, 4)          # 3x4 de zeros
uns   = torch.ones(2, 3)           # 2x3 de uns
ident = torch.eye(3)               # Matriz identidade 3x3
ale   = torch.rand(2, 3)           # Uniforme [0, 1)
norm  = torch.randn(2, 3)          # Normal(0, 1)
seq   = torch.linspace(0, 1, 100)  # 100 pontos de 0 a 1 (essencial para PINNs)

t = torch.randn(3, 4, 5)

t.shape       # torch.Size([3, 4, 5])
t.dtype       # torch.float32
t.device      # device(type='cpu')
t.ndim        # 3  (número de dimensões)
t.numel()     # 60 (número total de elementos)

# _____________________________________
# Operações essenciais
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# Aritméticas (element-wise)
x + y          # tensor([5., 7., 9.])
x * y          # tensor([4., 10., 18.])
x ** 2         # tensor([1., 4., 9.])
torch.sin(x)   # seno element-wise
torch.exp(x)   # exponencial element-wise

# Produto matricial
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B        # equivalente a torch.matmul(A, B) → shape (3, 5)

# Reshape
t = torch.arange(12)
t.view(3, 4)     # reshape para 3x4
t.view(-1, 4)    # infere a primeira dimensão → 3x4
t.unsqueeze(0)   # adiciona dimensão → shape (1, 12)
t.unsqueeze(1)   # shape (12, 1) — muito usado em PINNs

# _____________________________________
# Conversão NumPy ↔ PyTorch
import numpy as np

# NumPy → Torch (compartilham memória!)
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)

# Torch → NumPy
arr2 = t.numpy()

# Cópia independente (sem compartilhar memória)
t_copia = torch.tensor(arr)  # ou t.clone()

