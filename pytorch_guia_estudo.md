# Guia de Estudo — PyTorch

> **Disciplina:** Redes Neurais na Resolução de Equações Diferenciais  
> **Ambiente:** VSCode + Python 3.x + PyTorch  

---

## Sumário

- [Guia de Estudo — PyTorch](#guia-de-estudo--pytorch)
  - [Sumário](#sumário)
  - [1. O que é PyTorch?](#1-o-que-é-pytorch)
    - [PyTorch vs TensorFlow (resumo rápido)](#pytorch-vs-tensorflow-resumo-rápido)
  - [2. Instalação e Configuração no VSCode](#2-instalação-e-configuração-no-vscode)
    - [2.1 Criando um ambiente virtual](#21-criando-um-ambiente-virtual)
    - [2.2 Instalando PyTorch](#22-instalando-pytorch)
    - [2.3 Configurando o VSCode](#23-configurando-o-vscode)
    - [2.4 Verificando a instalação](#24-verificando-a-instalação)
  - [3. Tensores — A Base de Tudo](#3-tensores--a-base-de-tudo)
    - [3.1 Criando tensores](#31-criando-tensores)
    - [3.2 Propriedades importantes](#32-propriedades-importantes)
    - [3.3 Operações essenciais](#33-operações-essenciais)
    - [3.4 Conversão NumPy ↔ PyTorch](#34-conversão-numpy--pytorch)
  - [4. Autograd — Diferenciação Automática](#4-autograd--diferenciação-automática)
    - [4.1 Conceito básico](#41-conceito-básico)
    - [4.2 Gradientes de funções vetoriais](#42-gradientes-de-funções-vetoriais)
    - [4.3 `torch.autograd.grad` — Fundamental para PINNs](#43-torchautogradgrad--fundamental-para-pinns)
    - [4.4 Derivadas parciais (múltiplas variáveis)](#44-derivadas-parciais-múltiplas-variáveis)
  - [5. Construindo Redes Neurais com `torch.nn`](#5-construindo-redes-neurais-com-torchnn)
    - [5.1 `nn.Module` — A classe base](#51-nnmodule--a-classe-base)
    - [5.2 `nn.Sequential` — Atalho para redes sequenciais](#52-nnsequential--atalho-para-redes-sequenciais)
    - [5.3 Camadas e ativações comuns](#53-camadas-e-ativações-comuns)
    - [5.4 Inspecionando parâmetros](#54-inspecionando-parâmetros)
  - [6. Loop de Treinamento](#6-loop-de-treinamento)
    - [Anatomia do loop](#anatomia-do-loop)
  - [7. Otimizadores e Funções de Perda](#7-otimizadores-e-funções-de-perda)
    - [7.1 Otimizadores](#71-otimizadores)
    - [7.2 Schedulers (ajuste de learning rate)](#72-schedulers-ajuste-de-learning-rate)
    - [7.3 Funções de perda](#73-funções-de-perda)
  - [8. Datasets e DataLoaders](#8-datasets-e-dataloaders)
  - [9. GPU e CUDA](#9-gpu-e-cuda)
  - [10. Salvando e Carregando Modelos](#10-salvando-e-carregando-modelos)
  - [11. Exemplo Completo 1 — Regressão Simples](#11-exemplo-completo-1--regressão-simples)
  - [12. Exemplo Completo 2 — PINN para EDO](#12-exemplo-completo-2--pinn-para-edo)
    - [Explicação da loss da PINN](#explicação-da-loss-da-pinn)
  - [13. Exemplo Completo 3 — PINN para EDP (Equação do Calor)](#13-exemplo-completo-3--pinn-para-edp-equação-do-calor)
  - [14. Dicas e Boas Práticas](#14-dicas-e-boas-práticas)
    - [Para PINNs](#para-pinns)
    - [Gerais de PyTorch](#gerais-de-pytorch)
    - [Depuração no VSCode](#depuração-no-vscode)
  - [15. Referências](#15-referências)

---

## 1. O que é PyTorch?

PyTorch é uma biblioteca open-source de computação tensorial e diferenciação automática,
desenvolvida pelo Facebook AI Research (FAIR). É o framework dominante na pesquisa acadêmica
em deep learning por três razões principais:

- **Grafo computacional dinâmico (eager execution):** o grafo é construído a cada forward pass,
  permitindo usar `if`, `for` e depuração normal do Python.
- **Autograd:** sistema de diferenciação automática que calcula gradientes de qualquer
  computação tensorial — essencial para PINNs (Physics-Informed Neural Networks).
- **Ecossistema rico:** integração com NumPy, suporte a GPU via CUDA, e bibliotecas como
  `torchvision`, `torchdiffeq`, `deepxde`, etc.

### PyTorch vs TensorFlow (resumo rápido)

| Aspecto | PyTorch | TensorFlow |
|---------|---------|------------|
| Modo padrão | Eager (dinâmico) | Graph (estático, eager opcional) |
| Depuração | `print()` e breakpoints normais | Mais complexa |
| Comunidade acadêmica | Dominante | Mais voltado para produção |
| Diferenciação de ordem superior | Nativa e simples | Possível, porém mais verbosa |

Para **resolver EDOs/EDPs com redes neurais**, PyTorch é a escolha natural porque
a diferenciação de ordem superior (`torch.autograd.grad`) é direta e eficiente.

---

## 2. Instalação e Configuração no VSCode

### 2.1 Criando um ambiente virtual

```bash
# No terminal do VSCode (Ctrl + `)
python -m venv venv

# Ativando
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 2.2 Instalando PyTorch

Acesse [pytorch.org/get-started](https://pytorch.org/get-started/locally/) e selecione
sua configuração. Exemplos comuns:

```bash
# Somente CPU
pip install torch torchvision

# Com CUDA 12.1 (GPU NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2.3 Configurando o VSCode

1. Instale a extensão **Python** (Microsoft).
2. Selecione o interpretador do venv: `Ctrl+Shift+P` → "Python: Select Interpreter" → escolha o do `venv`.
3. (Opcional) Instale a extensão **Jupyter** para usar notebooks `.ipynb` dentro do VSCode.

### 2.4 Verificando a instalação

```python
import torch

print(f"PyTorch versão: {torch.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 3. Tensores — A Base de Tudo

Tensores são arrays multidimensionais, semelhantes a `numpy.ndarray`, mas com suporte
a GPU e diferenciação automática.

### 3.1 Criando tensores

```python
import torch

# A partir de listas
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# Tensores especiais
zeros = torch.zeros(3, 4)          # 3x4 de zeros
uns   = torch.ones(2, 3)           # 2x3 de uns
ident = torch.eye(3)               # Matriz identidade 3x3
ale   = torch.rand(2, 3)           # Uniforme [0, 1)
norm  = torch.randn(2, 3)          # Normal(0, 1)
seq   = torch.linspace(0, 1, 100)  # 100 pontos de 0 a 1 (essencial para PINNs)
```

### 3.2 Propriedades importantes

```python
t = torch.randn(3, 4, 5)

t.shape       # torch.Size([3, 4, 5])
t.dtype       # torch.float32
t.device      # device(type='cpu')
t.ndim        # 3  (número de dimensões)
t.numel()     # 60 (número total de elementos)
```

### 3.3 Operações essenciais

```python
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
```

### 3.4 Conversão NumPy ↔ PyTorch

```python
import numpy as np

# NumPy → Torch (compartilham memória!)
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)

# Torch → NumPy
arr2 = t.numpy()

# Cópia independente (sem compartilhar memória)
t_copia = torch.tensor(arr)  # ou t.clone()
```

---

## 4. Autograd — Diferenciação Automática

O **autograd** é o coração do PyTorch para quem trabalha com PINNs. Ele rastreia todas
as operações feitas em tensores marcados com `requires_grad=True` e calcula gradientes
automaticamente.

### 4.1 Conceito básico

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1   # y = x² + 2x + 1

y.backward()  # calcula dy/dx

print(x.grad)  # tensor(8.)  →  dy/dx = 2x + 2 = 2(3) + 2 = 8
```

### 4.2 Gradientes de funções vetoriais

```python
x = torch.linspace(0, 1, 5, requires_grad=True)
y = torch.sin(x)

# backward() exige escalar → somamos
y.sum().backward()
print(x.grad)  # cos(x) avaliado em cada ponto
```

### 4.3 `torch.autograd.grad` — Fundamental para PINNs

Para PINNs, usamos `torch.autograd.grad` em vez de `.backward()` porque precisamos
de derivadas de ordem superior e controle fino:

```python
x = torch.linspace(0, 1, 100, requires_grad=True).unsqueeze(1)  # shape (100, 1)

# Suponha que u(x) é a saída de uma rede neural
u = torch.sin(x)  # exemplo; na prática seria model(x)

# Primeira derivada: du/dx
du_dx = torch.autograd.grad(
    outputs=u,
    inputs=x,
    grad_outputs=torch.ones_like(u),  # necessário quando u não é escalar
    create_graph=True                  # ESSENCIAL para derivadas de ordem superior
)[0]

# Segunda derivada: d²u/dx²
d2u_dx2 = torch.autograd.grad(
    outputs=du_dx,
    inputs=x,
    grad_outputs=torch.ones_like(du_dx),
    create_graph=True
)[0]

print(du_dx.shape)   # (100, 1)
print(d2u_dx2.shape) # (100, 1)
```

> **Ponto-chave:** `create_graph=True` mantém o grafo computacional das derivadas,
> permitindo calcular derivadas de ordem superior e fazer backpropagation
> através das próprias derivadas — exatamente o que precisamos para incluir
> a EDP/EDO na função de perda.

### 4.4 Derivadas parciais (múltiplas variáveis)

Para EDPs com variáveis $(x, t)$:

```python
# Pontos de colocação
x = torch.rand(500, 1, requires_grad=True)
t = torch.rand(500, 1, requires_grad=True)

# Saída da rede: u(x, t)
inp = torch.cat([x, t], dim=1)  # shape (500, 2)
u = model(inp)                  # shape (500, 1)

# Derivadas parciais
du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
du_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
d2u_dx2 = torch.autograd.grad(du_dx, x, torch.ones_like(du_dx), create_graph=True)[0]
```

---

## 5. Construindo Redes Neurais com `torch.nn`

### 5.1 `nn.Module` — A classe base

Toda rede neural em PyTorch herda de `nn.Module`:

```python
import torch
import torch.nn as nn

class MinhaRede(nn.Module):
    def __init__(self, n_entrada, n_oculta, n_saida):
        super().__init__()
        self.camada1 = nn.Linear(n_entrada, n_oculta)
        self.camada2 = nn.Linear(n_oculta, n_oculta)
        self.camada3 = nn.Linear(n_oculta, n_saida)
        self.ativacao = nn.Tanh()  # Tanh é muito usada em PINNs
    
    def forward(self, x):
        x = self.ativacao(self.camada1(x))
        x = self.ativacao(self.camada2(x))
        x = self.camada3(x)  # última camada sem ativação (regressão)
        return x

model = MinhaRede(n_entrada=1, n_oculta=32, n_saida=1)
print(model)
```

### 5.2 `nn.Sequential` — Atalho para redes sequenciais

```python
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 1)
)
```

### 5.3 Camadas e ativações comuns

| Camada / Ativação | Uso |
|---|---|
| `nn.Linear(in, out)` | Camada densa (fully connected) |
| `nn.Tanh()` | Ativação suave, diferenciável — boa para PINNs |
| `nn.ReLU()` | Ativação popular em classificação — **evite em PINNs** (derivada descontínua) |
| `nn.Sigmoid()` | Saída entre 0 e 1 |
| `nn.SiLU()` / `nn.GELU()` | Alternativas suaves ao ReLU |
| `nn.Softmax(dim)` | Saída probabilística |

> **Para PINNs:** prefira `Tanh`, `SiLU` ou `Softplus`. Evite `ReLU` porque sua
> segunda derivada é zero em quase todo lugar, o que prejudica EDPs que dependem
> de $\nabla^2 u$.

### 5.4 Inspecionando parâmetros

```python
# Ver todos os parâmetros treináveis
for nome, param in model.named_parameters():
    print(f"{nome}: shape={param.shape}, total={param.numel()}")

# Número total de parâmetros
total = sum(p.numel() for p in model.parameters())
print(f"Total de parâmetros: {total}")
```

---

## 6. Loop de Treinamento

O loop de treinamento em PyTorch é **explícito** — você controla cada passo:

```python
model = MinhaRede(1, 32, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Dados de exemplo
x_train = torch.linspace(0, 1, 100).unsqueeze(1)
y_train = torch.sin(2 * 3.14159 * x_train)

n_epochs = 1000

for epoch in range(n_epochs):
    # 1. Forward pass
    y_pred = model(x_train)
    
    # 2. Calcular perda
    loss = loss_fn(y_pred, y_train)
    
    # 3. Backward pass (calcula gradientes)
    optimizer.zero_grad()  # IMPORTANTE: zerar gradientes anteriores
    loss.backward()
    
    # 4. Atualizar pesos
    optimizer.step()
    
    # Log
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {loss.item():.6f}")
```

### Anatomia do loop

1. **Forward pass:** dados passam pela rede → predições.
2. **Loss:** mede o erro entre predição e alvo.
3. **`optimizer.zero_grad()`:** zera os gradientes (eles acumulam por padrão!).
4. **`loss.backward()`:** calcula $\partial \mathcal{L} / \partial \theta$ para todos os parâmetros.
5. **`optimizer.step()`:** atualiza $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$.

---

## 7. Otimizadores e Funções de Perda

### 7.1 Otimizadores

```python
# Os mais usados
optim_sgd   = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optim_adam   = torch.optim.Adam(model.parameters(), lr=1e-3)
optim_lbfgs  = torch.optim.LBFGS(model.parameters(), lr=1.0)

# L-BFGS é popular em PINNs (convergência rápida para problemas menores)
# Uso do L-BFGS requer um closure:
def closure():
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    return loss

optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20)
for epoch in range(100):
    loss = optimizer.step(closure)
```

### 7.2 Schedulers (ajuste de learning rate)

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

for epoch in range(n_epochs):
    # ... treino ...
    scheduler.step()  # atualiza lr

# Outros schedulers úteis:
# ReduceLROnPlateau — reduz lr quando a loss estagna
# CosineAnnealingLR — decaimento cosseno
```

### 7.3 Funções de perda

```python
# Regressão
mse  = nn.MSELoss()       # Erro quadrático médio
mae  = nn.L1Loss()        # Erro absoluto médio
hub  = nn.HuberLoss()     # Combinação suave de MSE e MAE

# Classificação
ce   = nn.CrossEntropyLoss()   # Multiclasse
bce  = nn.BCEWithLogitsLoss()  # Binária

# Loss customizada (muito comum em PINNs)
def pinn_loss(model, x, x_bc, u_bc):
    """Exemplo de loss para uma PINN."""
    u = model(x)
    
    # Resíduo da EDP
    du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    residuo = du_dx - torch.cos(x)  # exemplo: du/dx = cos(x)
    loss_pde = torch.mean(residuo ** 2)
    
    # Condição de contorno
    u_pred_bc = model(x_bc)
    loss_bc = torch.mean((u_pred_bc - u_bc) ** 2)
    
    return loss_pde + loss_bc
```

---

## 8. Datasets e DataLoaders

Para problemas maiores ou com dados em disco:

```python
from torch.utils.data import Dataset, DataLoader

class MeuDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Criar dataset e dataloader
dataset = MeuDataset(x_train, y_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# No loop de treino:
for epoch in range(n_epochs):
    for x_batch, y_batch in loader:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

> **Em PINNs:** frequentemente não usamos DataLoader porque amostramos pontos de
> colocação diretamente com `torch.rand`. Mas para problemas com dados experimentais
> combinados com a EDP, o DataLoader é útil.

---

## 9. GPU e CUDA

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {device}")

# Mover modelo e dados para a GPU
model = model.to(device)
x = x.to(device)
y = y.to(device)

# No loop de treino, garanta que tudo está no mesmo device
x_batch = x_batch.to(device)
```

**Dica:** defina o `device` no início do script e use `.to(device)` em tudo.
Assim o código funciona tanto com CPU quanto com GPU sem alterações.

---

## 10. Salvando e Carregando Modelos

```python
# Salvar (apenas state_dict — recomendado)
torch.save(model.state_dict(), "modelo.pth")

# Carregar
model = MinhaRede(1, 32, 1)              # recriar a arquitetura
model.load_state_dict(torch.load("modelo.pth"))
model.eval()  # modo de avaliação (desativa dropout, batchnorm, etc.)

# Salvar checkpoint completo (para retomar treino)
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss.item(),
}, "checkpoint.pth")

# Carregar checkpoint
ckpt = torch.load("checkpoint.pth")
model.load_state_dict(ckpt["model_state_dict"])
optimizer.load_state_dict(ckpt["optimizer_state_dict"])
```

---

## 11. Exemplo Completo 1 — Regressão Simples

Aproximar $f(x) = \sin(2\pi x)$ com uma rede neural:

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Dados
x_train = torch.linspace(0, 1, 200).unsqueeze(1)
y_train = torch.sin(2 * torch.pi * x_train)

# Modelo
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Treino
for epoch in range(3000):
    pred = model(x_train)
    loss = loss_fn(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1} | Loss: {loss.item():.2e}")

# Visualização
model.eval()
with torch.no_grad():
    y_pred = model(x_train)

plt.figure(figsize=(8, 4))
plt.plot(x_train.numpy(), y_train.numpy(), label="Exata", linewidth=2)
plt.plot(x_train.numpy(), y_pred.numpy(), "--", label="Rede Neural", linewidth=2)
plt.legend()
plt.title("Aproximação de sin(2πx)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("regressao.png", dpi=150)
plt.show()
```

---

## 12. Exemplo Completo 2 — PINN para EDO

Resolver a EDO:

$$\frac{du}{dx} = -u, \quad u(0) = 1$$

Solução exata: $u(x) = e^{-x}$.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Rede neural
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)

model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Pontos de colocação no domínio [0, 2]
n_pontos = 100

for epoch in range(5000):
    # Amostrar pontos (re-amostrar a cada época pode ajudar)
    x = torch.linspace(0, 2, n_pontos, requires_grad=True).unsqueeze(1)
    
    # Forward
    u = model(x)
    
    # Derivada du/dx via autograd
    du_dx = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    
    # Resíduo da EDO: du/dx + u = 0
    residuo = du_dx + u
    loss_edo = torch.mean(residuo ** 2)
    
    # Condição inicial: u(0) = 1
    x0 = torch.zeros(1, 1)
    u0 = model(x0)
    loss_ci = (u0 - 1.0) ** 2
    
    # Loss total
    loss = loss_edo + 10.0 * loss_ci  # peso maior na condição inicial
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1} | Loss EDO: {loss_edo.item():.2e} | "
              f"Loss CI: {loss_ci.item():.2e}")

# Comparação com solução exata
model.eval()
x_test = torch.linspace(0, 2, 200).unsqueeze(1)
with torch.no_grad():
    u_pred = model(x_test)
u_exata = torch.exp(-x_test)

plt.figure(figsize=(8, 4))
plt.plot(x_test.numpy(), u_exata.numpy(), label="Exata: $e^{-x}$", linewidth=2)
plt.plot(x_test.numpy(), u_pred.numpy(), "--", label="PINN", linewidth=2)
plt.legend()
plt.title("PINN resolvendo du/dx = -u, u(0) = 1")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pinn_edo.png", dpi=150)
plt.show()
```

### Explicação da loss da PINN

A função de perda tem dois termos:

- **Loss da EDP/EDO** ($\mathcal{L}_{\text{PDE}}$): o resíduo da equação diferencial
  avaliado nos pontos de colocação. Queremos que $\frac{du}{dx} + u \approx 0$.
- **Loss das condições de contorno/iniciais** ($\mathcal{L}_{\text{BC}}$): forçar a rede
  a satisfazer $u(0) = 1$.

$$\mathcal{L} = \mathcal{L}_{\text{PDE}} + \lambda \, \mathcal{L}_{\text{BC}}$$

O peso $\lambda$ (aqui 10.0) balanceia os dois termos.

---

## 13. Exemplo Completo 3 — PINN para EDP (Equação do Calor)

Resolver a equação do calor 1D:

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}, \quad x \in [0, 1], \; t \in [0, 0.5]$$

Com condições:
- $u(x, 0) = \sin(\pi x)$ (condição inicial)
- $u(0, t) = u(1, t) = 0$ (condições de contorno)

Solução exata: $u(x,t) = e^{-\alpha \pi^2 t} \sin(\pi x)$

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

alpha = 0.01  # difusividade térmica

class PINN_Calor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),    # entrada: (x, t)
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)     # saída: u(x, t)
        )
    
    def forward(self, x, t):
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)

model = PINN_Calor()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 10000
n_col = 2000   # pontos de colocação internos
n_bc  = 200    # pontos nas condições de contorno/iniciais

for epoch in range(n_epochs):
    # --- Pontos de colocação internos ---
    x_col = torch.rand(n_col, 1, requires_grad=True)
    t_col = torch.rand(n_col, 1, requires_grad=True) * 0.5
    
    u = model(x_col, t_col)
    
    # Derivadas parciais
    du_dt = torch.autograd.grad(u, t_col, torch.ones_like(u), create_graph=True)[0]
    du_dx = torch.autograd.grad(u, x_col, torch.ones_like(u), create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x_col, torch.ones_like(du_dx), create_graph=True)[0]
    
    # Resíduo: du/dt - α * d²u/dx² = 0
    residuo = du_dt - alpha * d2u_dx2
    loss_pde = torch.mean(residuo ** 2)
    
    # --- Condição Inicial: u(x, 0) = sin(πx) ---
    x_ci = torch.rand(n_bc, 1)
    t_ci = torch.zeros(n_bc, 1)
    u_ci = model(x_ci, t_ci)
    loss_ci = torch.mean((u_ci - torch.sin(torch.pi * x_ci)) ** 2)
    
    # --- Condições de Contorno: u(0,t) = u(1,t) = 0 ---
    t_bc = torch.rand(n_bc, 1) * 0.5
    u_x0 = model(torch.zeros(n_bc, 1), t_bc)
    u_x1 = model(torch.ones(n_bc, 1), t_bc)
    loss_bc = torch.mean(u_x0 ** 2) + torch.mean(u_x1 ** 2)
    
    # --- Loss total ---
    loss = loss_pde + 10 * loss_ci + 10 * loss_bc
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 2000 == 0:
        print(f"Epoch {epoch+1} | PDE: {loss_pde.item():.2e} | "
              f"CI: {loss_ci.item():.2e} | BC: {loss_bc.item():.2e}")

# --- Visualização ---
model.eval()
x_plot = torch.linspace(0, 1, 100)
t_plot = torch.linspace(0, 0.5, 100)
X, T = torch.meshgrid(x_plot, t_plot, indexing="ij")
x_flat = X.reshape(-1, 1)
t_flat = T.reshape(-1, 1)

with torch.no_grad():
    u_pred = model(x_flat, t_flat).reshape(100, 100)

u_exata = torch.exp(-alpha * torch.pi**2 * T) * torch.sin(torch.pi * X)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

c1 = axes[0].contourf(X.numpy(), T.numpy(), u_pred.numpy(), levels=50, cmap="hot")
axes[0].set_title("PINN")
axes[0].set_xlabel("x")
axes[0].set_ylabel("t")
plt.colorbar(c1, ax=axes[0])

c2 = axes[1].contourf(X.numpy(), T.numpy(), u_exata.numpy(), levels=50, cmap="hot")
axes[1].set_title("Solução Exata")
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")
plt.colorbar(c2, ax=axes[1])

plt.tight_layout()
plt.savefig("pinn_calor.png", dpi=150)
plt.show()
```

---

## 14. Dicas e Boas Práticas

### Para PINNs

- **Ativação:** use `Tanh` ou `SiLU` — precisamos de derivadas suaves de ordem superior.
- **Normalização das entradas:** escale $x$, $t$ para $[0, 1]$ ou $[-1, 1]$.
- **Pesos da loss:** ajuste $\lambda$ dos termos de contorno/iniciais. Pesos adaptativos
  (ex: método de Maddu et al.) podem melhorar a convergência.
- **Otimizador híbrido:** comece com Adam (~5000 épocas) e depois refine com L-BFGS.
- **Re-amostragem:** amostrar novos pontos de colocação a cada época (ou a cada N épocas)
  melhora a generalização.
- **Inicialização de Xavier:** `nn.init.xavier_normal_(layer.weight)`.

### Gerais de PyTorch

- **`torch.no_grad()`:** use para inferência (desativa autograd → economiza memória).
- **`model.eval()`:** muda comportamento de Dropout e BatchNorm.
- **`model.train()`:** volta ao modo de treino.
- **`.item()`:** converte tensor escalar para float Python (use no print de losses).
- **`detach()`:** remove tensor do grafo computacional.
- **Reprodutibilidade:**

```python
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

### Depuração no VSCode

- Use breakpoints normais — o modo eager do PyTorch permite isso.
- Inspecione tensores no debugger: `tensor.shape`, `tensor.min()`, `tensor.max()`.
- Instale `torchinfo` para visualizar a arquitetura: `pip install torchinfo`.

```python
from torchinfo import summary
summary(model, input_size=(1, 2))  # (batch, features)
```

---

## 15. Referências

- **Documentação oficial:** [pytorch.org/docs](https://pytorch.org/docs/stable/)
- **Tutorials:** [pytorch.org/tutorials](https://pytorch.org/tutorials/)
- **Raissi, M. et al. (2019).** *Physics-informed neural networks.* Journal of
  Computational Physics. — O paper original de PINNs.
- **DeepXDE:** [deepxde.readthedocs.io](https://deepxde.readthedocs.io/) — Biblioteca
  de alto nível para PINNs construída sobre PyTorch/TensorFlow.
- **torchdiffeq:** [github.com/rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq)
  — Integração de EDOs com Neural ODEs em PyTorch.

---

> *Última atualização: Março 2026*  
> *Gerado como material de estudo para doutorado.*
