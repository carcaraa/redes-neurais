import torch

# _____________________________________
# Autograd — Diferenciação Automática

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1   # y = x² + 2x + 1

y.backward()  # calcula dy/dx

print(x.grad)  # tensor(8.)  →  dy/dx = 2x + 2 = 2(3) + 2 = 8

# _____________________________________
# Gradientes de funções vetoriais
x = torch.linspace(0, 1, 5, requires_grad=True)
y = torch.sin(x)

# backward() exige escalar → somamos
y.sum().backward()
print(x.grad)  # cos(x) avaliado em cada ponto

# _____________________________________
# torch.autograd.grad — Fundamental para PINNs

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

# _____________________________________
# Derivadas parciais (múltiplas variáveis)

# Pontos de colocação
x = torch.rand(500, 1, requires_grad=True)
t = torch.rand(500, 1, requires_grad=True)

# Simulando u(x, t) = sin(x) * exp(-t)  (no lugar de model)
u = torch.sin(x) * torch.exp(-t)

# Derivadas parciais
du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
du_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
d2u_dx2 = torch.autograd.grad(du_dx, x, torch.ones_like(du_dx), create_graph=True)[0]

print(f"du/dx shape:   {du_dx.shape}")    # (500, 1)
print(f"du/dt shape:   {du_dt.shape}")    # (500, 1)
print(f"d²u/dx² shape: {d2u_dx2.shape}")  # (500, 1)