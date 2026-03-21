# Redes Neurais na Resolução de Equações Diferenciais

Material de estudo da disciplina **Redes Neurais na Resolução de Equações Diferenciais** — Doutorado PPGMC/UESC.

## Sobre

Este repositório reúne anotações, implementações e exemplos desenvolvidos ao longo da disciplina, com foco em:

- Fundamentos de PyTorch (tensores, autograd, construção de redes)
- Fundamentos de Machine Learning (viés-variância, MLE, regularização, SGD)
- Physics-Informed Neural Networks (PINNs)
- Resolução de EDOs e EDPs via redes neurais

## Estrutura

```
.
├── notebooks/
│   ├── Notebook_Pytorch.ipynb           # Notebook da aula (regressão, redes MLP)
│   └── seminario_ml_basics.ipynb        # Seminário 4 — ML Basics (Cap. 5, Goodfellow)
├── scripts/
│   ├── pytorch_test.py                  # Teste de instalação do PyTorch
│   ├── tensores.py                      # Exemplos de operações com tensores
│   └── autograd.py                      # Exemplos de diferenciação automática
├── pytorch_guia_estudo.md               # Guia completo de PyTorch (do básico a PINNs)
└── README.md
```

## Seminários

| # | Tema | Referência | Notebook |
|---|------|------------|----------|
| 4 | Fundamentos de Machine Learning | Goodfellow et al., Cap. 5 | [`seminario_ml_basics.ipynb`](notebooks/seminario_ml_basics.ipynb) |

### Seminário 4 — Machine Learning Basics

Notebook interativo (~60 min) cobrindo o Capítulo 5 do *Deep Learning* (Goodfellow, Bengio & Courville), com implementações em PyTorch e scikit-learn:

- Regressão linear com solução analítica (equações normais)
- Capacidade, overfitting/underfitting e regularização L1/L2
- Trade-off viés-variância e consistência de estimadores
- MLE, conexão com KL divergence e cross-entropy
- Estimação MAP e equivalência com weight decay
- Algoritmos clássicos (k-NN, SVM, árvores de decisão)
- SGD: batch vs. mini-batch vs. estocástico puro
- Pipeline completo de classificação com MLP
- Desafios que motivam Deep Learning: maldição da dimensionalidade, limitações da suavidade local e manifold learning

## Requisitos

- Python 3.10+
- PyTorch 2.x (com CUDA 12.1)
- NumPy
- Matplotlib
- Scikit-learn
- SciPy

### Instalação

```bash
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux/Mac

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib scikit-learn scipy jupyter ipykernel
```

### Verificação

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Referências

- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press. — [deeplearningbook.org](https://www.deeplearningbook.org/)
- [Documentação PyTorch](https://pytorch.org/docs/stable/)
- Raissi, M. et al. (2019). *Physics-informed neural networks.* Journal of Computational Physics.
- [DeepXDE](https://deepxde.readthedocs.io/) — Biblioteca de alto nível para PINNs
- Mitchell, T. (1997). *Machine Learning.* McGraw-Hill.
