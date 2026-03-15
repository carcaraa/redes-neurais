# Redes Neurais na Resolução de Equações Diferenciais

Material de estudo da disciplina **Redes Neurais na Resolução de Equações Diferenciais** - Doutorado PPGMC.

## Sobre

Este repositório reúne anotações, implementações e exemplos desenvolvidos ao longo da disciplina, com foco em:

- Fundamentos de PyTorch (tensores, autograd, construção de redes)
- Physics-Informed Neural Networks (PINNs)
- Resolução de EDOs e EDPs via redes neurais

## Estrutura

```
.
├── pytorch_guia_estudo.md   # Guia completo de PyTorch (do básico a PINNs)
└── README.md
```

## Requisitos

- Python 3.10+
- PyTorch 2.x
- Matplotlib

### Instalação

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install torch torchvision matplotlib
```

Para suporte a GPU (CUDA 12.1):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Referências

- [Documentação PyTorch](https://pytorch.org/docs/stable/)
- Raissi, M. et al. (2019). *Physics-informed neural networks.* Journal of Computational Physics.
- [DeepXDE](https://deepxde.readthedocs.io/) — Biblioteca de alto nível para PINNs
