# SimCLR.jl
SimCLR implementation in Julia with [Knet](https://denizyuret.github.io/Knet.jl/latest/)

[SimCLR paper](https://arxiv.org/abs/2002.05709):
```
@article{chen2020simple,
  title={A Simple Framework for Contrastive Learning of Visual Representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  journal={arXiv preprint arXiv:2002.05709},
  year={2020}
}
``` 

Required packages
```
- CUDA
- Knet
- MLDatasets
- PyCall
- LinearAlgebra
- JLD2
```

For pretraining: 
```
julia pretrain.jl
```

For linear evaluation: 
```
julia pretrain.jl
```

For semi supervised learning: 
```
julia pretrain.jl
```

## Experiments (ResNet18)

- CIFAR-10 linear evaluation test accuracy results:

| Epoch      | Accuracy |
| ----------- | ----------- |
| 200     | 84.08       |
| 400   | 87.13       |
| 600       | 88.11       |
|800   | 88.49      |
|1000  | 89.25*| 

\* : SimCLR 90.97

- CIFAR-10 linear evaluation test accuracy results:


| Task   | Accuracy |
| ----------- | ----------- |
| Fine-tuning w/ all data    | 92.59       |
| Semi-supervised (10\%)  | 88.21      |
| Semi-supervised (1\%)      | 79.56     |

