# KSGAN
Source code for model KSGAN
Paper: 
Hu, K., Liu, H., & Hao, T. A Knowledge Selective Adversarial Network for Link Prediction in Knowledge Graph. In CCF International Conference on Natural Language Processing and Chinese Computing (pp. 171-183).

## Installation
Run:
`conda env create -f py3torch.yaml`
to setup anaconda environment including pytorch 0.2.0, numpy, etc.

## Running models
To reproduce the results, run with FB15k-237:  set margin γ = 3, regularization λ = 1 in config_fb15k237.yaml
```
python gan_train.py --config=config_fb15k237.yaml --g_config=ComplEx --d_config=TransE
```

```
python gan_train.py --config=config_fb15k237.yaml --g_config=ComplEx --d_config=TransD
```

```
python gan_train.py --config=config_fb15k237.yaml --g_config=TransE --d_config=ComplEx
```

```
python gan_train.py --config=config_fb15k237.yaml --g_config=TransD --d_config=ComplEx
```

To reproduce the results, run with WN18: set margin γ = 3, regularization λ = 0.1 in config_wn18.yaml
```
python gan_train.py --config=config_wn18.yaml --g_config=ComplEx --d_config=TransE
```

```
python gan_train.py --config=config_wn18.yaml --g_config=ComplEx --d_config=TransD
```

```
python gan_train.py --config=config_wn18.yaml --g_config=TransE --d_config=ComplEx
```

```
python gan_train.py --config=config_wn18.yaml --g_config=TransD --d_config=ComplEx
```

To reproduce the results, run with WN18RR: set margin γ = 3, regularization λ = 0.1 in wn18rr.yaml
```
python gan_train.py --config=config_wn18rr.yaml --g_config=ComplEx --d_config=TransE
```

```
python gan_train.py --config=config_wn18rr.yaml --g_config=ComplEx --d_config=TransD
```

```
python gan_train.py --config=config_wn18rr.yaml --g_config=TransE --d_config=ComplEx
```

```
python gan_train.py --config=config_wn18rr.yaml --g_config=TransD --d_config=ComplEx
```

## Acknowledgements
The code is inspired by [KBGAN](https://github.com/cai-lw/KBGAN).

## Citation

If you found this codebase or our work useful please cite us:
```
@inproceedings{KSGAN2019,
  title={A knowledge selective adversarial network for link prediction in knowledge graph},
  author={Hu, Kairong and Liu, Hai and Hao, Tianyong},
  booktitle={CCF International Conference on Natural Language Processing and Chinese Computing},
  pages={171--183},
  year={2019},
  organization={Springer}
}

```
