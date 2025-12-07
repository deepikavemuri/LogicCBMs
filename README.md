# LogicCBMs: Logic-Enhanced Concept-Based Learning

This is the repository that contains the source code for our WACV 2026 paper **LogicCBMs: Logic-Enhanced Concept-Based Learning**. The code is build on top of the [Concept Bottleneck Models](https://github.com/yewsiang/ConceptBottleneck) and [difflogic](https://github.com/Felix-Petersen/difflogic) codebases.
It specifically contains the Vanilla and Boolean CBM and LogicCBM implementations.

This codebase contains library levels changes to the difflogic package. This repo is a WIP, so in order to run the models, install the difflogic package and replace the contents of `difflogic.py` and `functional.py` with the ones provided in this repo.

### Training Models
A joint model can be trained on CUB using the following script.
```
python3 experiments.py cub Joint --seed 2 -save_dir <SAVE_DIR_PATH> -e 40 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -n_attributes 312 -attr_loss_weight 0.001 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -save_step 20 -end2end
```

A logic model can be trained on CUB using the following script.
```
python3 experiments.py cub Logic --seed 0 -save_dir <SAVE_DIR_PATH> -e 40 -ll_connections random -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -n_attributes 312 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -n_logic_neurons 250 -n_logic_layers 1 -save_step 100 -end2end -fixed_gates
```
Logic models can use the `use_pretrained_bb_con` flag inorder to load and finetune joint model weights.
```
python3 experiments.py cub Logic --seed 0 -save_dir <SAVE_DIR_PATH> -e 40 -ll_connections random -optimizer sgd -pretrained -use_pretrained_bb_con -use_aux -use_attr -weighted_loss multiple -n_attributes 312 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -n_logic_neurons 250 -n_logic_layers 1 -save_step 100 -end2end -fixed_gates
```

Should you find our work useful, please cite

