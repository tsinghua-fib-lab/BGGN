# BGGN

This is the official PyTorch implementation of BGGN model in [Bundle Recommendation and Generation with Graph Neural Networks](https://ieeexplore.ieee.org/document/9546546) as described in the following TKDE 2021 paper:

```
@article{chang2021bundle,
  title={Bundle Recommendation and Generation with Graph Neural Networks},
  author={Chang, Jianxin and Gao, Chen and He, Xiangnan and Jin, Depeng and Li, Yong},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2021},
  publisher={IEEE}
}
```

# Dependencies
Python 3, PyTorch(1.2.0)

Other dependencies can be installed via 

```
pip install -r requirements.txt
```

## Run Demos
You can simply run the following command to reproduce the experiments on corresponding dataset and model

```
python run_exp.py -c config/bggn_Youshu.yaml -t -f exp/Youshu -n 4 -g 3
```

## Misc

The implemention is based on *[GRAN](https://github.com/lrjconan/GRAN)*.
