# TermDiscovery KNN

This repository contains the code of KNN based term discovery algorithm, implemented for our paper  [Unsupervised discovery of sign terms by K-nearest neighbours approach (ECCV '20)](https://link.springer.com/chapter/10.1007/978-3-030-66096-3_22  "Link to paper"). Even though we implemented it for discovery of sign language terms, this repo is intended for general purpose term (motif) disvovery. Given a set of feature vector time series, the algorithm outputs pairs of similar segments. 

This implementation closely follows the original algorithm that is proposed by Thual et. al. in [A K-nearest neighbours approach to unsupervised spoken term discovery](https://hal.archives-ouvertes.fr/hal-01947953). 


### Demo

Have a look at the [notebook](./Run_KNN_UTD.ipynb "Run_KNN_UTD.ipynb") to learn how you can call the module and what each parameter does. The provided [sample data](./data) contains the time series features computed from sign language videos.  

You can feed your own time series features from another domain and tune the parameters for your problem. 

### Dependencies

The main dependency is the [FAISS](https://github.com/facebookresearch/faiss) library, which greatly speeds up the KNN search, by utilizing CUDA capable GPU's.  If you don't have a CUDA capable GPU, then you can set `use_gpu`parameter to `False`.  

The other dependencies are common packages such as Numpy, Pandas, Scipy, Numba etc.  
The code is developed using Python version 3.6.

### Notes

If you want to compare discovered pairs to a set of ground truth labels, you can make use of [Term Discovery Evaluation](https://github.com/korhanpolat/tdev2 "TDE Toolkit") toolkit. 


### Cite

Please cite the original [paper](https://hal.archives-ouvertes.fr/hal-01947953 "A K-nearest neighbours approach to unsupervised spoken term discovery")

```
@INPROCEEDINGS{8639515,
  author={A. {Thual} and C. {Dancette} and J. {Karadayi} and J. {Benjumea} and E. {Dupoux}},
  booktitle={2018 IEEE Spoken Language Technology Workshop (SLT)}, 
  title={A K-Nearest Neighbours Approach To Unsupervised Spoken Term Discovery}, 
  year={2018},
  pages={491-497},
  doi={10.1109/SLT.2018.8639515}}
```

and [our paper](https://link.springer.com/chapter/10.1007/978-3-030-66096-3_22 "Unsupervised discovery of sign terms by K-nearest neighbours approach (ECCV '20)")

```
@InProceedings{10.1007/978-3-030-66096-3_22,
author="Polat, Korhan and Sara{\c{c}}lar, Murat",
title="Unsupervised Discovery of Sign Terms by K-Nearest Neighbours Approach",
booktitle="Computer Vision -- ECCV 2020 Workshops",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="310--321"}
```

if you use this code in your work.
