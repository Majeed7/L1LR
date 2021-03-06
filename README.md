<p align="center">
    <img alt="GitHub Repo license" src="https://img.shields.io/github/license/Majeed7/L1LR?logo=license&style=flat-square">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Majeed7/L1LR?logo=stars&style=flat-square">
    <img alt="GitHub issues" src="https://img.shields.io/github/issues-raw/Majeed7/L1LR?logo=open_issues&style=flat-square">
</p>

# Description

This is a repository of the code for the ***An efficient projection neural network for ℓ<sub>1</sub>-regularized logistic regression*** paper.

## Getting Started

### Dependencies

* Python 3.6+
* PyTorch 1.6+
* sklearn 0.23
* Matplotlib (for graphs and figures)

### Installing

* Download repository
* Install Dependencies
* Datasets must be in Libsvm format, download and put them in the [datasets](./datasets) folder in the root of the project. Download link available in the **Acknowledgments**.
    
### Executing program

* Run each file in form of `run_(dataset_name).py` to obtain corresponding results of the proposed method and sklearn LogisticRegression model

* Run each file in form of `script_(figure's_name).py` to generate the paper's figures

* **Note:** Results that acquired from all methods available in .mat format in the [results folder](./results).

## Authors

* Amir Atashin
* Majid Mohammadi

## Version History

* 1.0
    * Initial Release

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details

## Citation
Please consider referencing the following research paper of this repository if you find it useful or relevant to your research:
```
@misc{mohammadi2021efficient,
      title={An efficient projection neural network for $\ell_1$-regularized logistic regression}, 
      author={Majid Mohammadi and Amir Ahooye Atashin and Damian A. Tamburri},
      year={2021},
      eprint={2105.05449},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgments

Inspiration, code snippets, etc.
* [LIBSVM Datasets](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html)
* Results of other methods obtained from [L1General.zip](https://www.cs.ubc.ca/~schmidtm/Software/L1General.html), *a package by Mark Schmidt*
