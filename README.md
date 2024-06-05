# Supervised Gromov-Wasserstein Optimal Transport

<p align="center">
  <img src="sgw_git.png" width="500" />
</p>

Supervised Gromov-Wasserstein (sGW) optimal transport, a novel extension of Gromov-Wasserstein by incorporating potential **infinity pattern** in the cost tensor. sGW enables the enforcement of application-induced constraints such as the preservation of pairwise distances by implementing the constraints as an **infinity pattern**.

<p align="center">
  <img src="Ocexample_1_full(1).png" width="400" style="margin-right: 20px;" />
  <img src="Bioexample_3_real_two_embeddings.png" width="400" />
</p>

## Requirements

Python packages: 

POT

Networkx 

SciPy

Geosketch 

Kepler-Mapper 

UMAP

Scikit-learn


## Tutorial

We included a tutorial in this repository <tutorial.ipynb>. The tutorial will guide you through the required steps to run sGW with your own dataset.

## References

[1] Z Cang, Y Wu, Y Zhao. Supervised Gromov-Wasserstein Optimal Transport. `arxiv <https://arxiv.org/abs/2401.06266>`

If you found this library helpful, please consider citing it as:

    @article{cang2024supervised,
      title={Supervised Gromov-Wasserstein Optimal Transport},
      author={Cang, Zixuan and Wu, Yaqi and Zhao, Yanxiang},
      journal={arXiv preprint arXiv:2401.06266},
      year={2024}
    }
