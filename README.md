# [NeurIPS'24] Asynchronous Perception Machine For Efficient Test Time Training 
<p align="center">
  <a href="https://arxiv.org/pdf/2410.20535">Paper</a> | 
  <a href="https://openreview.net/forum?id=7Ye12RLZ4P&referrer=%5Bthe%20profile%20of%20Rajat%20Modi%5D(%2Fprofile%3Fid%3D~Rajat_Modi1)">Openreview</a> | 
  <a href="https://rajatmodi62.github.io/2024/10/26/hinton_apm-copy/">Blog</a> | 
  <a href="https://rajatmodi62.github.io/apm_project_page/">Project Page</a>
</p>

<p align="center">
  <img src="assets/arch.png" alt="Rotating Features for Object Discovery" width="600"/>
</p>



<!-- ### [Paper](https://arxiv.org/pdf/2410.20535) | [Openreview](https://openreview.net/forum?id=7Ye12RLZ4P&referrer=%5Bthe%20profile%20of%20Rajat%20Modi%5D(%2Fprofile%3Fid%3D~Rajat_Modi1))| [Blog](https://rajatmodi62.github.io/2024/10/26/hinton_apm-copy/) | [Project Page](https://rajatmodi62.github.io/apm_project_page/) |  -->

This repository contains the code for the
paper [Asynchronous Perception Machine For Efficient Test Time Training](https://arxiv.org/pdf/2410.20535) by Rajat Modi And Yogesh Singh Rawat

Our proposed Asynchronous Perception Machine represents a new way to do machine perception: i.e. asynchronous perception. This involves processing patches of an image one at a time in any order, and still encode semantic awareness in the network. This helps us in moving towards architectures which consume less flops and occupy less on-device-memory, and predict almost same features that a transformer predicts. This also allows us to achieve strong performance on test-time-training benchmarks. 

This is the public official release of our model and coco checkpoints, and we urge people across the world to try some more of GLOM's ideas. We will add more code here as we make progress. 


## Setup

- Install [conda](https://www.anaconda.com/products/distribution)
- Install Pytorch 
We used version 1.13.0, an A6000 gpu on Ubuntu 22.04 However, our codebase is pretty simple, and should remain robust to library changes in future, since it contains minimal dependencies. 


- Run the download script to download the checkpoints and the coco dataset (validation set):

```bash
bash download.sh
```


## Run Experiments

- Visualize semantic clusterings on the coco val set. Note that the model was trained on coco train set 

```bash
python visualize_coco.py
```

- Visualize islands of agreement on any image in the wild.

```bash
python predict_test_image.py
```

- Interpolate between any two images in the wild. A similar result was shown in the GAN paper, and diffusion too. We can do such interpolation in the MLP now. 

```bash
python interpolate.py
```

- One sample learning, which is used in test-time-training. This illustrates the ability of the APM to learn on a single CLS token distilled from a teacher, for eg, CLIP. 

In practice, we observed that a higher-parameterized teacher leads to higher-performance. 

```bash
cd single_token_segmentation
python train_tta.py
```

Please follow the installation setting of the original clip repo to run this particular part of the code. You can find those installation instruction [here](https://github.com/openai/CLIP).


- Computational Analysis

```bash
cd flop_analysis
python count_flops.py
python count_memory.py
python count_parameters.py
```

This should yield the same numbers as the computational analysis table i.e. Table 4 in the APM paper.

- Scaling Up experiments on COCO dataset

The above code is pure inference on COCO. Here we share the training code on the COCO dataset. Basically, we first dump features from Dinov2 backbone on coco-train set. 

```bash
cd misc_scripts
python resize_coco_images.py
python 1_extract_coco_features.py
python train.py
```

## Islands of Agreement

We illustrate that the idea of islands of agreement in the GLOM paper actually works. The below video has been shared **with permission** from Geoffrey Hinton. 

<div align="center">
<img src="assets/island_hinton.gif" alt="Hinton's Islands of agreement" width="600" height="300">
</div>

To plot similar islands for any image in the wild, please follow the proper steps [here](https://github.com/rajatmodi62/OccludedActionBenchmark/tree/main?tab=readme-ov-file#glom-hintons-islands-of-agreement)



## Citation

When using this code, please cite our paper:

```
@article{modi2024asynchronous,
  title={Asynchronous Perception Machine For Efficient Test-Time-Training},
  author={Modi, Rajat and Rawat, Yogesh Singh},
  journal={arXiv preprint arXiv:2410.20535},
  year={2024}
}


@article{modi2024apm,
  title={Rotating Features for Object Discovery},
  author={Modi, Rajat and Rawat, Yogesh},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```

## Contact

For questions and suggestions, feel free to open an issue on GitHub or send an email
to [rajatmodi62@gmail.com](mailto:rajatmodi62@gmail.com). I will try to get onto it whenever i get time. 


## Acknowledgements

This achievement reflects the collective effort of many brilliant minds and supporters, and we are deeply grateful for their contributions. We are greatly indebted to the contributions of Vonn Neumann to theory of self-reproducing automata and Alan Turing's Morphogenesis. 

We are grateful to Sindy Lowe's [work](https://github.com/loeweX/RotatingFeatures) for help in organizing our codebase.

There are many other conversations behind this work. A story best told in-person, respecting the mutual identities of everyone involved and honoring academic tradition/professionalism.

- rajat