# Asynchronous Perception Machine For Efficient Test Time Training 

<p align="center">
  <img src="assets/arch.png" alt="Rotating Features for Object Discovery" width="600"/>
</p>

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


