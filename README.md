# noise2self
self supervised image denoising

noise2self is a self supervised image denoisation method developed by:
https://arxiv.org/abs/1901.11365

This is a pytorch implementation of the algorithm. This algorithm uses the DnCnn network for the training process. Here is the link to the DnCnn:
https://arxiv.org/pdf/1608.03981.pdf

noise2self.py is the training code of the project.  

Model2.pt: trained on a subset of the COCO dataset

model-mnist.pt: trained on the MNIST dataset

Bernoli, Gaussian, and Poisson noises were added to the raw images


![Alt text](noisy.jpg?raw=true "noisy image 1")
![Alt text](out.jpg?raw=true "refined image 1")

PSNR has been improved from 9.49 to 16.30


![Alt text](mnist-noisy.jpg?raw=true "noisy image 2")

![Alt text](mnist-out.jpg?raw=true "refined image 2")

PSNR has been improved from 6.7  to 9.4



