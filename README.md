# Multitask-Learning-With-Enhanced-Modules
In multitask learning(MTL) paradigm, modularity
is an effective way to achieve component and parameter reuse
as well as system extensibility. In this work, we introduce two
enhanced modules named res-fire module(RF) and dimension
reduction module(DR) to improve the performance of modular
MTL network – PathNet. In addition, in order to further improve
the transfer ability of the network, we apply learnable scale
parameters to merge the outputs of the modules in the same
layer and then scatter to the next layer. Experiments on MNIST,
CIFAR, SVHN and MiniImageNet demonstrate that, with the
similar scale as PathNet, our architecture achieves remarkable
improvement in both transfer ability and expression ability. Our
design used x5.23 fewer generations to achieve 99% accuracy
on a source-to-target MNIST classification task compared with
DeepMind’s PathNet. We also increase the accuracy of CIFAR-
SVHN transfer task by x1.9. Also we get 70.75% accuracy on
miniImageNet.

RF and DR Modules 

  ![alt tag](https://github.com/Wind-Wing/readme_images/blob/master/enhanced_modules.png)
  
Scale Parameters

  ![alt tag](https://github.com/Wind-Wing/readme_images/blob/master/scale_parameters.png)
