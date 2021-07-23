# <p align="center">VisionX</p>


### <p align="center">PyTorch based repository for structured and modularised code for various Computer Vision models, for results/metrics, ease of use and out of the box experimentation : `work in progress`</p>

<a href="link" style="text-align: center">

<img src="https://i2.wp.com/metrology.news/wp-content/uploads/2020/11/AI-Based-Machine-Vision.jpg?zoom=2&resize=800%2C445&ssl=1" align="center"></a>

--------

### How to interpret the repository:

1) `Models` dir has all CV models, custom models are under `custom.py`.

2) `Utils` dir contains utility functions and components.

3) Extra resources like images or gifs would go under `resources` .

4) `Notebooks` dir will hold all Jupyter Notebooks for experiments and/or well as final runs for presenting results.


----

#### Interesting results/experiments:

  - 89K and 95K parameter models on Cifar10 with 85%+ validation accuracy - uses augmentation + `groups` param
  - 93% validation accuracy on Cifar10 with Custom ResNet model - uses augmentation + One Cycle LR
