# Unsupervised Attention Mechanism (UAM)


![Unsupervised Attention Mechanism](./img/unsuper_attention.png "Unsupervised Attention Mechanism")

 

Code for our paper: 

**"Constraining Implicit Space with MDL: Regularity Normalization as Unsupervised Attention"** 

by [Baihan Lin](http://www.columbia.edu/~bl2681/) (Columbia). 



For the latest full paper: https://arxiv.org/abs/1902.10658

All the experimental results and analysis can be reproduced using the code in this repository. Feel free to contact me by doerlbh@gmail.com if you have any question about our work.



**Abstract**

Inspired by the adaptation phenomenon of neuronal firing, we propose the regularity normalization (RN) as an unsupervised attention mechanism (UAM) which computes the statistical regularity in the implicit space of neural networks under the Minimum Description Length (MDL) principle. Treating the neural network optimization process as a partially observable model selection problem, UAM constrains the implicit space by a normalization factor, the universal code length. We compute this universal code incrementally across neural network layers and demonstrated the flexibility to include data priors such as top-down attention and other oracle information. Empirically, our approach outperforms existing normalization methods in tackling limited, imbalanced and non-stationary input distribution in image classification, classic control, procedurally-generated reinforcement learning, generative modeling, handwriting generation and question answering tasks with various neural network architectures. Lastly, UAM tracks dependency and critical learning stages across layers and recurrent time steps of deep networks.



## Info

Language: Python3, bash

Platform: MacOS, Linux, Windows

by Baihan Lin, Feb 2019




## Citation

If you find this work helpful, please try the models out and cite our work. Thanks!

    @article{lin2019constraining,
      title={{Constraining Implicit Space with MDL: Regularity Normalization as Unsupervised Attention}},
      author={Lin, Baihan},
      journal={arXiv preprint arXiv:1902.10658},
      year={2019}
    }

  

An earlier version of the work was presented at the IJCAI 2019 Workshop on Human Brain and Artificial Intelligence in Macau, China. See the slides [here](https://www.baihan.nyc/pdfs/IJCAI_RN_slides.pdf) (with only partial results in the arXiv above).




## Tasks

* Imbalanced MNIST task with FFNN and RNN
* OpenAI gym's LunarLander-v2 game with DQN
* OpenAI gym's CarPole-v0 game with DQN
* MiniGrid gym's RedBlueDoors with PPO
* Handwritten digit generative modeling with DRAW
* Facebook AI's bAbI reasoning task 18 and 19 with GGNN





## Requirements

* Python 3
* [PyTorch](http://pytorch.org/)
* numpy and scikit-learn



## Acknowledgements

For the MiniGrid, Generative modeling, and bAbI tasks, this repository built upon the following repositories.   Thank you!

- https://github.com/lcswillems/rl-starter-files
- https://github.com/chingyaoc/ggnn.pytorch
- https://github.com/chenzhaomin123/draw_pytorch

