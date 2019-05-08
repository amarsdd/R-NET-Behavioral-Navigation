# Finding a High-Level Plan for Behavioral Robot Navigation with R-NET

In this work, we utlize a modified R-NET with gated attention and 
self-matching attention to develop an understanding of the behavioral 
navigational graph to enable the pointer network to produce a sequence 
of behaviors representing the path for robot navigation. The source code 
for data preparation is adapted from [this](https://github.com/StanfordVL/behavioral_navigation_nlp) 
tensorflow implementation of 
[Translating Navigation Instructions in Natural Language to a High-Level Plan for Behavioral Robot Navigation](https://arxiv.org/abs/1810.00663).
The source code for R-NET is adapted from [this](https://github.com/YerevaNN/R-NET-in-Keras) 
Keras implementation of 
[R-NET: MACHINE READING COMPREHENSION WITH SELF-MATCHING NETWORKS](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)

![](images/rnet.png)

## Getting Started

The work is trained and tested on the behavioral navigation graph dataset from [Stanford](http://follow-nav-directions.stanford.edu/).
In case natural language directions are used, [GLOVE embeddings](https://nlp.stanford.edu/projects/glove/) is also required.
All these necessary dataset and embedding are available in "data" folder. Please clone the entire project.
```
git clone https://github.com/amarsdd/R-NET-Behavioral-Navigation.git
```

### Prerequisites

The code has only been tested on python 2.7.16 with the following requirements

```
matplotlib==2.2.4
six==1.12.0
tensorflow_gpu==1.13.1
pandas==0.24.2
nltk==3.4
tqdm==4.31.1
Keras==2.0.6
numpy==1.16.2
tensorflow==1.13.1
```


## Training

Run the following to train the model with default configurations:

```
python train.py
```
The following are the set of arguments that can be provided:
```
('--with_instruction', default=False, help='Use instruction or not', type=int)
('--hdim', default=100, help='Number of units in BiRNN', type=int)
('--nlayers', default=3, help='Number of layers in BiRNN', type=int)
('--batch_size', default=128, help='Batch size', type=int)
('--nb_epochs', default=50, help='Number of Epochs', type=int)
('--optimizer', default='adam', help='Optimizer', type=str)
('--lr', default=None, help='Learning rate', type=float)
('--dropout', default='0.0', help='Dropout', type=str)
('--name', default='Rnet_navigation', help='Model dump name prefix', type=str)

('--data_dir', default='data', help='Data directory', type=str)
('--exp_dir', default='experiments', help='Experiment results directory (Model checkpoint and Tensorboard logs)', type=str)

```


## Testing

Run the following to evaluate the model with default configurations on both
test sets:

```
python evaluate.py
```
Trained models for various configurations are in 'trained_models' folder. Given
the following arguments the trained model is selected.
The following are the set of arguments that can be provided:

```
('--with_instruction', default=False, help='Use instruction or not', type=int)
('--hdim', default=100, help='Number of units in BiRNN', type=int)
('--nlayers', default=3, help='Number of layers in BiRNN', type=int)
('--batch_size', default=128, help='Batch size', type=int)
('--nb_epochs', default=50, help='Number of Epochs', type=int)
('--optimizer', default='adam', help='Optimizer', type=str)
('--lr', default=None, help='Learning rate', type=float)
('--dropout', default='0.0', help='Dropout', type=str)
('--name', default='Rnet_navigation', help='Model dump name prefix', type=str)

('--data_dir', default='data', help='Data directory', type=str)
('--exp_dir', default='experiments', help='Experiment results directory (Model checkpoint and Tensorboard logs)', type=str)
('--model_dir', default='trained_models', help='Trained Model directory', type=str)
```

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.


## Authors

* **Amar Shrestha**  amshrest@syr.edu


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
