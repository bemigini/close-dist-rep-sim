# When Does Closeness in Distribution Imply Representational Similarity? An Identifiability Perspective
Code for experiments and plots for the article: ["When Does Closeness in Distribution Imply Representational Similarity? An Identifiability Perspective"](https://arxiv.org/abs/2506.03784) by Beatrix M. G. Nielsen, Emanuele Marconato, Andrea Dittadi and Luigi Gresele, accepted for NeurIPS 2025.   


## Description
Training models, calculating distances and generating plots can be done through run.py. See the "Usage" section  below for details.

Note that some of the code for creating plots requires trained models or pre-computed distances. 

Some checkpoints from models on synthetic data have been included as examples in the "checkpoints" folder. 

Some result files with computed distances and accuracies are in the "results.zip" folder. Remember to unzip before using these to make plots.  



## Installation guide

The outline of the installation is the following:

**1. Create and activate conda environment**

**2. Conda install relevant conda packages**

In 2. there might be differences depending on your machine and preferences. All relevant commands can also be found in conda_environment.txt. 

**1. Create and activate conda environment**

Use the commands:
```
conda create -n close-dist python=3.11
conda activate close-dist
```

**2. Conda install relevant conda packages** 

Install the relevant packages with the following commands:
```
conda install matplotlib docopt h5py scikit-learn tqdm scipy sympy pandas
```
If you are on a CPU only machine continue with:
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
If you are on a GPU machine use instead with the relevant cuda version:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```



## Datasets
We use [CIFAR-10](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html) loaded with the torchvision package.


## Usage

Configurations for training models are in the configs folder. There are three kinds of configs: 1. model config, which specify the model architecture, 2. dataset config, which has settings for the dataset to use for training, and 3. training config, which specifies the hyperparameters to use for training, e.g. learning rate. Classes for the configs are in the src/config folder. When using run.py to train a model paths to all three configs must be given. 

To see all options, use:
```
python run.py -h
```
To train models use:
```
run.py train-variations --output-folder=<file> --dataset-config=<file> --model-var-config=<file> --train-config=<file> [options]
```
Where output-folder is where checkpoints will be saved. 

For example:
```
python run.py train-variations --output-folder=checkpoints --dataset-config=configs/cifar10_0_cls10.json --model-var-config=configs/model_variations_config_resnetcifar10_128_fd3.json --train-config=configs/cifar10_0_32_ADAM_0_0001_20000steps.json --cuda
```
Will train a resnet classifier on CIFAR-10 with 3-dimensional representations for 20K steps. Training can be resumed from a saved checkpoint by using the date-str and continue-from options. For example:  
```
python run.py train-variations --output-folder=checkpoints --dataset-config=configs/cifar10_0_cls10.json --model-var-config=configs/model_variations_config_resnetcifar10_128_fd3.json --train-config=configs/cifar10_0_32_ADAM_0_0001_20000steps.json --date-str=2025-04-22 --continue-from=5000 --cuda
```
Will load a checkpoint of a resnet classifier on CIFAR-10 with 3-dimensional representations which began training on 2025-04-22 and has already trained for 5000 steps. Training of this model will continue until it reaches the 20K steps specified in the config. 

A template for a bash script can be found next to the run file.  

For calculating the distances between distributions and models use 
```
run.py get-distances --date-str=<string> --layer-size=<List> --num-classes=<int> --dist-type=<str> [options]
```
for the models trained on synthetic data and 
```
run.py get-distances --date-str=<string> --dataset-config=<file> --model-var-config=<file> --train-config=<file> --layer-size=<List> --num-classes=<int> --dist-type=<str> [options]
```
for the models trained on CIFAR-10. For example:
```
python run.py get-distances --date-str=2025-04-22 --dataset-config=configs/cifar10_0_cls10.json --model-var-config=configs/model_variations_config_resnetcifar10_128_fd2.json --train-config=configs/cifar10_0_32_ADAM_0_0001_20000steps.json --layer-size=128 --num-classes=10 --dist-type=max --weight=0.00001 --sum-or-max=max --num-samples=200
```
will measure distances between models from 2025-04-22 (date-str), trained on CIFAR-10 (dataset-config), using a resnet with 2-dimensional representations (model-var-config), trained for 20K steps (train-config), using a width of 128 for the unembedding network (layer-size), using 10 classes (num-classes), using max for each term in d_LLV like in the article (dist-type), using 0.00001 for the lambda parameter for d_LLV like in the article (weight), using the max over the terms for d_LLV like in the article (sum-or-max) and calculating using 200 samples of possible input sets (num-samples).  

For making the plots use 
```
run.py make-plot --plot-type=<string>
``` 
usig the relevant plot-type. See possible plot types in the "Plots" section below. 



## Plots
The plot types are 'cifar10_reps', 'd_LLV_constructed', 'd_LLV_train_synthetic', 'd_LLV_vs_width', 'loss_diff_vs_mcca', 'synthetic_data' and 'KL_table'.

'cifar10_reps' makes plots of all embeddings and unembeddings for the models we trained on CIFAR-10 with 2-dimensional embeddings. These show that when we train the same model with different seeds, we do sometimes get quite different representations, in the sense that the classes can be swapped around. In the article the plots are in the appendix F.5 "All Two-dimensional Representations of CIFAR-10 Models". It also makes the comparison of embeddings plot from the left part of figure 3. 

'd_LLV_constructed' makes a plot of d_LLV vs max d_SVD for some constructed models. In the paper, the plot can be found in the appenbdix F.6 "Illustration of Bound on Constructed and Trained Models", figure 16 (left). This illustrates that max d_SVD is larger for more different models, and shows how max d_SVD is always below the bound we find in the article. 

'd_LLV_train_synthetic' makes a plot of d_LLV vs max d_SVD for models trained on some synthetic data. In the paper, the plot can be found in the appenbdix F.6 "Illustration of Bound on Constructed and Trained Models", figure 16 (right). This illustrates that max d_SVD is larger for more different models, and shows how max d_SVD is always below the bound we find in the article.  

'd_LLV_vs_width' makes plots of the mean d_LLV vs the width used in the model before the representation layer and also the max d_SVD vs the width. We see that both d_LLV and d_SVD are smaller for wider models. In the article, this is figure 3 (right) for models with 4 and 6 classes and the appendix F.7 "Wider Models Have more Similar Distributions - Extra Plots" for 10 and 18 classes.   

'loss_diff_vs_mcca' makes the scatter plots which show that a small difference in test loss, does not guarantee that representations are closer to being linear transformations of each other measured with m_CCA. These plots are in the appendix F.4 "Loss Difference vs Embedding m_CCA", figure 5.   

'synthetic_data' makes a plot which illustrates the synthetic data we used. This plot can be found in the appendix F.2 "Models Trained on Synthetic Data", figure 4.  

'KL_table' generates the latex for the table which shows how we can make KL-divergence go to zero without making the embeddings more similar. This is table 1 in the article. 




## Acknowledgments

We thank Simon Buchholz for helpful discussions and feedback. Beatrix M. G. Nielsen was supported
by the Danish Pioneer Centre for AI, DNRF grant number P1. E.M. acknowledges support from
TANGO, Grant Agreement No. 101120763, funded by the European Union. Views and opinions
expressed are however those of the author(s) only and do not necessarily reflect those of the European
Union or the European Health and Digital Executive Agency (HaDEA). Neither the European Union
nor the granting authority can be held responsible for them. A.D. thanks G-Research for the support.
L.G. was supported by the Danish Data Science Academy, which is funded by the Novo Nordisk
Foundation (NNF21SA0069429). 

## License 

See LICENSE.




