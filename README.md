# Listening to Chaotic Whispers: A Deep Learning Framework for News-oriented Stock Trend Prediction

This guide explains the main ideas in the paper by Hu and others 2019. For the source code, the one given on the list was for the statsnet, which provides the dataset for our code implementation. The source code was not available from the authors. Thus, we used the following code:

https://github.com/donghyeonk/han

This implementation does not include the Self-paced Learning mechanism, which will be dscribed shortly. This implementation also does not include the stock prediction strategy based on the classification of stock trends. For these sections, we refer to the original paper. 

## Introduction

Stock trend prediction is an interesting problem with many applications in finance. For instance, understanding how individual stocks trend will allow for maximizing profit from the stock investment (Hu et al. 2019). To this end, the authors propose a method based on the human learning processes: sequential context dependency, diverse influence, and effective and efficient learning. 

The guide is structured as follows: 

## HAN - Hybrid Attention Networks

### News-level Embedding
<img src="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;u_{ti}&space;&=&space;\text{sigmoid}(W_nn_{ti}&space;&plus;&space;b_n),&space;\\&space;\alpha_{ti}&space;&=&space;\frac{\text{exp}(u_{ti})}{\sum_j&space;\text{exp}(u_{tj})},&space;\\&space;d_{t}&space;&=&space;\sum_{i}\alpha_{ti}n_{ti}&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;u_{ti}&space;&=&space;\text{sigmoid}(W_nn_{ti}&space;&plus;&space;b_n),&space;\\&space;\alpha_{ti}&space;&=&space;\frac{\text{exp}(u_{ti})}{\sum_j&space;\text{exp}(u_{tj})},&space;\\&space;d_{t}&space;&=&space;\sum_{i}\alpha_{ti}n_{ti}&space;\end{align*}" title="\begin{align*} u_{ti} &= \text{sigmoid}(W_nn_{ti} + b_n), \\ \alpha_{ti} &= \frac{\text{exp}(u_{ti})}{\sum_j \text{exp}(u_{tj})}, \\ d_{t} &= \sum_{i}\alpha_{ti}n_{ti} \end{align*}" /></a>

- Python 2.7.11
- Tensorflow 1.4.0
- Scipy 1.0.0
- NLTK 3.2.5

## Directories
- src: source files;

## Running
