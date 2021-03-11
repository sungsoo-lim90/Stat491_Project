# Listening to Chaotic Whispers: A Deep Learning Framework for News-oriented Stock Trend Prediction

This guide explains the main ideas in the paper by Hu and others 2019. For the source code, the one given on the list was for the Statsnet, which provides the dataset for our code implementation. The source code was not available from the authors. Thus, we used the following code:

https://github.com/donghyeonk/han

This implementation does not include the Self-paced Learning mechanism, which will be dscribed shortly. This implementation also does not include the stock prediction strategy based on the classification of stock trends. For these sections, we refer to the original paper. 

## Introduction

Stock trend prediction is an interesting problem with many applications in finance. For instance, understanding how individual stocks trend will allow for maximizing profit from the stock investment (Hu et al. 2019). To this end, the authors propose a method based on the human learning processes: sequential context dependency, diverse influence, and effective and efficient learning. 

Sequential context dependency refers to the ability of human investors to implement and combine a variety of news sources on a single stock into a unified context before making a decision. Thus, to mimic such human analytical process, the authors propose that the prediction framework should incorporate the news sources in a sequential, temporal context. 

Diverse influence indicates that some news sources have more influential and durable effects on the stocks' trends. Thus, these news sources should have more attention paid to them than those that do not have such characteristics. 

Effective and efficient learning indicates that human investors tend to first gain an overall knowledge with common occasions, and then turn to exceptional cases (Hu et al. 2019). Thus a learning mechanism should incorporate such discrepancies in the informativeness of the news sources, and conducts learning on more informative news at the earlier stage, and further optimize to tackle harder samples (Hu et al. 2019). 

The guide is structured as follows: we first describe how the first two strategies are incorporated into the Hybrid Attention Network (HAN) proposed by the authors. We describe the self-paced learning mechanism. Then, we describe the learning of the network on the Statsnet dataset, with a focus on the description of the data and the learning curves. Lastly, we describe the stock trading strategy based on stock trend classifications as proposed by the authors. 

## HAN - Hybrid Attention Network

Hybrid Attention Network (HAN), as proposed by the authors, have the ability to predict a stock's trend as a classification problem. For the Statsnet dataset, the classification problem can be divided into the following two classes: DOWN, UP. To classify into these two classes, the rise percentage at a given time point is calculated, where rise percent(t) = [open price(t+1) - open price(t)]/ open price(t)

### News-level Embedding
<img src="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;u_{ti}&space;&=&space;\text{sigmoid}(W_nn_{ti}&space;&plus;&space;b_n),&space;\\&space;\alpha_{ti}&space;&=&space;\frac{\text{exp}(u_{ti})}{\sum_j&space;\text{exp}(u_{tj})},&space;\\&space;d_{t}&space;&=&space;\sum_{i}\alpha_{ti}n_{ti}&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;u_{ti}&space;&=&space;\text{sigmoid}(W_nn_{ti}&space;&plus;&space;b_n),&space;\\&space;\alpha_{ti}&space;&=&space;\frac{\text{exp}(u_{ti})}{\sum_j&space;\text{exp}(u_{tj})},&space;\\&space;d_{t}&space;&=&space;\sum_{i}\alpha_{ti}n_{ti}&space;\end{align*}" title="\begin{align*} u_{ti} &= \text{sigmoid}(W_nn_{ti} + b_n), \\ \alpha_{ti} &= \frac{\text{exp}(u_{ti})}{\sum_j \text{exp}(u_{tj})}, \\ d_{t} &= \sum_{i}\alpha_{ti}n_{ti} \end{align*}" /></a>

- Python 2.7.11
- Tensorflow 1.4.0
- Scipy 1.0.0
- NLTK 3.2.5

## Directories
- src: source files;

## Running
