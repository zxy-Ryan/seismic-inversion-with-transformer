# Blind Inversion Transformers

Implementation of Transformer model (code reviced from [Transformers for Time Series](https://github.com/maxjcohen/transformer)) 
applied to seismic blind inversion (Powered by [PyTorch](https://pytorch.org/)).

## Transformer model

Transformer are attention based neural networks designed to solve NLP tasks. Their key features are:

- linear complexity in the dimension of the feature vector ;
- paralellisation of computing of a sequence, as opposed to sequential computing ;
- long term memory, as we can look at any input time sequence step directly.

This repo will focus on their application to seismic blind inversion.

## Dataset and application as metamodel

We created a dataset by marmousi dataset and ricker wavelets with different frequency and phase. Using seismic convolution model, we got sythetic seismic data. We then convert these data in csv format, and feed it to the transformer network.

## Installation

All required packages can be found in `requirements.txt`, and expect to be run with `python3.7`. 

## Usage

### Running training script

Using python IDE, run the default `training.py` or `testing.py`.

