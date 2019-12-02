# `helpful-bookworm`: Tackling AutoNLP

## Introduction

AutoNLP consists of automatically solving NLP problems with an approach that generalizes well to a variety of datasets and problems. This repository tackles the **text categorization problem**, which is the task of assigning free-text documents predefined categories. The solutions implemented here are designed to interface with the [AutoNLP Challenge](https://autodl.lri.fr/competitions/35#home) on Codalab.

## Initial Research

I chose to explore this particular problem in AutoDL because I am currently researching methods for state-of-the-art MLC (multi-label classification) in NLP. I quickly realised that this challenge is *not* a multi-label problem (each documents belongs to precisely one category). Still, my intuition was that models that perform well on MLC will be able to handle text categorization pretty well.

I began my research on the [most recent paper I read on MLC](https://arxiv.org/abs/1904.08049). The proposed model is quite novel and not yet included in most libraries, which is why I used the listed baselines as a base to discover more models. In the end, I shortlisted a few models and papers that I thought were promising:

* A simple embedding followed by a multi-layer perceptron model that obtained excellent performance in MLC tasks according to the [original paper](https://arxiv.org/abs/1312.5419).

* A model that encodes inputs using an RNN and them feeds the encoding into a fully connected layer to predict the classes. It is based on [a Seq2Seq approach to MLC](https://papers.nips.cc/paper/7125-maximizing-subset-accuracy-with-recurrent-neural-networks-in-multi-label-classification), with the notable difference that we only need to predict a single label after encoding, so we can do away with the decoder and use a standard neural network.

* Finally, a [transformer](https://arxiv.org/abs/1706.03762)-based model that encodes the input text with a transformer stack and feeds the encoding to fully-connected layer for predictions. I choose to base the model on [BERT](https://arxiv.org/abs/1810.04805) by Google because it provides pretrained versions in both English and Chinese.

In the end, I chose to not implement the RNN encoder model due to time constraints and the belief that the BERT model would outperform it.

## Implementation

All the models were implemented using Keras with the TensotFlow backend.

The first model, which I call `emb+mlp`, used the FastText embeddings that were provided by the challenge organizers. I tested multiple architectures, but the best performance was achieved with a single hidden layer with 1000 hidden units. Both Adagrad and Adam optimizers were tested, with no noticeable differences in performance between them.

The BERT-based model, which I call `bert`, used the 12 layer and 768-dimensional output versions of BERT, opting for the uncased version when applicable. The pooled output of the BERT encoder was fed into a single fully-connected with softmax activation to predict each class. The choice of the optimizer, as well as most other decisions, was based on the implementation of BERT-based classifiers provided by Google Research on [their Github](https://github.com/google-research/bert). I noticed that the preprocessing pipeline that transforms the texts to BERT inputs has an impact on performance. As a result, the preprocessed datasets are cached on the implementation.

## Results

The `emb+mlp` model performed well when the number of categories was small: on the third provided dataset, which was a binary classification problem, the model achieved ~92% accuracy after only one epoch. However, in datasets in which there were multiple labels, the performance of the model was insufficient. Its training performance curve on the second dataset (which has 20 categories) is shown in Image 1. As we can see, the model underfits the data.
The `bert` model was much more successful at achieving excellent performance in even complex data. However, it is much bigger and fails to make a quick first prediction, which results in reduced overall scores. Image 3 displays the training performance of the model in the second dataset (the same one that emb+mlp underfitted).

## Interpretation of Results

Upon revisiting the paper on which the `emb+mlp` paper was based, I realized the importance of their choice of the loss function. They used the Pairwise Ranking Loss, which turns the task of determining the valid labels of an instance into a binary problem for each label (which is perfectly logical since the original problem was MLC). As result, the model proposed by the paper has excellent performance when *deciding a series of binary decisions*, which does not necessarly translate to reasonable performance on *deciding within a set of possible choices*. The experimental results back this finding: the model had excellent performance on binary classification and poor performance when the number of labels increases. In the end, the intuition that this model would be the right choice for the challenge at hand did not lead to good results.

The results of the `bert` model were expected. It has proven several times to achieve excellent performance when fine-tuning to the task at hand. I also expected that the model would be slow to train and test, which was the case. BERT was also proven to be too big for applications in which computational resources or time constraints are limited (such as in edge devices).

## Possible Improvements

Possible improvements to the current strategy are listed below:

* **Use smaller BERT models**: smaller BERT models such as ALBERT or Hugging Face's DistillBERT might be able also to achieve excellent performance while benefiting from reduced training/inference time cost.

* **Use ensemble methods**: the `emb+mlp` model is lightweight and achieves excellent performance in binary classification. It may be worthwhile to use it for this particular case and switch to a more robust model when the number of classes becomes bigger. It also might be interesting to develop light models to make a decent and fast first prediction and then switch to more complex models for the remaining of the challenge.

* **Reduce preprocessing overhead**: on the proposed implementations, I already moved as much of the preprocessing I could to the model creation, but there is still a significant overhead when preprocessing the datasets, especially in `bert`. Implementations of the preprocessing functions in other languages (such as C) might result in a considerable speed-up and improved performance.

* **Develop an scheduling strategy**: when comparing the training curves obtained by the ones of the leaderboards and me, I noticed that the predictions of the best models are made in (seemly) regular intervals, as opposed to my results. This finding leads me to believe that the leaders were using a strategy where the number of epochs/batches that the models train on before making a prediction grows exponentially. I also noticed that by training for an epoch before making a prediction, I am at a disadvantage on large datasets in which a reasonable prediction can be made before iterating over all training instances. A good per batch scheduling strategy would improve on these shortcomings.

## Conclusion

I was able to test both a simple model leveraging the power of embeddings - `emb+mlp` - and a complex model that fine-tunes a big transformer network - `bert`. Initial results show that `emb+mlp` is not powerful enough to handle datasets with multiple categories and that `bert` is too bulky to deliver good predictions quickly. Future improvements hinge on using ensemble methods to get the best of both worlds and also experiment with smaller BERT-like pretrained transformer models. Also, for the sake of optimizing score in the competition, it might be interesting to reduce preprocessing overhead and develop a scheduling strategy.
