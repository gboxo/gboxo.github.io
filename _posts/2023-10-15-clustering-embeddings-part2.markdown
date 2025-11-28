---
layout: post
title: "Word Embeddings: A Comprehensive Guide Part 2"
date: 2023-10-15
categories: ai-safety
---

# Word Embeddings: A Comprehensive Guide Part 2

In this 2nd part of our series, we delve into unsupervised techniques for Named Entity Recognition (NER). As NER advances, unsupervised methods are paving the way for more efficient and versatile applications.

## Towards Unsupervised Named Entity Recognition

### Table of Contents

- [Word Embeddings: A Comprehensive Guide Part 2](#word-embeddings-a-comprehensive-guide-part-2)
  - [Towards Unsupervised Named Entity Recognition](#towards-unsupervised-named-entity-recognition)
    - [Table of Contents](#table-of-contents)
    - [Introduction](#introduction)
    - [Background](#background)
      - [What is Named Entity Recognition (NER)?](#what-is-named-entity-recognition-ner)
      - [Challenges in Supervised NER](#challenges-in-supervised-ner)
    - [Current Approaches](#current-approaches)
      - [Cycle NER](#cycle-ner)
      - [SSL could avoid supervised learning](#ssl-could-avoid-supervised-learning)
      - [Proposed Approach](#proposed-approach)
    - [Applications](#applications)
      - [Security](#security)
      - [Interpretability](#interpretability)
      - [Efficiency](#efficiency)
    - [Potential Improvements and Future Directions](#potential-improvements-and-future-directions)
    - [Conclusion](#conclusion)

---

### Introduction

There exists a core conflict in the field of NLP; this conflict is often known as *disambiguation*.

Words and simple phrases are not always monosemantic, and their meaning can change depending on the context and intentions of the actors. How can we deal with the polysemy of words in real-world texts? NER is often used to categorize words in a corpus into specific categories. Current NER techniques are heavily dependent on vast amounts of data and are domain-specific. A better ontology of the tokens that constitute the vocabulary space of a language model can aid in core tasks like security, interpretability, and efficiency.

A possible path to achieving this is proposed in the following articles. The code and technical details are available upon request via email.

### Background

#### What is Named Entity Recognition (NER)?

NER stands for Named Entity Recognition. It's a sub-discipline of the broader field of Neural Language Processing that tries to identify the entities or categories within a given text.

The categories are usually Person, Organization, Location, etc.

Nowadays, NER is usually done by fine-tuning an existing LLM like BERT or RoBERTa to classify tokens or parts of the text into several categories. This process is done by providing an extensive training corpus where the words are annotated with the category they belong to.

This labeling or annotation is usually done manually.

**It's important to note that with the inception of powerful generative models, it is now possible to create custom annotated datasets, an example of this is the Universal-Pile-NER.**

Even though frontier models like GPT-4 are completely capable of performing NER on any text, several limitations make this approach not feasible in all cases.

The main limitations are the following:
  - Cost: The inference cost in terms of time and API request costs can be prohibitive.
  - List of entities: A list of entities must be provided, and sometimes this might not be an easy task.
  - Lack of access to the embeddings and hidden states of the model.
  - Context length: The most significant limitation is the limited context window that a model can access.

#### Challenges in Supervised NER

As previously mentioned, some of the limitations of current NER approaches include the need for a high-quality, domain-specific corpus to train on, and computational resources to perform the fine-tuning, which limits the use of NER in some domains.

One possible way of overcoming these limitations is to leverage the use of unlabeled data to either find partitions in the embedding spaces that resonate with some semantic or syntactic nature of the embedded tokens or to train a model to classify hidden states of a language model to determine context-specific meanings.

This iterative use of unlabeled data to minimize some objective function is commonly known as **Self Supervised Learning**.

### Current Approaches

#### Cycle NER

This work from Andrea Iovine et al. employs a technique called *Cycle-Consistency-Training* to train a NER algorithm by jointly training two components to minimize 2 functions.

- The first function tries to generate entities given a certain sentence.
- The second function tries to generate sentences given a list of entities.

In this approach, a pretrained language model (Google's T5) is used to generate the outputs of each function, and the output of one function is used as input for the other.

**Training Task**: 

```
Given S and Q, we define 2 subtasks:
- Sentence-To-Entity:For a sentence s in S, S2E outputs an entity sequence q'.This represents the NER task in cycleNER:
  - ("Australia",LOC),("Sri Lanka",LOC)
- Entity-to-sentence:For an input entity sequence q in Q, E2S generates an output sentenes containing the entities.
  - ("EU",ORG)("German",LOC)("",MISC)-->"Eu rejects German call to boycott British Lamb" 
```


#### SSL could avoid supervised learning

This approach is from a Medium Post and respective GitHub Repository by Ajit Rajasekharan.

This technique focuses heavily on partitioning the embedding space into a set of overlapping clusters, by minimally annotating a subset of BERT tokens, and then applying a clustering technique to attribute one or multiple clusters to each word in BERT's vocabulary space.

This generates a so-called *entity vector* for each token in the vocabulary that represents the strength of the association of each token to the clusters.

This approach later relies on the model's predictions to weight each token *entity vector* by the logits attributed to them. Finally, some summary statistics are used to annotate a given token with a cluster.

#### Proposed Approach

Limitations of generative approaches like Cycle NER:
  - Computationally intensive.
  - The set of labels must be provided.
  - The performance is heavily dependent on the dataset.

Limitations of embedding space partitioning approaches like SSL:
  - An initial list of annotated tokens must be provided.
  - The method is sensitive to very overlapping labels.
  - Winner-takes-all behavior; some majority cluster is usually promoted as the entity.
  - Tendency to label meaningless tokens as the majority cluster.

Since, for the applications I care about, a general attribution of labels to the whole token set is more important than fine-grained labeling of tokens in a very constrained context, a good model should prioritize partitioning the vocabulary space in a way that allows operation in a reduced dimensionality while maintaining the general sense of a sentence.


Building on the limitations of past approaches, we designed a protocol to partition the embedding space. This design aims to minimize the resampling within cluster Kullback-Leibler divergence on the distribution of logits for a specific prediction, both before and after resampling.

**Outline of the algorithm:**
  - Cluster the embedding space using Spectral Clustering.
  - Cache the logits of the model for a given corpus.
  - Resample 20% of the tokens from within cluster tokens.
  - Measure the KLD between the logit distributions.
  - Compute the gradients of the loss with respect to each edge weight (using SGD).
  - Update the weights.
  - Iterate for 3 epochs.

### Applications

#### Security

The field of ML security is burgeoning at an accelerated pace due to the advent of more sophisticated models. A secure model can be characterized as one that:
  - Does not propagate misinformation.
  - Protects private or sensitive information about individuals.
  - Refrains from sharing or promoting illegal or dangerous activities, among other considerations.

Limiting parts of the vocabulary space when a red flag classifier is triggered can guide prompt generation in a natural way, thus minimizing prompts like **"Sorry, as a large language model, I cannot..."**

#### Interpretability

Historically, Neural Networks have been perceived as "black box" models. This means that their internal computations aren't easily understood by researchers.

However, recent efforts have been made to improve the interpretability of these models. A significant challenge in current interpretability research is transitioning from specialized, narrow-distribution studies to broader, more generalizable research.

A comprehensive and automated understanding of the structure of prompts could assist in developing automated interpretability tools, similar to the ACDC technique.

#### Efficiency

Understanding the structure of a prompt can enhance the efficiency of a model's inference process. By narrowing down the distribution of tokens that the model needs to consider at specific stages of the generation process, it's possible to improve its performance and speed.

### Potential Improvements and Future Directions

*As of 15 October 2023, only a reduced version of this algorithm has been implemented due to a lack of computing power.*

### Conclusion

1) The embedding space can be clustered into semantically consistent clusters. 
2) Understanding the overall structure of a prompt can be useful for various tasks. 
3) Unsupervised NER always represents a trade-off between specificity and the coverage of the vocabulary space. 
4) Algorithmic improvements are necessary to make the algorithm more feasible. 
5) This method could be combined with Swap Graphs or ACDC to generalize interpretability work. 

