---
layout: post
title: "AI Safety Fundamental Final Project"
date: 2024-09-29
---


## Investigating Local circuit in the feature basis


This post is part of the Final Project for the AI Safety Fundamentals Course, which I recommend to anyone who's interested in AI Safety.


This project has spanned 1 month and the final product is an implmementation of various techniques used Mechanistic Interprtability, with the goal intent of finding local circuits in Language Models.


This project started with a grandiose objective; **To investigate multi-token circuits in the wild**

*In this context Multi-Token refers to, circuits that explain a behavior over multiple tokens, eg. A neuron fires in 3 consecutive tokens, for US States., not just to circuits that span multiple token positions*



This initial effort to accomplish such ambitious research was  frustrated by the technical difficulties of getting the most basic set up to work.

Despite the hardships, I've carried trough with the project to the best of my abilities and I've accomplished ~60% of the initial goal.


At the end the project, was mostly unchanged but with slight modifications make it feasible with the available time.


**For a complete technical explanation of the methods, visit the following website**





# Table of Contents
1. [Motivation](#motivation)
2. [Introduction](#introduction)
3. [Relevant Definitions](#definitions)
4. [Methods](#methods)
5. [Results](#results)
6. [Frustration Dump](#frustrations)



### Motivation

It's difficult to provide a reason that led me to choose this Project, as the public shareable result for the AI Safety Fundamentals Course.

From the beginning of the project, I was aware of the difficulty of the endeavor I was undertaking. 

Whether I was being naive or courageous, I had the profound conviction that doing the hard thing was important.

To be brief and stop yapping, the reason that I chose this project was because I think Interpretability is important.

**¡¡¡I think that interpretability is important!!!** And the state of the world is that we've gone from a barely useful GPT-3 to O1 in less than 3 years.

**¡¡¡I think Interpretability is important !!!** And it is a good research agenda to leverage, the driving factors of the growth in capabilities **Scale, Search and Synthetic Data**


**¡¡¡I think Interpretability is important !!!** And I want you to show you why. 




### Introduction


Mechanistic Interpretability is a field of Machine Learning that is concerned with understanding, what internals of Neural Networks.

Most of the MI literature is focused in a certain kind of Neural Network, the Langugage Model, which is behind popular products like chat-gpt.

Most of the Language Models in the present are based on the Transformer which is a NN architecture introduces in 2017.

In the simplest terms Transformers are composed of stacked Transformer Layers, which in turn are composed of Attention Layers and MLPs.

All the layers are connected by a Residual Stream which can be thought of a large shared communication line, which enables trans-layer communication.

- Attention Layers are composed of Attention heads, and can be thought of as components that 
    1) Select information in the context to attend to.
    2) Extract information from the context.
- The MLP layer is composed of neurons, that further process information present in the residual stream  (including the information from the Attention Layer)

Despite the simple nature of the elements that compose the transformer, understanding even the most simple model behavior proves to be extremely challenging. 

Many reasons are behind of the difficulty of understanding a model behavior from it's internals, but at high level I can name a few.

- The sheer number of components,a amounting to millions of individual neurons.
- Neuron superposition and polysemanticity.
- Tradeoff between explanation length and accuracy.
- Model's sensibility to changes in the input.


Neuron polysemanticity and superposition are, two different but strongly related hypothetical properties of Neural Networks, feature superposition is a claim grounded in theory (aka Johnson-Lindenstrauss Lemma) which states that there a model can represent more features than neurons have.

*The definition of a feature is somewhat contested but I will role with "A feature is a property of the input"*


Neuron polysemanticity is a observed phenomena in which a neuron fires in seemingly unrelated context.

Neuron polysemanticity was a major problem in mechanistic interpretability, holding back the field.


Hopefully neuron superposition has been partly adressed in the last year with the introduction of dicitonary learning techniques, mainly Sparse Autoencoders.


The basic principle of sparse autoencoders is that, we can train a model with the simple objective of reconstructing the activations of a Neural Networks, and by applying simple constrains like sparsity and an overcomple basis we end up with a model that can reconstruct activations into sparse and Monosemantic features, that are also human interpretable.

For this investigation I leverage Sparse Autoencoders and Transcoders to investigate local circuits in GPT small.


### Relevant Definitions

Mechanistic Interpretability has plenty of jargon and concepts that are difficult to grasp without context, unfortunately I don't think that the ease of understanding that comes from familiarity with the field can be replicated with good or exhaustive definitions.

Hence I will ask the unfamiliar reader to trust their intuition for what things mean, and the reader who's familiar with prior literature to trust my understanding of the concepts, and roll with it.


**Neural Network Activations**

Activations are the intermediate computations of the model components. 

We can access the activations of a transformers with multiple levels of granularity, the important takeaway is that for each token position and each layer we have:
- A resiual stream (d= 768)
- An attention layer input and output (d = 768) 
- The MLP layer with input and output (d = 768)

At each layer the outputs of the MLP and Attn Layers are added to the residual stream.


**Sparse Autoencoders**

Sparse Autoencoders (from now on SAEs) are a dictionary learning technique, that consists of training a model to reconstruct model actiavations (the location can be the residual stream, the Attn Output or the MLP output). By imposing a constraint on sparsity and ussing an overcomplete basis we obtain a set of sparse, monosemantic and human understandable features.



**Transcoders**


Transoders (from now on TCs) are a variation of the SAE, with the difference that instead of reconstructing the activations from some location, TCs approximate the computation of an entire component (mostly the MLP layer).


The main difference between the MLP and a TC trained to behave like an MLP is that the TC is trained with and overcomplete basis wrt the original MLP and sparsity is enfored in the same way as in SAEs.

The sparsity enable us to treat the TC as linear, since the sparsity enforeces a kind of input independence.


**Features**


Features are properties of the input, like the language of a prompt, the tone, or wheter or not a US State has been mentioned.


Features are mostly believed to be linear (what does linear mean in this context is a rabbit hole by itself), in the sense that a feature is represented by a direction in activation space (high dot product in presence of the feature) 

In the context of Mechanistic Interpretability work that uses SAEs or it's variations, the features are given by the SAEs.

*If we reconstruct an activation with a SAE and the n-th SAE-neuron fires, we say that the activation contain's the n-th feature from that SAE*

In this project we use SAEs for the attention output and TCs for the MLP output, this defines a set of possible features that could be active for a given input.


$$
\text{Set of Features} = \bigcup_{l=0}^{layers}[F_{MLP}^{l}\cup F_{Att}^{l}]
$$




**Circuits**

The term circuit has been used since the start of the field.
The simplest definition of a circuit is that it is a composition of features.

Circuits are defined with respect to a task or model behavior that we want to understand.

A circuit is a subgraph of the model's computational graph, a good circuit must be faithfulness and complete.

The nodes are the various model components across layers and positions.



**Local Circuits**

Local circuits are a kind in which the root node is not the unembeding.


**Circuits in the Feature Basis**

Circuits in the Feature Basis are subgraphs of the model's computational graph, when we apply use dictionary learning techniques in the model's forward pass.

Hence the node of a circuit in the Feature Basis are Features, rather thatn model components.




### Methods


In this project we investigated local circuit in GPT2 small, mainly:

We investigate why do certain features fire, concretly we investigate why MLP features in the 5-th layer in GPT2 small fire.


To do so we use attribution techniques to discover which upstream features are more influential fore the fireing of a given target feature. 

Concretely we use Hierarchical Attributio, a technique introduced in "Automatically Identifying Local and Global Circuits with Linear Computation Graphs".


Hierarchical Attribution differs from direct attribution because it detaches unimportant nodes in the backward pass, instead of after the backward pass.

The use of Attn Output and MLP Transcoders allow us to treat the computational graph as a linear graph (if we freeze the Attention Pattern)


Once hierarchical attribution has been performed we can use the OV and QK circuits to get the edges between the feature nodes resulting from the hierarchical attribution.


### Results



### Frustration Dump

**I've rushed to write this section,for the reader take into account that most of my complains are just skill-issues.**




