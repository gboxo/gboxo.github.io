---
layout: post
title: "Introduction to mechanistic Interpretability"
date: 2023-05-19
categories: ai-safety
---

In the second post on my blog, I will present a brief introduction to the main goals and methods of Mechanistic Interpretability (MI), as well as its history.

### What is Mechanistic interpretability?

***Mechanistic interpretability is a field of Machine Learning that tries to understand what's happening inside the large matrix of weight of a neural network.*** When a model is very big and difficult to interpret is sometimes called a *black box*.The line of reasoning is as follows, when you prompt a token to a model such as GPT, or give a piece of text to classify to BERT, after some (many) computations are made an output is given, but the exact criteria followed by the model is essentially unknown.

In contrast a simple linear model performing regression of say wages over age is easy to interpret and the underlaying computations are easy to follow, Most of the time when dealing with linear model, the relations found are easy to interpret and strick us as logical and funded in common sense, specially when prior knowledge is hold over the task.

Basically the 2 framework to which MI has been applied are image generation and text generation.

Why isn't interpretability straightforward? After all as much as this generative models are huge and weird, they are also transparent in the sence that we can know at each time which weights the components have. The basic obstacle to easy interpretation of this models, are *non linearities* and the *huge dimesionality* of the model.

One of the reason NN has been so succesuful is that by using non linearities (such as layer norm and non linear activation functions) the model is able to extract hidden features of the input.

### Why does it matter at all?

-   There are several reasons that somewhat justify thhis line of research this are, and not limited to:

    -   Building trust in the mode

        -   As NN based models penetrate in our daily life, and as such are integrated in the decision making process the standards in regard to their accuracy and specificity are increased.

        -   For example if a model is used to detect the beggining phases of cancer, it is disireable that the model not only gives a TRUE/FALSE answer, but a range of metrics such as confidence intervals based on prior knowledge or iven a \*\*human interpretable\*\* explanation of the output.

        -   NOTE: A model being able to given confidence intervals (something like a Bayesian model) or being able to explain himself, are not really part of the MI field, but are easy to understand and gives a sense of "Why is a \*black box\* bad and having insights on the inference process of the model is good"

    -   AI security

        -   Ethical pen-testing

        -   Deception detection

        -   Prediction of capabilities development

        -   White-boxing bootleneck

    -   Capabilities

        -   It's easy to envision a point in which understanding the inner dynamics of for example a Transformer model, would help on implementation via training dynamics (traininig phases, training dataset design, etc) pre define neurons or pretrained aggregable circuits.

***In this Post we will focus of Mechanistic Interpretability of transformer-based models***

### Historical contentxt

The field of mechanistic interpretability of NN models, began somewhere on the late 2010, when sets of neurons in CNN image classification networks, where found to "detect curves".

In regards to the interpretation of Transformer networks one of the first paper was *A Mathematical Framework for Transformer Circuits* that proposes several techniques to reverse engineer 0,1,2 attention only layer models, and some intuitions to reason about this objects.

Since then several papers has been published from the big actors (Red Wood Research, Antropic, etc) and others from independent researchers.

The main findings and contributions, to mention some had been:

-   The discovery of induction heads
-   The discovery of the Indirect Object Identifier Circuit
-   Advancement on Causal Intervention Techniques
-   Advancement on the Open Source infrastructure for MI (see NeelNanda TransformerLens)

This findings will be object of future posts.

### What is the object of study in M.I

While the end goal of M.I is to be able of extract human interpretable strucutres of LLM such as GPT4, at this current point a framework for studying such models don't exist.
To get around the actual limitations that make impossible full on interprate LLM, some alternatives had been proposed.

Depending on the objectives of the ***Intrepretation experiment*** there are 3 main types of model that can be somewhat easy interpreted:

-Toy models that perform some (human interpretable) algorithmic task such as modular adition (see the Groking paper)
-Small models with or witout MLPs (multilaye perceptrons) that have 0 to 6 layers.
-Later gen models (such as GPT2)

### Some techniques being used

A brief non exhaustive introduction of the methods being used in M.I can be divided in 2 categories:

-Causal Intervention:

    -   Ablation of some component of the model: This referse to the modification of the activation whether is setting it to zero, performing the mean, or sampling over activation. The goal is to see how the model performance changes when an ablation is performed.
      - Causal Scrubbing: It's an algorithmic technique, to automate causal interventions in vise of finding Circuits

-Metrics and other exploratory tools:

    - Logit difference
    -  Max activation