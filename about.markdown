---
layout: single
title: "About Me"
permalink: /about/
author_profile: true
research_interests:
  - icon: "fas fa-search"
    title: "Mechanistic Interpretability"
    excerpt: "Reverse-engineering neural networks to understand their internal cognition."
  - icon: "fas fa-dna"
    title: "AI x Biology"
    excerpt: "Applying interpretable ML to protein engineering and drug discovery."
  - icon: "fas fa-shield-alt"
    title: "AI Safety"
    excerpt: "Ensuring advanced AI systems remain aligned and controllable."
education:
  - icon: "fas fa-graduation-cap"
    title: "MSc in Bioinformatics"
    excerpt: "Universitat Autònoma de Barcelona (UAB)<br>_Focus: Computational Biology, Machine Learning_"
  - icon: "fas fa-university"
    title: "BSc in Statistics"
    excerpt: "Universitat Politècnica de Catalunya (UPC), Barcelona<br>_Focus: Statistics, Data Science_"
---

Hello! I’m **Gerard Boxó**, an MSc graduate with an interdisciplinary background in Statistics and Bioinformatics based in Barcelona.

My work sits at the intersection of **Artificial Intelligence** and **Biology**. I am deeply invested in **Mechanistic Interpretability (MI)**, working to decipher the black-box nature of modern machine learning models—from Large Language Models (LLMs) to Biological ML systems.

This blog is a living document of my research journey where I publish experiments, analyses, and insights.

## Research Interests

<div class="research-interests-list" style="display: flex; flex-direction: column; gap: 25px; margin-bottom: 2em;">
{% for interest in page.research_interests %}
  <div class="interest-item" style="display: flex; align-items: flex-start; gap: 20px; padding: 15px; background: #f9f9f9; border-radius: 6px; border: 1px solid #eee;">
    <div class="interest-icon" style="flex-shrink: 0; width: 50px; text-align: center; color: #5a5a5a;">
      <i class="{{ interest.icon }} fa-2x"></i>
    </div>
    <div class="interest-content">
      <h3 style="margin: 0 0 5px 0; font-size: 1.1em;">{{ interest.title }}</h3>
      <p style="margin: 0; font-size: 0.95em; color: #666;">{{ interest.excerpt }}</p>
    </div>
  </div>
{% endfor %}
</div>

## Current Focus

<div class="notice--info">
  <h4 class="no_toc">Current Research</h4>
  I am currently focused on <strong>evaluating the capabilities of frontier models for autonomous protein design</strong>.
</div>

I have contributed to both **AI Safety** and **Computational Biology**, with past work including interpretability research applied to safety and protein engineering.

## Education

<div class="education-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 2em;">
{% for edu in page.education %}
  <div class="edu-item" style="padding: 20px; background: #fff; border: 1px solid #eaeaea; border-radius: 8px; text-align: left;">
    <div style="margin-bottom: 10px; color: #5a5a5a;">
      <i class="{{ edu.icon }} fa-2x"></i>
    </div>
    <h3 style="margin: 0 0 10px 0; font-size: 1.1em;">{{ edu.title }}</h3>
    <p style="margin: 0; font-size: 0.9em; color: #666;">{{ edu.excerpt }}</p>
  </div>
{% endfor %}
</div>

## Curriculum Vitae

For a detailed look at my experience, skills, and publications.

<div class="text-center">
  <a href="/assets/CV%20Gerard.pdf" class="btn btn--primary btn--large"><i class="fas fa-file-pdf"></i> View My CV</a>
</div>
