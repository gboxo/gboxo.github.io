---
layout: single
author_profile: true
header:
  overlay_image: /assets/images/banner.jpeg
  overlay_filter: 0.5
  caption: "Mechanistic Interpretability & Biology"
excerpt: "Deciphering the black box of AI in Safety and Biology."
---

Welcome to my research blog. I explore the inner workings of **Artificial Intelligence** models, focusing on two critical domains:

<div class="feature__wrapper">
  <div class="feature__item">
    <div class="archive__item">
      <div class="archive__item-teaser">
        <i class="fas fa-shield-alt fa-3x"></i>
      </div>
      <div class="archive__item-body">
        <h2 class="archive__item-title">AI Safety</h2>
        <p>Mechanistic Interpretability applied to making AI systems aligned, controllable, and transparent.</p>
        <p><a href="/ai-safety/" class="btn btn--primary">Explore AI Safety</a></p>
        {% assign ai_safety_posts = site.posts | where_exp: "post", "post.categories contains 'ai-safety'" | limit: 2 %}
        {% for post in ai_safety_posts %}
        <div class="archive__item-excerpt">
          <h3><a href="{{ post.url }}">{{ post.title }}</a></h3>
          <p class="page__meta"><i class="far fa-clock" aria-hidden="true"></i> {{ post.read_time | default: "less than 1 minute read" }}</p>
        </div>
        {% endfor %}
      </div>
    </div>
  </div>

  <div class="feature__item">
    <div class="archive__item">
      <div class="archive__item-teaser">
        <i class="fas fa-dna fa-3x"></i>
      </div>
      <div class="archive__item-body">
        <h2 class="archive__item-title">AI x Bio</h2>
        <p>Investigating how LLMs and specialized models can accelerate protein engineering and drug discovery.</p>
        <p><a href="/aixbio/" class="btn btn--primary">Explore AIxBio</a></p>
        {% assign aixbio_posts = site.posts | where_exp: "post", "post.categories contains 'aixbio'" | limit: 2 %}
        {% for post in aixbio_posts %}
        <div class="archive__item-excerpt">
          <h3><a href="{{ post.url }}">{{ post.title }}</a></h3>
          <p class="page__meta"><i class="far fa-clock" aria-hidden="true"></i> {{ post.read_time | default: "less than 1 minute read" }}</p>
        </div>
        {% endfor %}
      </div>
    </div>
  </div>
</div>
