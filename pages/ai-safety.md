---
layout: archive
title: "AI Safety"
permalink: /ai-safety/
author_profile: true
entries_layout: grid
classes: wide
---

Here you will find posts related to **AI Safety** and Mechanistic Interpretability.

{% assign posts = site.categories['ai-safety'] %}
{% assign postsByYear = posts | group_by_exp: "post", "post.date | date: '%Y'" %}

{% for year in postsByYear %}
  <h2 id="{{ year.name }}" class="archive__subtitle">{{ year.name }}</h2>
  <div class="year-grid">
    {% for post in year.items %}
      {% include archive-single.html type="grid" %}
    {% endfor %}
  </div>
{% endfor %}
