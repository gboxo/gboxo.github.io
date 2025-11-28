---
layout: archive
title: "AIxBio"
permalink: /aixbio/
author_profile: true
entries_layout: grid
classes: wide
---

Here you will find posts related to **AI in Biology** (BIOML).

{% assign posts = site.categories['aixbio'] %}
{% assign postsByYear = posts | group_by_exp: "post", "post.date | date: '%Y'" %}

{% for year in postsByYear %}
  <h2 id="{{ year.name }}" class="archive__subtitle">{{ year.name }}</h2>
  <div class="year-grid">
    {% for post in year.items %}
      {% include archive-single.html type="grid" %}
    {% endfor %}
  </div>
{% endfor %}
