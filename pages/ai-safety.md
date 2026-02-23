---
layout: archive
title: "AI Safety"
permalink: /ai-safety/
---

Here you will find posts related to **AI Safety** and Mechanistic Interpretability.

{% assign posts = site.categories['ai-safety'] %}
{% assign postsByYear = posts | group_by_exp: "post", "post.date | date: '%Y'" %}

{% for year in postsByYear %}
### {{ year.name }}

<ul>
  {% for post in year.items %}
    <li>
      <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%Y-%m-%d" }}</time>
      &mdash; <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
{% endfor %}
