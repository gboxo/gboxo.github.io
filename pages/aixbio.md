---
layout: archive
title: "AIxBio"
permalink: /aixbio/
---

Here you will find posts related to **AI in Biology** (BIOML).

{% assign posts = site.categories['aixbio'] %}
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
