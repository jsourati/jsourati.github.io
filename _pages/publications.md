---
layout: page
permalink: /publications/
title: Publications
description: 
years: [2017, 2016]
nav: true
---
Below is a list of my selected publications by categories in reversed chronological order. See my [CV]({{ site.url }}/assets/pdf/myCV.pdf) for a more comprehensive list, and my [Google Scholar](https://scholar.google.com/citations?user=JeVyxh4AAAAJ&hl=en&oi=sra) for the complete records.

<h3><b> Generic Active Learning </b></h3>
<div class="publications">
{% bibliography -f papers -q @*[year=2022]* %}
{% for y in page.years %}
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}
</div>

<h3><b> Applied Active Learning </b></h3>

<div class="publications">
  {% bibliography -f papers_appl -q @*[year=2019]* %}
  {% bibliography -f papers_appl -q @*[year=2018]* %}
</div>

<h3><b> Unsuperised Learning </b></h3>
<div class="publications">
  {% bibliography -f papers_unsp -q @*[year=2014]* %}
</div>


<h3><b> AI-assisted Knowledge Discovery </b></h3>
<div class="publications">
  {% bibliography -f papers_sos -q @*[year=2022]* %}
  {% bibliography -f papers_sos -q @*[year=2021]* %}
</div>

