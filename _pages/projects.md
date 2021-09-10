---
layout: page
title: Projects
permalink: /projects/
description: Few of my research projects during my Ph.D. and postdoc are listed below.
nav: true
---
<a name="top"></a>
My projects could be divided into two distinct groups: (1) developing generic Active Learning (AL) methods and applying them on various applications including medical image processing, and (2) applying machine learning and network analysis to incorporate social aspects of science in stuyding knowledge production and scientific discoveries.

<h5><b> Contents </b></h5>

<ul>
  <li><a href="#AL">Active Learning </a></li>
  <ul>
     <li><a href="#GAL">Generic Active Learning </a></li>
     <li><a href="#AppAL">Application to Medical Image Processing </a></li>
     <li><a href="#ULAL">Interactive Unsupervised Learning </a></li>
  </ul>
  <li><a href="#SoS">Science of Science </a></li>
  <ul>
     <li><a href="#human_AI">Human AI for Accelerating Future Discoveries </a></li>
     <li><a href="#alien_AI">Alien AI for Punctuating Disruptive Discoveries </a></li>
  </ul>
</ul>


<a name="AL"></a>
<h3><b> Active Learning </b></h3>
Active learning (AL) is a branch of machine learning that addresses the problem of minimizing the cost of data annotations necessary for training accurate learning models. This task is increasingly paid more attention in the era of data science where raw, unannotated data are available in low cost but their labeling is time-consuming and financially expensive. Classically, AL methods assumed a fixed budget for labeling samples, and formulated the problem as how to select a certain number of queries such that their annotations will likely provide the most amount of information towards the best model fit.

<a name="GAL"></a>
<h4><b> Generic AL </b></h4>
A major part of my research included addressing two shortcomings of the most popular querying metric in AL, that is entropy-basede Unertainty Sampling (US). This metric selects those queries about which the model has the highest uncertainty. While US has a sound mathematical optimization problem, in practice it often leads to selecting redundant queries and/or outliers. These types of queries are not desirable in AL settings since they do not provide maximum information gain. The main source of sub-optimality in entropy-based US goes back to considering individual labeling candidates separately in the querying optimization.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/redundancy.png' | relative_url }}" alt="" title="Brain MRI images of three patient groups"/>
    </div>
</div>
<div class="caption">
     The issue of redundancy (image is taken from  <a href="https://link.springer.com/article/10.1007/s10115-012-0507-8">Fu et al. 2013</a>)
</div>

Numerous hueristics have been suggested to fix these issues of US. We proposed to change the querying objective to a more sophisticated information theoretical metric, i.e., Mutual Information (MI) and Fisher Information (FI). Both MI and FI by definition have mechansims that enable them to automatically address above issues: MI measures expected information gain of all the candidate queries and their (unknown) annotations collectively in its formulation, and FI aims at choosing samples that are most informative regarding the joint distribution of the whole data.

Our MI- and FI-based algorithms were first of their own kind. Our MI framework attacked the querying problem via directly computing MI between discrete class labels of the query candidates despite all the previous attempts that worked with crude approximations. On FI's side, we theoretically explained how FI-based active learning can improve the information gain when being used in conjunction with maximul-likeliood estimators. Our derivations have been published in the Journal of Machine Learning Research (JMLR) ([see here](https://www.jmlr.org/papers/volume18/15-104/15-104.pdf){:target"_blank" class="external"}), where we reviewed other similar methods and demonstrated their theoretical drawbacks with regards to ours.

<a href="#top">Back to Top </a>

<a name="AppAL"></a>
<h4><b>  Application to Medical Image Processing </b></h4>
Our proposed information theoretical AL methods explained above dramatically outperformed entropy-based US and random sampling baselines on popular machine learning benchmarks (e.g., [UCI datasets](https://archive.ics.uci.edu/ml/datasets.php){:target"_blank" class="external"}). Testing them on a real-world application was whta I did in my first postdoc at Boston Children's Hospital, Harvard Medical School.

In hospitals gigantic amount of unlabeled images are being produced on a daily basis and there is a need for AI-assisted algorithms to increase the speed and accuracy of their interpretation. There are several impediments in doing this task. Here's a brief list:

<ul>
  <li>The pre-trained state-of-the-art deep learning models cannot be readily applied to all groups of patients due to large visual characteristics of their medical images.</li>
  <li>Training or fine-tuning deep models is not possible for many patient groups as enough annotated data is not available for all ages or pathological conditions.</li>
  <li>Preparing annotations for medical images is a time-consuming and costly task and in most cases only a small number of image volumes can be labeled due to budget considerations.</li>
</ul>

These issues sound like a good application of AL to train or fine-tune a deep model with a fixed labeling budget. We slightly modified our FI-based AL to make it tractable for deep models and applied it with Convolutional Nneural Networks (CNN) on the problem of IntraCranial Cavity (ICC) extraction in brain MRI images. We worked with three distinct patient groups: healthy adolescents, newborns (with age between two months and 2.5 years) and patients with [Tuberous Sclerosis Complex (TSC)](https://www.tsalliance.org/about-tsc/what-is-tsc/){:target"_blank" class="external"}}. Visual characteristics of these images, specifically T1- and T2-weighted MR images, vary extensively among these groups. For example, we noticed that T1-MRI images of newborns show a completely reversed intensity contrast compared to older subjects. 

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/Brain_MRI.jpg' | relative_url }}" alt="" title="Brain MRI images of three patient groups"/>
    </div>
</div>
<div class="caption">
     <b>(left)</b> healthy adolescent, <b>(middle)</b> newborn, <b>(right)</b> TSC patient
</div>

Because of major variabilities across the images, models trained on one group will not be accurately generalized to the rest. We pre-trained a CNN based on healthy adolescents and used our modified FI-based AL to fine-tune it to the other two groups with limited number of annotations. Comparing the results with multiple variations of US that were common in medical image processing indicated that using FI enabled the model to reach a high performance with significantly smaller number of annotaions. In the setting explained above, this means that using FI queyring metric lead to a more generalizable model with a fixed labeling budget.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/ICC_AL_examples.png' | relative_url }}" alt="" title="Comparison of our FI-based AL and state-of-the-art US methods" width=1552 height=800/>
    </div>
</div>
<div class="caption">
     Results of ICC extraction after fine-tuning the pre-trained model with 50 AL-generated annotations. Our method is tagged as ``FI-based", and the rest are state-of-the-art US methods specifically designed for deep learning models. Our model resulted in a cleaner extraction in all shown examples. For further numerical evaluations see <a href="https://ieeexplore.ieee.org/abstract/document/8675438"> Sourati et al., 2019 </a>.
</div>

<a href="#top">Back to Top </a>

<a name="ULAL"></a>
<h4><b> Interactive Unsupervised Learning </b></h4>
All I explained so far was the development and application of AL in supervised learning problems. I also worked with similar issues in the context of unsupervised learning. Indeed, this line of research formed the very first project that I did in my Ph.D.. Since AL comes to the scene when some kinds of interaction exists between the machine and the user, we considered constrained clustering algorithm where the user-provided <em>pairwise</em> constraints were in form of must- and cannot-links.

The idea of entropy-based US can also be applied in this problem, but with the same redundancy and outlier vulnaribility issues. In my project, we addressed the latter issue by combining entropy and an estimate of data density to make sure that the queries are located in dense regions. We applied our density-weighted AL algorithm with an extended version of spectral clustering that could be scaled to segment intermediate size to large images. The final product was a practical user-interactive unsupervised framework that could segment complicated images by only using naive intensity features and with the help of AL-accelerated user pairwise constraints. 

<a href="#top">Back to Top </a>

<a name="SoS"></a>
<h3><b> Science of Science </b></h3>
My personal interests in humanities and social sciences drove me to start a second post-doc at Knowledge Lab in the Sociology Department of University of Chicago. In there, supervised by prominent computational sociologist James A. Evans, I becamse familiar with the topic of Science of Science and started a project about social aspects of scientific knowledge discoveries. 

<a name="human_AI"></a>
<h4><b> Human AI for Accelerating Future Discoveries </b></h4>
We investigated the importance of incorporating authors and their interconnections in the computational modeling of scientific findings. Artificial intelligence (AI) algorithms have been developed over scientific publications to automatically identify discovery candidates from a vast space of all possible combinatorial scientific knowledge. The number of possible candidates is so large in many disciplines that manual, exhaustive search through the possibilities has become intractable. As an example, a work by <a href="https://www.nature.com/articles/s41586-019-1335-8"> Tshitoyan et al. (2019) </a> proposed to utilize <a href="https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf">skipgram word2vec model</a> in the field of materials science to predict which materials possess electrochemical properties such as thermoelectricity. They showed that their model could achieve the precision of %40 in predicting future discoveries occurring in two decades by building their word embedding model only based on the prior knowledge, with no relevant scientific knowledge. This work and other similar algorithms in this line of research neglect authors of the publications that they used to create their models. The idea in our project was that scientists and experts, the main workforce of science's engine, are those who push forward the limits of knowlege in varios fields and it seems crucial to take their roles into account when analyzing advancement of scinece.

Interestingly, incorporating distribution of experts in predicting the future discoveries boosted the results' precision. The intuition is that a piece of knowledge would be unfolded only when its scientific plausibility is joined by enough attention paid by human scientists. For instance, if material X actually shows thermoelectric effects in reality, this property will not be discovered as long as there is no experts around to put forth this hypothesis and then experiment it. Hence, *cognitive availability* is key to discovering scietific knowledge. In order to incorporate expert distribution into our model, we built a hypergraph that included both author nodes and conceptual nodes (e.g., materials and their properties). The hyperedges comprised of papers linking nodes based on the authorship and mention of the materials/properties in their title or abstract. We then predict the future hypotheses by pairing material-property nodes with the largest hypergraph-induced similarities, which could be measured by any graph representation algorithm such as <a href="https://dl.acm.org/doi/10.1145/2623330.2623732">deepwalk embeddig</a>. 

We evaluated the effect of introducing experts distribution on the quality of discovery predictions in the context of materials science, disease therapies and vaccination. Numerical results showed that applying our algorithm led up to 100% precision increase in distinguishing materials with certain enery-related properties (thermoelectricity, ferroelectricity, being photovoltaic), average 43% increase in drugs repurposing for one hundred human diseases, and a boost of 260% in therapy/vaccine prediction for COVID-19 (see the figure below).

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/covid_results.png' | relative_url }}" alt="" title="results on COVID-19" width=1552 height=800/>
    </div>
</div>
<div class="caption">
     Results of predicting COVID-19 therapies and vaccines based on articles published before 2020 (~28M papers retrieved from PubMed database) using algorithms based on our expert-aware hypergraph (green and blue curves corresponding to deepwalk variants and transition probability scoring, respectively), content-based word2vec and a random baseline. We have also included the result of an algorithm that scored drugs based on their protein interaction with COVID-19, a piece of information that our hypergraph was blind to, which was still outperformed by our hypergraph-based measures. See further details in <a href="https://arxiv.org/pdf/2104.05188.pdf">our paper</a>.
</div>

<a href="#top">Back to Top </a>

<a name="alien_AI"></a>
<h4><b> Alien AI for Punctuating Disruptive Discoveries </b></h4>
Our expert-aware algorithm described above also enabled us to avoid the distribution of experts to generate hypotheses that are not imaginable by human scientists without machine intervention. These types of hypotheses are likely to escape scientists' collective attention based on their professional networking and prior research experience. We developed a simple algorithm to identify materials that are far from the property node in our hypergraph (in terms of shortest-path distance) and at the same time show high semantic relationship with the property. These materials are coginitively inavailable to scientists and authors due to their high distance to the property, yet are potentially good candidates of possessing the property due to their high semantic similarities with respect to the property. We actually showed that in practice such "alien" hypotheses could be very promising even when they are infinitely far from the property node (see the figure below).

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/alien_circles.png' | relative_url }}" alt="" title="result of our alien AI algorithm" width=1552 height=800/>
    </div>
</div>
<div class="caption">
   Distribution of discoveries and predictions in terms of their distance to the property node (here, thermoelectricity) color-coded in terms of how scientifically promising they are (measured by Power Factor (PF), a key theoretical score for thermoelectricity). The center indicates the property node. <b>(top)</b> actual thermoelectric discoveries in different years (concentrated in the first two orbits with high average PF values); <b>(bottom-left)</b> all materials that could be candidates of discovery in different years (distributed more uniformly across various orbits but lower PF values); <b>(right-bottom)</b> predictions of our alien AI algorithm with beta (between zero and one) defining how much emphasis we give to expert avoidance (a more balanced trade-off between distance and PF distributions). See further details in <a href="https://arxiv.org/pdf/2104.05188.pdf">our paper</a>.
</div>

Our findings imply that the pattern of scientific discoveries is highly predictable merely through the coauthorship network and the prior experience of the scientists, without any prior knowledge of the underlying theories and even without the knowledge of the content of the prior literature. These patterns, in one hand, could enable us to predict the upcoming future discoveries, and on the other hand, help us distinguish and escape collective biases in the course of scientific discoveries, and thereby open doors to previously unimaginable hypotheses.

<a href="#top">Back to Top </a>