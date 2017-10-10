---
layout: page
title: Implementing the Random Forest Classifier for the First Time
permalink: /sklearn/rfc-tutorial
resource: true
categories: [sklearn]
---

<h1 class="page-title">{{ page.title | escape }}</h1>

This tutorial walks you through implementing scikit-learn's Random Forest Classifier on the Iris training set. It demonstrates the use of a few other functions from scikit-learn such as train_test_split and classification_report.

Note: you will not be able to run the code unless you have scikit-learn and pandas installed. If you don't know how to do that or if you want to replicate the code yourself follow my tutorial to <a href="http://joshlawman.com/getting-set-up-in-jupyter-notebooks-using-anaconda-to-install-the-jupyter-pandas-sklearn-etc/">set up Jupyter Notebook</a> first.
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="1.-Import-dataset">1. Import the dataset</h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

We will be using the iris dataset (<a href="https://en.wikipedia.org/wiki/Iris_flower_data_set">https://en.wikipedia.org/wiki/Iris_flower_data_set</a>) to train our classifier. It comes preloaded with scikit-learn (sklearn).

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [1]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="c1">#Import dataset</span>
<span class="kn">from</span> <span class="nn">sklearn.datasets</span> <span class="k">import</span> <span class="n">load_iris</span>
<span class="n">iris</span> <span class="o">=</span> <span class="n">load_iris</span><span class="p">()</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="2.-Prepare-training-and-testing-data">2. Prepare training and testing data</h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

Each flower in this dataset contains the following features and labels
<ul>
    <li>features - measurements of the flower petals and sepals</li>
    <li>labels - the flower species (setosa, versicolor, or virginica) represented as a 0, 1, or 2.</li>
</ul>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

Our train_test_split function will seperate the data as follows
<ul>
    <li>(features_train, labels_train) - 80% of the data prepared for training</li>
    <li>(features_test, labels_test) - 20% of the data prepared for making our predictions and evaluating our model</li>
</ul>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [2]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="c1">#Import train_test_split</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="k">import</span> <span class="n">train_test_split</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [3]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">features_train</span><span class="p">,</span> <span class="n">features_test</span><span class="p">,</span> <span class="n">labels_train</span><span class="p">,</span> <span class="n">labels_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">iris</span><span class="o">.</span><span class="n">data</span><span class="p">,</span><span class="n">iris</span><span class="o">.</span><span class="n">target</span><span class="p">,</span><span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span><span class="n">random_state</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="3.-Create-and-fit-the-Random-Forest-Classifier">3. Create and fit the Random Forest Classifier</h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

This tutorial uses the RandomForestClassifier model for our predictions, but you can experiment with other classifiers. To do so, import another classifier and replace the relevant code in this section.

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [4]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="c1">#Import classifier</span>
<span class="kn">from</span> <span class="nn">sklearn.ensemble</span> <span class="k">import</span> <span class="n">RandomForestClassifier</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [5]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="c1">#Create an instance of the RandomForestClassifier</span>
<span class="n">rfc</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">()</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [6]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="c1">#Fit our model to the training features and labels</span>
<span class="n">rfc</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">features_train</span><span class="p">,</span><span class="n">labels_train</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[6]:</div>
<div class="output_text output_subarea output_execute_result">
<pre>RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
max_depth=None, max_features='auto', max_leaf_nodes=None,
min_impurity_decrease=0.0, min_impurity_split=None,
min_samples_leaf=1, min_samples_split=2,
min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
oob_score=False, random_state=None, verbose=0,
warm_start=False)</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="4.-Make-Predictions-using-Random-Forest-Classifier">4. Make Predictions using Random Forest Classifier</h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [7]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">rfc_predictions</span> <span class="o">=</span> <span class="n">rfc</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">features_test</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

<hr />

<strong>Understanding our predictions</strong>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

Our predictions will be an array of 0's 1's, and 2's, depending on which flower our algorithm believes each set of measurements to represent.

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [8]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="nb">print</span><span class="p">(</span><span class="n">rfc_predictions</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>[0 1 1 0 2 1 2 0 0 2 1 0 2 1 1 0 1 1 0 0 1 1 2 0 2 1 0 0 1 2]
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

To intepret this, consider the first set of measurements in features_test:

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [9]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="nb">print</span><span class="p">(</span><span class="n">features_test</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>[ 5.8 4. 1.2 0.2]
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

Our model believes that these measurements correspond to a setosa iris (label 0).

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [10]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="nb">print</span><span class="p">(</span><span class="n">rfc_predictions</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>0
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

In this case, our model is correct, since the true label indicates that this was a setosa iris (label 0).

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [11]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="nb">print</span><span class="p">(</span><span class="n">labels_test</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>0
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="5.-Evaluate-our-model">5. Evaluate our model</h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

For this section we will import two metrics from sklearn: confusion_matrix and classification_report. They will help us understand how well our model did.

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [12]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="c1">#Import pandas to create the confusion matrix dataframe</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>

<span class="c1">#Import classification_report and confusion_matrix to evaluate our model</span>
<span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="k">import</span> <span class="n">classification_report</span><span class="p">,</span> <span class="n">confusion_matrix</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

As seen in the confusion matrix below, most predictions are accurate but our model misclassified one specimen of versicolor (our model thought that it was virginca).

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [13]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="c1">#Create a dataframe with the confusion matrix</span>
<span class="n">confusion_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">confusion_matrix</span><span class="p">(</span><span class="n">labels_test</span><span class="p">,</span> <span class="n">rfc_predictions</span><span class="p">),</span>
<span class="n">columns</span><span class="o">=</span><span class="p">[</span><span class="s2">"Predicted "</span> <span class="o">+</span> <span class="n">name</span> <span class="k">for</span> <span class="n">name</span> <span class="ow">in</span> <span class="n">iris</span><span class="o">.</span><span class="n">target_names</span><span class="p">],</span>
<span class="n">index</span> <span class="o">=</span> <span class="n">iris</span><span class="o">.</span><span class="n">target_names</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [14]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">confusion_df</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[14]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div><style>
.dataframe thead tr:only-child th {<br />text-align: right;<br />}</p>
<p>.dataframe thead th {<br />text-align: left;<br />}</p>
<p>.dataframe tbody tr th {<br />vertical-align: top;<br />}<br /></style>
<table class="dataframe" border="1">
<thead>
<tr style="text-align: right;">
<th></th>
<th>Predicted setosa</th>
<th>Predicted versicolor</th>
<th>Predicted virginica</th>
</tr>
</thead>
<tbody>
<tr>
<th>setosa</th>
<td>11</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<th>versicolor</th>
<td>0</td>
<td>12</td>
<td>1</td>
</tr>
<tr>
<th>virginica</th>
<td>0</td>
<td>0</td>
<td>6</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

As seen in the classification report below, our model has 97% precision, recall, and accuracy.

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [15]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">labels_test</span><span class="p">,</span><span class="n">rfc_predictions</span><span class="p">))</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre> precision recall f1-score support

0 1.00 1.00 1.00 11
1 1.00 0.92 0.96 13
2 0.86 1.00 0.92 6

avg / total 0.97 0.97 0.97 30

</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

<hr />

<h3 id="-Note-on-the-RandomForestClassifier-from-sklearn"><strong> Note on the RandomForestClassifier from sklearn</strong></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

Documentation with full explanation of parameters and use: <a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html</a>.

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

Some useful parameters to experiment with:
<ul>
    <li>min_samples_leaf (the minimum samles which can be put into each lef)</li>
    <li>n_estimators (the number of decision trains)</li>
    <li>max_features (the size of the subset of features to be examined at each split)</li>
</ul>
An optional feature to take advantage of:
<ul>
    <li>oob_score (a way of seeing how well the estimator did by cross-validiting on the "out of bag" data, i.e. the data
for each tree that was not used in the sample). This would be usefull if you didn't want to split your dataset into a training dataset and a test dataset.</li>
</ul>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Note-on-metrics">Note on metrics</h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

Check out wikipedia if confusion matrices are new (<a href="https://en.wikipedia.org/wiki/Confusion_matrix">https://en.wikipedia.org/wiki/Confusion_matrix</a>) or if you want explanation on precision and recall (<a href="https://en.wikipedia.org/wiki/Precision_and_recall">https://en.wikipedia.org/wiki/Precision_and_recall</a>).

</div>
</div>
</div>