---
layout: page
title: DeepLearning.AI: Course 1 - Week 1 Notes
permalink: /deep-learning/course1-week1
resource: true
categories: [deeplearning]
---

<h1 class="page-title">{{ page.title | escape }}</h1>

<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="1.-Neural-Nets-are-Particularly-Useful-in-Supervised-Learning-Problems">1. Neural Nets are Particularly Useful in Supervised Learning Problems<a class="anchor-link" href="#1.-Neural-Nets-are-Particularly-Useful-in-Supervised-Learning-Problems">&#182;</a></h4><ul>
<li>Standard Neural nets: Real estate, online advertising (selecting best ad to show user based on prediction regarding whether they will click on ad or not).</li>
<li>Convolutional Neural Nets (CNNs): Image recognition.</li>
<li>Recurrent Neural Nets (RNNs): Audio processing (speech recognition) and text translation.</li>
<li>Hybrid: Autonomous driving.</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="2.-NNs-Work-Well-with-Structured-and-Unstructured-Data">2. NNs Work Well with Structured and Unstructured Data<a class="anchor-link" href="#2.-NNs-Work-Well-with-Structured-and-Unstructured-Data">&#182;</a></h4><ul>
<li>Structured data example: Data arranged neatly such as in a Pandas DataFrame.</li>
<li>Unstructured data examples: Audio, image, text.</li>
<li>Unstructured data has traditionally been harder for computers to understand which is why the NN improvements in this area have been so ground-breaking.</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="3.-Scale-(in-Data-and-Computational-Power)-Drives-NN-Performance">3. Scale (in Data and Computational Power) Drives NN Performance<a class="anchor-link" href="#3.-Scale-(in-Data-and-Computational-Power)-Drives-NN-Performance">&#182;</a></h4><ul>
<li>At small scale, the most important factor is manipulating the data (with feature engineering, etc.). At small scale, traditional machine learning algorithms perform comparably well to large neural nets.</li>
<li>At large scale, large neural nets with large amounts of data will perform the best.</li>
<li>Increasing the amount of training data usually does not harm the performance of the algorithm.</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The code below generates a replica of the graph which Andrew Ng drew to illustrate neural net performance as it compares to data size and neural net size.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Import libraries</span>
<span class="kn">import</span> <span class="nn">seaborn</span> <span class="k">as</span> <span class="nn">sns</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="o">%</span><span class="k">matplotlib</span> inline

<span class="c1">#Create x linspace</span>
<span class="n">x</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">50</span><span class="p">,</span><span class="mi">990</span><span class="p">)</span>

<span class="c1">#Create dummy functions to represent performance</span>
<span class="n">trad_alg</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
<span class="n">small_nn</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="n">x</span><span class="o">*</span><span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">3</span><span class="p">,</span><span class="mi">990</span><span class="p">))</span>
<span class="n">medium_nn</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="n">x</span><span class="o">*</span><span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">6</span><span class="p">,</span><span class="mi">990</span><span class="p">))</span>
<span class="n">large_nn</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="n">x</span><span class="o">*</span><span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">10</span><span class="p">,</span><span class="mi">990</span><span class="p">))</span><span class="o">*</span><span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mf">1.3</span><span class="p">,</span><span class="mi">990</span><span class="p">)</span>

<span class="c1">#Plot dummy functions</span>
<span class="n">fig</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span><span class="mi">6</span><span class="p">))</span>
<span class="n">g</span> <span class="o">=</span> <span class="n">sns</span><span class="o">.</span><span class="n">regplot</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">trad_alg</span><span class="p">,</span> <span class="n">fit_reg</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span><span class="n">label</span><span class="o">=</span><span class="s1">&#39;Traditional Algorithms&#39;</span><span class="p">)</span>
<span class="n">sns</span><span class="o">.</span><span class="n">regplot</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">small_nn</span><span class="p">,</span> <span class="n">fit_reg</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span><span class="n">label</span><span class="o">=</span><span class="s1">&#39;Small NN&#39;</span><span class="p">)</span>
<span class="n">sns</span><span class="o">.</span><span class="n">regplot</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">medium_nn</span><span class="p">,</span> <span class="n">fit_reg</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span><span class="n">label</span><span class="o">=</span><span class="s1">&#39;Medium NN&#39;</span><span class="p">)</span>
<span class="n">sns</span><span class="o">.</span><span class="n">regplot</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">large_nn</span><span class="p">,</span> <span class="n">fit_reg</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span><span class="n">label</span><span class="o">=</span><span class="s1">&#39;Large NN&#39;</span><span class="p">)</span>


<span class="c1">#Set labels and remove numbers for axes</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Amount of labeled data (m)&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Performance&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
<span class="n">g</span><span class="o">.</span><span class="n">set</span><span class="p">(</span><span class="n">yticks</span><span class="o">=</span><span class="p">[])</span>
<span class="n">g</span><span class="o">.</span><span class="n">set</span><span class="p">(</span><span class="n">xticks</span><span class="o">=</span><span class="p">[]);</span>

<span class="c1">#Add annotation</span>
<span class="n">g</span><span class="o">.</span><span class="n">text</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="o">-</span><span class="mi">2</span><span class="p">,</span> <span class="s2">&quot;|- - With small amounts of data - -|</span><span class="se">\n</span><span class="s2">   NN complexity does not </span><span class="se">\n</span><span class="s2">   greatly corresepond to </span><span class="se">\n</span><span class="s2">   improved performance&quot;</span><span class="p">,</span> <span class="n">fontsize</span> <span class="o">=</span> <span class="mi">12</span><span class="p">)</span>

<span class="c1">#Create title</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;Large Neural Nets with Large Training Sets have the Best Performance&#39;</span><span class="p">);</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt"></div>



<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
tgcpIiISg2VL17JwXhkrf95AbW1i1mYVFOYxZFiftJhaa0naBqUla75m/vIPWV2xli5tOjG8+z7s
2nlA1NtbuvR73nvvbaZOnYbH4+GbbyzXXTeZRx75b9zGHGjr1q3cfvu/uPTSq5pcV1zclhtuuJr7
73+UgoKChI1BRESkJQmfUiOzwpG/tAxKS9Z8zUvfv9Lwc3nF6oafow1Lbdu2ZeXKFbz88ovst9/+
7Lyz4f77HwGcytBOOw1g6dLvaNOmDYMHD+GDD+azadMmbr31LvLyvNx443Vs2rSR1avL+fWvT+K4
405oYY9w+OHj+OKLz3jvvXcYMeKARtf16tWbPfccwn333c3EiX+J6jGJiIhES+EoPGkZlOYv/zDk
5dEGpa5dt+PGG2/l2Wef4sEH76d169acc875jB59EAC77bY7f/nLxfz1rxfQunVr/v3vu7nuukl8
+ulCtt++GwcffCijRo1h9epyJk48J6yg5PV6ufzyyVx88Z8YOHBQk+vPOus8zj77dD777NOoHpOI
iEgkEt1vlI7N2LFKy6C0umJt8Mu3Br88HD/+uIzi4mIuu2wSAF999SUXX/wnhg7dG6DhnGYlJW3p
16+/++92VFVV0qlTJ55++gnmzp1DUVExNTU1Ye+3d+8+nHjiKfzrX/9schLZgoICLrtsEldffTlH
HXVc1I9NREQkFIWj2KRlUOrSphPlFaubXt46+l/Ad999w4svPs8//3kr+fn59O7dh7ZtS/B68wDw
BKYYP08++R8GDhzMccedwMKFHzF//rsR7fv440/mnXfm8t1333LMMcc3us6YXTjkkLE8/vgjYVWp
REREWpLrK9XiKS2D0vDu+zTqUfK/PFqjRo2htHQpZ511GkVFbair83H++X+mbdu2Ld53xIgDue22
m3jzzddo27YteXl5VFVVhb1vj8fDpZdO4vTTTw56/e9/fybvvfdO2NsTEREJlOhwlA39RtFI23O9
Nax627qWLq1jX/UmIiKSbRIajrJ8Ss1fc+d6S9ugJCIiIk0pHMWfToorIiKSoRK9jD+X+o2ioaAk
IiKSZhIdjnK13ygaCkoiIiJpQOEoPSkoiYiIpIjCUfpTUBIREUkihaPM4k31AJJl4cKPGDlyb954
Y1ajy08//RSmTJkc1jbKykqZOPEcACZNupTq6uq4jG3ixHO4445/NfxcWVnJCSccBcC0afdy9tmn
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
QpM370cO333vpI9JREREMltWBKVv5iyk1hOsPwnyaytYs12P5A9KREREMl5WBKWF73+H11d/2pJt
x5+HRmIAABuySURBVFHy1tXQtrqckWOPTdXQREREJINlRVDaXFNInaf+ofioryr5PF68eT+piVtE
RESikhVBqdaTj8/j36PkVJVa1VWzuXfvlI1LREREMlvGB6UZT71ErSff7xKnouTx1eH1VbL3r8al
amgiIiKS4TI+KK39diOehgbubdUkj68Or2ejpt1EREQkaq1avkl6q/G1xedpeiJcn8dLbaeC1AxK
REREskLGV5RqPQVB+5PyfNXsN/bglI1LREREMl9GB6Xm+pPy6qo07SYiIiIxyeigtPbbjXh9tXga
jqHk8PjqaOXdmKJRiYiISLbI6B6lOl9rWlFJtacNHl/ttis8Hjrt2C51AxMREZGskNEVJa9nK15f
Nfl1FQ1VJY+vjsLadYw7+egUj05EREQyXUYHpU47lQDg9VVTULeJwtoNFNRtosuA1ikemYiIiGSD
jA5K404+mi4D8vF6KwDweivoMiBf1SQRERGJC4/P52v5VhEqL98Y/42KiIiIJEDXriWeUNdldEVJ
REREJJEUlERERERCUFASERERCUFBSURERCQEBSURERGREBSUREREREJQUBIREREJQUFJREREJAQF
JREREZEQFJREREREQlBQEhEREQlBQUlEREQkhIScFFdEREQkG6iiJCIiIhKCgpKIiIhICApKIiIi
IiEoKIlkGGPMQGOMzxhzfIr2394Y80KE9znbGFNmjLk54PJSY0y/Zu432hjzVoT7anabQW7/sDHm
jBZuM9kYM7mF21xtjDkg3P269xlqjPlnBLe/xRgzJJJ9iEhsFJREMs+ZwDPAuSnaf0dgzwjv8xvg
bGvtJQkYT7oYBeRFeJ/bgLCDEnAj8O8I9yEiMWiV6gGISPiMMa2AU4EDgHnGmB2ttd8ZY0qBp4Bx
QA1wGXARsDNwkbX2aWPM9sA0oE/9bay1r9ZXSqy1k919lAKj3f/GAp2AHYDXrLXnA3cAPYwxz1tr
jwsY35nufn3Ax8BE4K/AvsDdxpg/WWtnBnlc7dyx9QJ6AG8Dp7lXdzHGvAr0BN4H/mitrTTGjAWu
AfKBpThBbI3fNvOAm93HkQc8bK29zRjjAf7lPlc/u9e9FWRMlwDnAKuBdcAH7uUTgd8DxUAdcDKw
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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="4.-Improving-Training-Speed-is-Essential-Given-the-Iterative-Process-of-Developing-with-NNs:">4. Improving Training Speed is Essential Given the Iterative Process of Developing with NNs:<a class="anchor-link" href="#4.-Improving-Training-Speed-is-Essential-Given-the-Iterative-Process-of-Developing-with-NNs:">&#182;</a></h4><ul>
<li>Neural networks are developed in an iterative fashion. This fact, combined with the need for large neural nets trained on massive data sets mean that improvements in training speed allows for faster innovation with NNs.</li>
<li>Algorithmic improvments have helped improve training speed. Developments such as using the ReLU function for activation instead of the sigmoid function have improved training speed which allow for faster iterations of implementing NNs.</li>
<li>Depending on the scale of the application, training can take a wide range of time from 10 minutes to an entire month.</li>
</ul>

</div>
</div>
</div>
 

