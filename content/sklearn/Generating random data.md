---
layout: page
title: Generating Random Data
permalink: /sklearn/generating-random-data
---

<h1 class="page-title">{{ page.title | escape }}</h1>

<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Scikit-learn has a number of methods to generate data in its datasets module. Below are some examples of the options available.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#Import pandas and plotting libraries to visualize data</span>
<span class="kn">import</span> <span class="nn">seaborn</span> <span class="k">as</span> <span class="nn">sns</span> 
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="o">%</span><span class="k">matplotlib</span> inline
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Blobs">Blobs<a class="anchor-link" href="#Blobs">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.datasets</span> <span class="k">import</span> <span class="n">make_blobs</span>

<span class="n">blob_data</span><span class="p">,</span> <span class="n">blob_labels</span> <span class="o">=</span> <span class="n">make_blobs</span><span class="p">(</span><span class="n">n_samples</span><span class="o">=</span><span class="mi">100</span><span class="p">,</span>
                                    <span class="n">n_features</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span>
                                    <span class="n">centers</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span>
                                    <span class="n">cluster_std</span><span class="o">=.</span><span class="mi">8</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">blob_data</span><span class="p">[:,</span><span class="mi">0</span><span class="p">],</span>
            <span class="n">blob_data</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> 
            <span class="n">c</span> <span class="o">=</span> <span class="n">blob_labels</span><span class="p">,</span>
            <span class="n">cmap</span><span class="o">=</span><span class="s1">&#39;viridis&#39;</span><span class="p">);</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt"></div>



<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAX0AAAD5CAYAAADLL+UrAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAAIABJREFUeJzt3Xd4XMW5+PHv2a5ebMm9l3HvxgabZhtsU0JJoSR0SCEV
Un5JbpJ7cxsJKaTckAChBQgQejUYjI0xGOPePS5ykyyr97Kr3T2/PyTLkraorbRa7ft5Hh60M+dM
sVbvnp0zZ8YwTRMhhBDxwRLtBgghhOg9EvSFECKOSNAXQog4IkFfCCHiiAR9IYSIIxL0hRAijti6
c7JSagHwa631RUqp8cATgAnsAb6ptfa3ONYCPAjMBNzAnVrrw92pXwghROd0OegrpX4E3ATUNCX9
HviZ1nqdUupvwFXAKy1OuRpwaa3PVUotBH7XdExYRUVVPfIgQUZGImVltT1RdFRJv2JPf+2b9Cu6
srJSjGDp3RneOQJc2+L1XODDpp9XAcvaHL8YeAdAa/0pMK8bdXebzWaNZvU9RvoVe/pr36RffVOX
r/S11i8ppUa3SDK01meuyquAtDanpAIVLV77lFI2rbU3XD0ZGYk99o+clZXSI+VGm/Qr9vTXvkm/
+p5ujem34W/xcwpQ3ia/sin9DEt7AR/osa9RWVkpFBVV9UjZ0ST9ij39tW/Sr+gK9cEUydk725VS
FzX9vBL4qE3+x8BlAE1j+rsjWLcQQogOiOSV/veBR5RSDmA/8CKAUuofwM9ovKl7iVLqE8AAbotg
3UIIITqgW0Ffa30MWNj080HgwiDH3Nzi5de7U584q+B4IaePFDBu9hiSM5Kj3RwhRIyI5JW+6AVV
JVU88v0n2PPRfuqr68kYnM45V8zlpv+6AYtFnrUTQoQnUSLGPHTPY2xZtZ366noAyk6X8+7f1/Dy
716PcsuEELFAgn4MydWn2PvRgaB5W1Zt7+XWCCFikQT9GHLyQC7uWnfQvMqiSvw+f9A8IYQ4Q4J+
DJk4fzwpmcFv2maNHIjFKr9OIUR4EiViyIChmcxdMTsg3e60c8F1i6LQIiFErJHZOzHm9vtvIiE1
gZ3v76KypIrsUdlccN15LL35omg3TQgRAyToxxib3cZNv7yeL//iS3jqPTgTnRhG0MX0hBAigAT9
GGWxWnAluaLdDCFEjJGgL0I6eSCX1Y9+QNHJYlIGpHD+F85lxsXTot0sIUQ3SNAXQe3fqPnrtx6h
OLe0OW3buzu4/mdf4JJbL45iy4QQ3SFBX3B093E+ev5j3HUexs0ewwXXLeL1P7/dKuAD1FXV8e4j
73Pxjedjc8hbR4hYJH+5cW7VQ6t58bevUVdZB8Dap9fzycubyD14Kujxpw7ns3fDfmYumd6bzRRC
RIjM049j5YUVvPS715sD/hn7Pj5AQ50n6DkWq4WEFLmBLESskiv9OFBRXMme9fsYNCqL8XPHAeD3
+/nNV/5AbUXwnclsThtUB6aPmTWaCfPGd7juI9tzeO/xtRSeKCJ1YAqLrl3I/MvmdqkfQojuk6Df
j5mmydP//jyfvPIpFYWV2Bw2Js4fzx2/uZld6/ZydOfxkOcOGJZJ9qhsjmzLaU4bNCab63/6hQ4/
F7Br3R4e+s5jlBWc3Tlzx/u7+dKPS7js65d2vWNCiC6ToN+PvfGXVax6eDU0bVfv9XjZ9/EBHr7n
cZLSEsOeO2nBRG789y/x4bMbyDuUT9rAVC65fQmJKQkdqrvweBEP3/tEq4AP4Knz8N4Ta1l268U4
XPYu9UsI0XUS9Puxbe/saA74LR3aeoQRk4eHPC8tO42rv3cFNrutS8s7HNlxlD9/9a+U5pUGzS84
WsDeDfuZvWxGp8sWQnRPRIO+UupW4Namly5gFjBYa13elH8PcCdQ1HTM17TWOpJtEGdVlVQFTfd7
/SSFuGI3rAZ3/f4WUgemdrne1/7wFoXHi0PmW2wWktOTuly+EKLrIhr0tdZPAE8AKKX+Ajx2JuA3
mQvcrLXeGsl6RXDZo7PJzykISHclu/j8D6+i9hfPc2x363H9c686hzmXzOpynaZpcnTXsbDHpGen
MX7u2C7XIYTouh4Z3lFKzQOmaq2/2SZrLvATpdRg4C2t9X09Ub9otPTmCzm09TC1Fa2nZM65dCaT
z5vE/3v2Ht74yyqO7z6OzWln2uLJrPxa92+wWmzWsPmmCQ1ub4fH9E3T5J//8xIfvvQplUWVZI3K
4oLrFnHh9Yu73VYh4k1Pjen/FPhlkPTngL8AlcArSqkrtNZvhisoIyMRWztBpKuyslJ6pNxIqSyp
4lROASPUUJJSg994PbD5MO89uY7aqnrGTB/JVd9cTsmxAt5++H1KC8pQ88dzYn8elcWVWG1WRk4e
zvcevIOMrBSyslK45y93Rqy9Pq+P959ej9MZ/m1VfroMs95N1ojMDpX7t+8/yct/eBOz6f5EcV4p
OTuOkphg57I7l3W32X1CX38vdpX0q++JeNBXSqUDSmu9tk26AfxBa13R9PotYDYQNuiXlQWfR95d
WVkpFBUFH/OONk+dh0d/9A92frCbyuIqModmkJ6VSmVpDe7aerJGDmTh5+Zjtdt44VevNG+SDvD2
I+9RUVJFdWlNQLkNbi8Htxzhp5f/ih8/fy+JKeFn8HRGTWUtv7/lz+z/pP1bNGlZafgslg79+1eX
17D2uQ3NAf8Md62HNx56n3mfOyfml5buy+/F7pB+RVeoD6aeuNK/AFgTJD0V2KOUmgzUAEuAx3qg
/pj39x8+yYYXNja/Lj1VRumpsubXVSXV5Gw/hivJRX1Nfatz8w6dbrf8w1tzWPW31Xz+h1dHrM3/
uu/lDgV8gBkXTyMxxDeXtg5vPUJpfnnQvIJjBbhrPbiSnB1upxDxrieCvgKan+hRSt0IJGutH1ZK
/RRYC7iBNVrrt3ug/phWWVzJzg92d+jYtgG/M47uCv1gVlfojQdD5jkTHbhrPSSmJTJzyXRu+9VX
OlzuoFHZuJKc1NcEbgiflJ4sc/2F6KSIB32t9W/avP5ni5+fAp6KdJ39SX5OAVUlQdY/iDCbM3LB
sq66jtNHC0PmX/ntyxg5ZQSjpo4ga8TATpU9ZPxgJp+n2P7eroC8WUumyWbwQnSS/MX0McPVMNIH
p/doHYbFYNbSyK2S+dof38JTH3yBttQBKVx62xLmrZjd6YB/xp2/u5V5y2dib/qgSkpLZNEXzuXG
X3ypy20WIl7JE7l9TFJaIvNWzOL9J9a1e+zwSUMpPlnSauhj5ORhuOs8FBwrak6zWAz8/sY7oY4E
O+d/KbLTHY9sOxo602KQnJHcrfIzBqVz36qf8enqnRzfm8uURYrBYwZ1q0wh4pUE/T7olv/5Mnan
nW2rd1GcV4yvwRewnEL26Cx+8dqPydP5bHhxI3XVdYyYNIwv/+QaCgsqWf3YGsryy8kaOZAZS6ax
6bXN+Bp8zF05mwlNK21GimEJPXumtqKW00cLIhKkx80ey7jZ8lCXEN0hQb8Pstqs3PSfN3DDz75I
TXkNez85wIfPbiD/yGmsVgtzls/iK7+8HsMwUAsmoBZMaD43IclFYkoDV3/3ilZljpwUeq2d7pp4
znj2rN8XNM/r8bJ3wwG5Mheij5Cg34fZHDbSstM47+oFnHf1gmg3J6TPfftyVj20mrqqwNlEdqed
kVOGRaFVQohg5Eau6DaHy85l31geNG/SwolMmNvxTVeEED1LrvRFRFxzz5XUlNey6bXNlBWU40xw
oM5V3PW7W6LdNCFECxL0+xlPvYdXH3iTQ1sOgwFqwUQu+9ql2Bw9+6u2WCzc/F83cM09V3Bg40EG
jxvEiB68jyCE6BoJ+v2I1+PlF1f/ka2rdzanbX9vFwc+Pcj3n/w21h5auK6llMwU5l8ue+AK0VdJ
0O9H1jy1rlXAP2PH+7tY//zHXPzlC6LQKiE6zufz8+wbm9m+N5fqOjfDB6Wz8sKpzJsxKtpN6zfk
Rm4/cnjLkZB5+rNDvdgSIbrmL099yEvv7CDnZDGFxVVs23uSPz25jq17IrtWVDyToN+PWB2h19Pp
6TF9IborN7+MTTuOBaRX1dTzzofBnwMRnSeRoB+Zt2IWn7y0EW+Dr1W63WnnnC6Ms5umyaEtR2jw
eJm0YEKv3BMQ/ceuA7m8umoHNXUehg1K58ql00lOcoU8fsf+XGpDrOF0qrCip5oZdyTo9yPzVs7h
c99cwZsPrcZT1wCAM9HJpbcvYcZF0zpV1q61u3nh16+Ss/MYpt9k+KRhXHH3Ci64blFPNF30M2+t
3cM/X99Mbd3ZIL5p5zF+8vXlDMpKDXrOoIGpGAYBG+YAJCd2fs+EguJKduzLZcTQdKaMH9rp8/sr
Cfr9zDd+fyuzls/isze3YRiw8KpzGNPJm2AVRRX8/Qf/oDi3pDkt90AeT//iOYaMHxzxtXtE/1JX
38Dr7+9sFfABjueV8vxbW/nOrRcHPW/e9JFMHDMInVMQNK+jfD4/9z+0mq17T+DzNX6CDMhIYsTg
dKpq3KQmJ7Bo7jiWLlKd6FX/IUG/Hxo/Zxzj53Q9MK9+7INWAf+M6vIa1j2zXoK+COujzYcpDLEn
xMFjofddMAyDb3z5Ah7653oOHi3E5zdJSXKyaO44Pr9iTofrv/+h1XzWZpOgkrIaSsrObiG6W+dR
WlHNFy+Lv+nFEvRFgMri0Pt/Vpb0/b1BRXTZwmxsY2lnP+NRwzL5nx9cxc4DuZwuqmLu1BFkDQi9
CbnX58Pt8ZLocmAYjUuIb993st02en1+3v9Yc+XSGbgiuKFQLJCgLwIMGpMdMq+rG6GI+LFo3jhe
XLWN/KLKgDw1tv3VVg3DYNbkETA59DENDT4ef3EjO/adpKqmniFZaSw5TzFuZBYNXn+H2llYUsWe
g6eYNz2+ngGQKZsiwCW3XsyoaYFjqFkjB7Lirkui0CIRS5wOG1+4bA7pKQmt0sePyuKGK+dFpI4/
/2Mdqz7cS35RJdW1Hg4dL+LxFzeypRPz+e02C5lpSRFpTyyJ+JW+UmobcOYj/qjW+rYWeVcCvwC8
wGNa60ciXb/oPmeik+/+/W6e/9+XOLTlMD6vn7EzR3H1964ge1RWtJsnYsCScxUL5ozh+de3UFvn
YcSQDFZcOBVnBJ4XOVVQztbdJwLSPQ0+9ug8BmYkUdxi/D4UNXYwY0fG3zfXiAZ9pZQLMLTWFwXJ
swMPAPOBGuBjpdTrWuvAW/Ui6gaPyea7j3wDr8eL32/icMXXuKfovtHDB3D7F8+LeLl7D+eHnM9f
WFrD3V+5kP998B28vtDDPONHDeTOOJ1+HOkr/ZlAolJqdVPZP9Vaf9qUNxk4rLUuA1BKbQAuAF4I
V2BGRiK2HnooKCsr9A2iWCb9ij39tW890a8500fidNhwe7yB9WUmc+lFU5g5dRi33Psk1bWBHw5z
po3ggV98EWuYG87tieXfV6SDfi3wW+DvwARglVJKaa29QCrQ8rG6KiCtvQLLymoj3MRGWVkpFBX1
v5ko0q/Y01/71lP9GpCaxNQJQ9i2t/UsHQOYO20kRUVVWLDwy+9dwf899SFHTzZOP7ZaDWZOGs4P
7ryE0tL2h39CiZXfV6gPpkgH/YM0Xs2bwEGlVAkwBDhJ4zh/y1akAOURrl8IEQe+fctF/PXp9ezW
p6hzN5CVmcyieeO4dvms5mPGjszi/h9fw/pNhykuq2bimGxmTh6O0c600f4u0kH/dmA6cLdSaiiN
V/f5TXn7gQlKqUygmsahnd9GuH4hRAzwNHj5ZGsOXp+fxfPGBZ0r7/P7+WjzYXJOFJOS5GLFBVNI
SW5cuyc9NZGf3L2CguJKCourGD86iwSXI6AMm9XKkvPi88nbUCId9B8Fnmgarzdp/BD4klIqWWv9
sFLqXuBdGqeKPqa1zotw/UKIPm7dpwd54e1tzYuovbhqG59bNpPLLprafEx1rZtf/201ew6eak57
b8N+vnr94lZr6w8amMqggcHX8umK4tJq3lm/j5paN6OGZbJ00STs/WyhwYgGfa21B7ixTfInLfLf
AN6IZJ1CiNiRd7qcx1/aSGVVfXNaQXEVT7+2iTHDM5k8fggAT7/6WauAD1BUWs3Tr37G7KkjunUT
NpRPth3h789/QlnF2fuI6zYd4sffuJT0lMSg5/j8fjDpkfb0lNhpqRAi5q3+aH+rgH9GXV0Dazce
bH6991B+wDEAx0+V8sSLG/GFmY7ZFQ1eH8++vqVVwAfQOQU88+rmgOMLiir53d/f5+s/e5av/tsz
/O+D73AozLpCfYkswyCE6DU1de4weWenV3q9vpDHvbl2D7t0HkvPm0R6SgKzpgwntc3Tv6H4/H4+
3HSIvYfysVktLJg5hjnTRrBxew65p4PPKzlw5HSr1253A/c//B45J4ub00rLj3PyVBm/vOdysgdE
bripJ0jQF0L0muFDMkLmDck+O4N73IiBnA6yds8ZJ06V8fiLGwHISEvkgnPGc8u1C8POzPH6fPzm
4ff5bOex5rQPPtGsvHAqI4eFbpfP33qB/1dW72wV8M84XVzJmx/s6ZEH0iJJhneEEL1m5YVTGR9k
KY+RQzO4cun05tfXrpjF4A7eoC2rqOW193Zx98+f5fk3t+D1Bf+WsGrd3lYBHxpX23z3o30MzEwm
KzM56HkT2rT35KnSkG0pKA79QdVRbo+XtRs1azfqoA+gdZdc6Qsheo3TYeMn31jOP1/fjM4pwG+a
TBidzZcun0NaiyGasSOz+Pl3VvLmmj2s/+xQq6GfUE4XV/Hcm1s5mlvC//vapQFX/XsPBr9P4Gnw
sX3PSS5fMo3nXt9KvaehOW/YoDQ+v3J2q+ND3dQFSE3u2DBTKO9v2M/L7+5oXqH0hbe3ce3yWSxb
HGbJ0U6SoC+E6FWZ6Ul86+aL2j1uaHY6X71hMQ1eH+9/fKDD5W/ZdZxte08wd9rZqZ0+n5/jeaGv
0P2myVXLZjJq6ADWf3aI6lo3Q7LT+NzS6QzIaP0N4JoVs3jrg90Bi7olJTpY2o1nAnJOFPPEy5uo
qT173yO/qJInX9nEuFFZjInQsuYS9IUQfdoVS6ax60AehR3cwMfnN9mtT7UK+o88t4HTIYZe7DYr
58wcDcCsKcOZNWV40OMKiit5Y81uSitqGJyVitVmpbCoEhMYMSSDzy2bzqRxgzvVt5bWfHKgVcA/
o7rGzfsfH+Cu6xd3ueyWJOgLIfq0UcMG8MO7lvHqe7s4erKYguLKgJurbbkcZ5/wraisY1ObsfyW
lpw7kelqWNjyck4U85tH3mt1c9nlsHHpBZOZP2M0MycPw2bt3kNc1UEC/hkdGd7qKLmRK4To88aP
zuYHdy3jL/95fatZPsFkpCZw6QVnx8APnyiivLIu6LEul51bPr+w3fpfemd7wGyieo+X7XtzmTph
SLcDPsCwwemh8waFzussCfpCiJiSEGZvh+REJ1+5ZkGrHbFGDE4nMci6PAADM5JxOtrfK+LIiaKg
6YUlVWzcntPu+R1x5ZLpjB4+ICB9zIgBXLFkWkTqABneEULEmOlqGIeOBQbhtJQEHvjZ58loswVi
9sBUZk4exsbtRwPOOWfGKCyW9lfdDLfZe4Iz+AdKZyW4HPzkG8t59o0tHDzauLeUGjOI66+cF3Qx
ua6SoC+EiCnXXzmP/MIKtuw+3rwJ+tDsNO68blFAwD/jmzddCIbBzv251NZ5yEhNZOHs0dx41fwO
1Tlp3GDyCioC0kcNzWT+jMhtrJ49IIXv3npxxMoLRoK+ECKm2G1WfvS1S9mt89hz8BSpyS6WLZoU
dpgmKdHJj756CYWlVZw6Xc64kVnNyzR3xFeuOYe8gnIOHDm7u+vAjGS+fPX8mFpsDcAwzfB3waOt
qKiqRxoYK7vfdJb0K/b01771t355fT4+2HiQwtIqbIaFFRdOJT21ew9j9aSsrJSg41ZypS+EEB1g
s1q5dPHkmP8wi63vJUIIIbpFgr4QQsQRCfpCCBFHIjqmr5SyA48BowEn8N9a69db5N8D3AmcmWT7
Na21jmQbhBBChBbpG7lfAUq01jcppTKBHcDrLfLnAjdrrbdGuF4hhBAdEOmg/wLwYtPPBtB2B4C5
wE+UUoOBt7TW90W4fiGEEGH0yDx9pVQKjVf4j2it/9ki/d+BvwCVwCvAX7XWb4Yry+v1mTZb9xcz
EkKIONM78/SVUiNoDOgPtgn4BvAHrXVF0+u3gNlA2KBfVlYbLrvLYn2ubSjSr9jTX/sm/YqurKyU
oOmRvpE7CFgNfEtrvaZNdiqwRyk1GagBltB401cIIUQvifSV/k+BDODnSqmfN6U9AiRprR9WSv0U
WAu4gTVa67cjXL8QQogwIhr0tdbfBb4bJv8p4KlI1imEEKLj5OEsIYSIIxL0hRAijkjQF0KIOCJB
Xwgh4ogEfSGEiCMS9IUQIo5I0BdCiDgiQV8IIeKIBH0hhIgjEvSFECKOSNAXQog4IkFfCCHiiAR9
IYSIIxL0hRAijkjQF0KIOCJBXwgh4ogEfSGEiCMR3xi9LynwnOajig8p8hbhsiQwI3Emc1PmRbtZ
QggRNZHeGN0CPAjMpHEf3Du11odb5F8J/ALwAo9prR+JZP0tnaw/wVNFT1DqLW1OO1C7j8KGQlZm
XtZT1QohRJ8W6eGdqwGX1vpc4MfA785kKKXswAPApcCFwFeVUoMiXH+ztRUftAr4AD58bK76lGpv
dU9VK4QQfVqkg/5i4B0ArfWnQMuxlMnAYa11mdbaA2wALohw/QDo2gPouv1B8yr9leyq2dET1UaF
aZr4TF+0myGEiBGRHtNPBSpavPYppWxaa2+QvCogrb0CMzISsdmsHW7AK7mv8VbhKhrMhpDHDMrI
BCArK6XD5YZimiZ7KvZwqv40U1MnMzxxeIfOy6vLZ0vpFuwWOxcMXEyyPblT9R6sOMRjx56g0F2E
iUmyLYkVg5dzOSsj0q++qL/2C/pv36RffU+kg34l0PJfw9IU8IPlpQDl7RVYVlbb4cpLGkpYffr9
sAF/sH0Io30KgKKiqg6XHUyxp4h/FT/LMfcx/PhxGk4mJ07l+qwbsRnB/2lN0+T10lfZXLWJerMe
gFV5q7k0YzkLU8/rUL257lwezP8THtPTnFbpreJfuS9iAPPti7vVr74oKyul27+vvqq/9k36FV2h
PpgiPbzzMXAZgFJqIbC7Rd5+YIJSKlMp5aBxaGdjJCvfVr2FWn/oD4lM2wCuyLwSq9Hxbw7hvFjy
L3LcOfjxA+A23eyo2cZbpa+HPGdr9RY2VK5vDvgAlf4KVpW9TWlDacjzWlpfsbZVwG9pTeFa/Ka/
E70QQsSTSAf9V4B6pdQnNN60vUcpdaNS6qta6wbgXuBdGoP9Y1rrvEhWboTpToYtk3uH/ZBJiVMi
UtfJ+hMcqz8aNE/XakzTDJq3r3YPJoF5Nf5qNlV17DPwlPtUyLwyTzk1/poOlSOEiD8RHd7RWvuB
r7dJPtAi/w3gjUjW2dK85Hl8VLkuaNCblTgbl8UVsbpKvCV48QbNq/fXYWJiYATkhbpCB/D4Q+e1
lGJN4bQ3P2ieYRi4jMj1UwjRv/SrJ3LT7RlcmHYxDsPRKn2cazzLMi6NaF0TXBNItaQGzRvoyMZi
BP+nHWQfHDTdwGBMwtgO1T0nZW7IPK/pZUPl+g6VI4SIP/3uidwl6csY5xrPtuotuE0PI50jOSdl
Ycgbq12VZEtmVvIc1leua5XuMlycm7Io5HkXpy3hUN1BTjW0HtmalDCF6YkzOlT3/JQF7Knezd76
PUHz99bu5uL0pR0qSwgRX/pd0AcY5RrNKNfoHq/nysyrSLGmsKd2NzW+GtKsacxIms3MpJkhz0m2
pXDH4LtYU/4+ee6TWA0b41zjWZpxCYYROBwUyrjE8SGDfo2v4zOehBDxpV8G/XAO1mo2VK6nMK8A
u+lgfMIELsu4ArvF3umyDMPg4vSlzEyaxWslr5BTf4Sj7hw+q97IotTFnJOysNXxpmmyvXobB+s1
AItSL2B28pxOBfszJrgm4jScuE13QN5A+4BOlyeEiA9xEfTdfjfbqrdQ4Clge802avxnl2HIbzhF
mbeUWwfd0aWy/aafZ4qe4rj7WHNanieXF4qfZ3v1Nq4acA2DHUMwTZNni55hW82W5uO2VH/Ggbr9
3JD15U4H/iHOoUxOnMqOmm2t0hOtiSwMM7wkhIhv/T7of1q5kTUV71HmDT0H/kDtAXLqDjM2YXyn
y99Rvb1VwD/DxORQ/UEeL3iUW7PvIN9zqlXAP2NbzRYmJ0xmdpibs6Fcn3UjqdZUdJ2m3l9Ltn0Q
K4YvY5RvYqfLEkLEh34d9E+783m77I2wD2wBeGkgpz4nZNCv89fh9teTak0LmJVT2FAQtuwSbzEf
Vn4AQaZvnqHrdZeCvs2w8bkBVwONQ0eGYZCVGRtPCwohoqNfB/1N1Z+2G/DPSLelB6RVeit4reQV
Dtcfxu2vZ5BjCOelLmJBi7H6bHv7C4Xme/IZ4hja8YZ3QVfuCwgh4k+/mqfflttf3/5BQIKRwNH6
o7xZ+jrFDUVA45XzM0VPsbN2BzX+arx4yfOc5LWSl9ldvbP53FnJsxnlHB22fKfhZKJLhcwPlyeE
EJHUr4N+R6+u68w6NlVvZF3FB/z51B/YVPUp+2r2cqT+cMCxHtPD5urNza8thoUvZ93MZNfUkOVP
SJjIrOTZzEkK3LVrTtI8ZiXP7lA7hRCiu/r18M7ClPMab7R6jnX4nBp/De+VvUuaNfjTtgAVvrJW
rzPtmdwx5C62Vm1lVdmblDfl27EzLWkGS9KXYTEsXJ91I5MSJnOwvnFliomuScxKnh3y6V0hhIi0
fh307RY7tw26g3fL3+G4+ygV3kq8phe3WU+yNQmf30+dWRdwXrmvjCpfZchyU63BtwGYmzKXaUnT
+KzqU+p8dUxMnMToFg+JWQwLc1LmBl1Gwe1382HFWk558nAYDqYmTWdm0qzOd1oIIcLo10EfGp+A
/fzALza/9ppeijxFjBk0hPv2/pY8T27Q83yE3o1qbnLozdWdFifnp13YqTbW+Gp47PTDHPccb07b
VrOV9+2EyqB8AAAdxklEQVRDsRk2rIaVsa5xXJK+vEsPkQkhxBn9Pui3ZTNsDHEOIcWewgjHiKBB
32m4cJvBbwKnWtKYlTwnom36oPz9VgH/jPyGs0soH3Mf5ZQnj9sH3SXDQXHK4s/BwRrAjpvPYVoy
o90kEYPiOnosTb+EIfbWN3tt2Ficej5DQ9wE7ombrifdJzp03IG6/eyq2dn+gaJ/MU0S/b8jzbyN
JPNvJJl/Js28Hqf/hWi3TMSguLvSbynDnsnXBt/N+sp1FHhO47S4mJk0k6lJ05lQN5GXi1+k0Nv4
8JUVK5MSJrMy4/KIt6MzV+7H6nNktk+ccZircJnPY7QYcrRSQqL5Vxr85+K3dGxfZiEgzoM+QLIt
mcsyrwhIH58wgXuGfZ/Pqj+j2lfFWNdYJiR0bD691/Syrnwtx+pzgMZVPy9KWxJyPH6MayyH6w91
qGyHxdmh40T/4TDXtwr4Z1iowMkr1PHtKLRKxKq4D/rh2C0OFqV2bpNxn+njiYJHOVC3vzntQP1+
jtbncPvgu4Ku678kbRkn3SdanRNMkiWZhSkd2zxd9CeBM8zOMELcexIilIgFfaVUGvA0kAo4gHu1
1hvbHPNHYDFwZnGYq7TWFZFqQ1/wWdWmoMH7YL1mU+VGFqWdH5Bnt9i5fdBdbKveyjH3UeyGnRpf
Lftr9zRPKU2zpnFJ+goy7XLzLt74jAlgbghIN7HgNWSoT3ROJK/07wXWaK3/oJRSwLNA22kuc4Hl
WuviCNbbp4TaLB3guPsYiwgM+tA4rj8vZT7zUuY3pxU3FLOjejs2i41zkhaQaEuMeHtF31fPV3Cw
EdvZ7aYB8LAIj7EkSq0SsSqSQf8B4MyOHjag1fdOpZQFmAA8rJQaBDyqtX4sgvX3CeG2Zezslo0D
7QNZlnFJd5skYpxpSafS/0cSzMexsQ8TO15jDnXG7SDTd0UnGaZpdvokpdQdwD1tkm/TWm9WSg0G
VgHf01p/2OKcFOC7wO8BK7AWuF1rvStcXV6vz7TZrJ1uY7TsKd/LHw79mQazoVW6DSvfmvBNZmeE
3kpRCCEiKOjSu10K+qEopaYDzwE/0FqvapNnBRK11lVNr+8HdmutnwpXZlFRVeQa2EJWVs+tO/9W
yRt8XPURHtMDgAMH56aex5VNa9/3pJ7sVzT1135B/+2b9Cu6srJSggb9SN7InQK8AFyntQ72BNFE
4Hml1GwaHwpbDDwZqfr7kssHXMnM5NnsrNkOmMxMms1w54hoN0tEic2/HYf5HtCA15iPx1gmwzIi
aiI5pn8f4AL+2Hgflwqt9VVKqXuBw1rr15VSTwGfAg3AP7TWeyNYf7tM02R/3T4KGwqZnTCVNLJ7
rK7hzuEMd8pDM/Euwf8XEsxnMJpud5nmK3jM96m2/C908h6PEJEQ0eGdnhCp4Z2ShhKeK3qGY+6j
mJjYDTsTEhRfzroJZz964ClWvnp2Vlf7ZfevwWGuw8CN11DUGzeCkdADLQxk9e8j1fwqliDz7Kv5
Pm7rjYD8zmJNrPQr1PBO3HzHfLnkBY66czBp/AxpMBvYV7uH10tejXLLRE9J9D9AivlTXLyNkzUk
mQ+S6v8mhr93/mCd5uqgAR/AxXO90gYh2oqLoF/gKSCn7kjQvEP1B/GZoZdRFrHJ4j+M03wFA2+r
dDs7cfF4L7Ui9PvKQgGG/3QvtUOIs+Ii6Fd4y2igIWhevb8er+kNmidil5P3sFATNM9m7umVNniM
Cwg1NmnBi4N1vdIOIVqKiztJo1xjyLBlUuYtDcgbZM/GYTii0CrRs8Jdz/TOcx9ey3z8vjSsBK40
YgKGWYbD/wZ+30pi+U/RMKtxms9iNU/hNzKp54uYlsHRbpYIIS6u9J0WJ/OS5mNp012n4eLclEUY
RtD7HSKGubkCP8G3tfQavbcNpZurgqabuEjkUVLM/4CSK0nw/wm6OanCMKtw+p/B5f87Fn/o5UAi
yeLPIdXfuM6/i9dJNJ8gzbwNuz9wrSDRN8Tu5UUnLc9cSbI1mV21O6n2VjE4KZuZjvnMSJ4R7aaJ
HuC3DKPOfwsJ5t+xUAs0Xl03cB51xm2Rrcz0YDfXA1YajMVgnF1Cu85yN1b/KRx81Dxt048DS8tV
SvyFJPAUfgbhNq7rUhMc/rdINB/ESuN9ggSewu27glrLD6AHL2oSzQexkdMqzUohCeZDmObyHqtX
dF3cBH2ARWnnN69yGSvTrkTX1VtuweM/B5f5FuDGa8zCY6wAo3PDOxZ/Lk5ewTBr8Roz8BjLmx+u
cvpfxWX+AxuN2116zXHUGnfRYGlaM8mwU239NTb/DmxsxWLm4eK1gDoM/DjMdbjpfNA3/MUkmn/C
ytl1DC1U4+Jf+Pxj8BuZgIMG49xO9z0s04ON3UGzbBzAbNgFjItcfSIi4iroi/jjt0ymlsldPt/p
f4UE8/+wUg6Aab5Ag/kWVZbfYjUPk2j+rvmbBICNIySZv6HSPwW/ZVhzutcyCy+zcPkfxwgximNQ
gcP/BjZzK1by8DGBeuM6/JZRYdvo4qVWAf9seX4S+QMWsw4T8JoTqTO+ToPlws7/QwRlNv0XjB/M
4JMnRHTFxZi+EF1h+CtJMB9uDvgABiYONpLgf5hk8+etAv4ZVkpw8WLQMr3MxiT4xAELRSSb/0EC
b+BgGwk8T5p5A4n+B8KP95uhN1k585yAAdg5SJL5Kwx/YeiyOsNw4mVq0CwvCsPRdmV10RdI0Bci
BAdvYCV4gHTyLjZCb2hvmJVB072WWXgI3P3MjxMrpQHLIlpw4zKfxWGuCjinuUxjDmYHZyRZKcTF
vzp0bEfUGXfio/VyIz4yqDNuwYjkUJKIGAn6QoRgNG8PESyvPGQegM8IPSRTbflf6rgeL2PxMRgc
F+JjaJi6fDjMdSHzG4zz8XBB2Pa0Ks+M3GZ1PstUKoyHqTNuws0l1PFFKo0HabDITdy+Ssb0hQjB
w3ISeAoLgVftBqH3pvUxkHrji6ELNpzUWn/Y/DIrMwVOfyFsW4ww++RiGFRb7sNnPord3Ap4ADd2
DgVvnzEmbF2dZVoGUcv3Ilqm6DkS9IUIwW8ZRr3/mqZVMls/tR1qEqSJlSp+3ulF3bxMxk7o/YR8
RjuzYAw7dcbXmz8aDP9p0sxvYG0zBNXAZNzGtZ1qm+hfJOgLEUad5Tv4/JNxmB8ANdjYjjXIzVto
nMfi5nJ81sWdrsfDApy8ggVPQF4DE6jjpk6VZ1oGU+W/jwTzMWzsBaw0MINa41tguDrdPtF/SNAX
Pcc0aRxqcPToA0I9zWO5BA+XYJglZPivDHmcn4HUWH4etizDX0gC/8Ri5mGSgtu4EtNcSBL/FxDw
Gx8mm0218d+YlgGdbrfPMolq7j878yeGfwciciToi8gzfSSYf8VhfohBGX6G4jYux23p2tOmvcnh
X43DfB+DKnyMot64Eb9lJAAmGfhJx0pB0HNNXGF3xLL4D5Ni/qj5QS4Ah7kGs+LigKda4cwQkrP7
69hIsBctSNAXEZfo/y0JLaYFWinDZh7E8Pupt9wQxZaFl+B/mATzcYzmK+7PcJgbqfL/Gp9lEhgW
3KwggSeDjun7GRK2/ETz760CPjQ+OYv7o5DnGMhT4yKyZMqmiCjDX46DDwLTacBhvt3tRcV6iuGv
wGm+1CLgN7KSS4J5divnOut38DIp4HwTJ27jirB1NI6tB2GWh5xn72NkOy0XonMiuTG6AeRC8zyx
jVrrn7Q55i7ga4AX+G+t9ZuRql/0DTb2BV0SAMDCKaAOSOzVNnWEg9Uh221Ft3pdaXmEJP/92NmM
QSU+RuE2rsJjCR/0wz1AZWLHaLPpio9s6o3rO9gDITomksM744BtWuugd7qUUoOB7wDzaNxAfYNS
6j2tdegnYETM8TEKP0lBNzAxSQf65n7EJslh8tosm2AkUmP9DzDrMajBJCPsWP4ZXmZg42TQPAv1
eBmKSTIW6vAynnrjRnyWaZ3phhDtimTQnwsMU0qtpfFy7h6tdctLpHOAj5uCvFspdRiYAWyOYBtE
lPktw2jwLcAZZIinwTg/sqs8RpDHWIbXfCzoDVUvs4OfZLgab952UK1xNzbzMLY23xzOsFBBueVx
TGNgh8sUorO6FPSVUncA97RJ/iZwn9b6BaXUYuBpYH6L/FRotYVQFYTY5aKFjIxEbLaeCRRZWSk9
Um60Rbtfft+vofKn4N4IVIMxEFxLSUz9CUlG168zerpf/vofQeV/gT+vKcUAx7kkpv+EJEtSBGpI
we/9PyheCQRu0WmhlswME4u9/7wvo/1e7Cmx3K8u/QVqrR8FHm2ZppRKpOmdrLXeoJQaqpQytNZn
7txVAi3/pVKgnQVMgLKy4A/CdFd/XU+/b/TLAvwKi3ECK8fwMhXTMwCKwywl0I7e6dd8DPNpnMaL
GGYlPmMKHu8yKPFDpGbRmOmkMhE7+wKyGphAZVkmGNH+/UVG33gvRl6s9CvUB1Mkh3f+HSgB7ldK
zQROtgj4AJ8B/6OUctE4sDsZ6J0dqkVU+C0j8cfY7BPTkko9t/dcBYaFeuM6rOZvGqdrNvGTiNv4
Qp8d/hL9RySD/q+Ap5VSl9N4xX8rgFLqXuCw1vp1pdSfgI9ovBT8N6116FWrRL/jN03WnjzKloI8
7BYLy0aOZ0bWoGg3q9d5LFdg+tNwmm9g4TR25xCqPStosFwc7aaJOGCYfXTe9BlFRVU90sBY+YrW
WX21X16/n59/soZ1J4/ib0pzWW3cOGk6X5sxP+y50Hf7FQn9tW/Sr+jKykoJ+ii2PJwlesULh/by
QYuAD1Dv8/Ks3o0uCz4/XggReRL0Ra/YWnAqaHqd18vq44d7uTVCxC8J+qJX+MMMI/r8fXuIUYj+
RIK+6BVTB2QHTXdYLJw/LLZm+AgRyyToi17x5UkzmJvdeh9YC7ByzETmDhoWnUYJEYdkaWXRK1w2
Gw9ctIIXDu5ld3EhDouF84aOZMXo8dFumhBxRYK+6DVOq42vTJ4Z7WYIEddkeEcIIeKIXOmLXuX1
+3l87za2FpzC4/MxMWMgN0+ZydDk1Gg3TYi4IEFf9BrTNPnZx2tYm3u0OW1vaRE7i0/zx4tWkp0Y
ek17IURkyPCO6DWfnDrJR3nHAtJzKsp4ev/O3m+QEHFIrvTjmNvn5V96D/tKi3BYrCweNoplI8di
GEGX7Oi27YX5eEM8pHWkvKxH6hRCtCZBP07Ve73c8+EqthXmN6etPn6YHUX5/HDe4h6pM8EW+u2W
YLe3e36lu55/6t0cKS8l0W5nyYixXDh8dARbKET/J0E/Tj25b0ergA/gB97MOcjK0ROYNjDySx5f
M2EKLx/eT3F9641xLMDioSPCnltYU8231r6FLitpTvvgRA5fnjyTr3dglU4hRCMZ049T+0oKg6bX
+7ysa3GjNZIyXQl8a9YCBre4YZtks3PthClcNW5y2HP/vGljq4AP4PH7eeXQPvJrQi9ze6yinL/s
+IwHtm7kk1Mn6OtLiQvR0+RKP05ZwozbW3rwWmDlmAmcP3wUbxw5QL3Xy0UjRjMmLbPd83YXBv+Q
Kve4ee/4EW6eMisg75n9u3h873aqGtwAvHhoDxePGMsvz70Yq0Wud0R8kqAfp2ZnD+GT/JMB6Uk2
O8t7eGmEZLuDGybN6NQ5tjBB+nhl4FbLJysreHzfNqoaPM1pXtPkvRNHmDogmxsmTe9U/UL0F3K5
E6dunDSDC4aNouX1vstq44ZJMxiX3v6Vd2+bN3RoyLz1ucfZlJ/bKu3Nowep8niCHr+5IC+ibRMi
lsiVfpyyWSz8+vxLWX38MNsL83FYrFwyejwzeuAGbiR8Z8F5rD96jANBdtmqbHDz8uF9LBgyvDnN
6/eFLKshTJ4Q/V3Egr5S6sfAiqaX6cBgrfXgNsf8EVgMnLnzdpXWuiJSbRCdYzEMVoyewIrRE6Ld
lHY1TtEcEzToA+RWVbZ6vWjYKP51cA8evz/g2MmZWT3SRiFiQcSCvtb6V8CvAJRSbwI/CnLYXGC5
1lo2RRWdNiw5FQMINv8m3eVq9XpO9hAuHTWeN48ebJU+bUA2N8tKnyKORXx4Ryl1LVCmtV7dJt0C
TAAeVkoNAh7VWj8W6fpF/3XxiDFMzsxiX2lRq3QLcPHwMQHH/9uCC5k2IJtPT+fi8flQmQP5yqQZ
JDucvdRiIfoeoyvzlpVSdwD3tEm+TWu9WSm1GbhBa324zTkpwHeB3wNWYC1wu9Z6V7i6vF6fabNZ
O91G0T/p4iL+/cMP2J5/iga/n8HJyVyjpvCD8xb32PIRQsSooH8QXQr6oSilpgB/1FpfEiTPCiRq
rauaXt8P7NZaPxWuzKKiqh55miYrK4WiotAP9cSqeOiXaZrsKS6koLaaedlD+bQgl70lhSTaHVw7
bjKDkrq2Wqdpmrx25AAf5h6jusHDyJQ0rlfTmZAxIJJdCRAPv7P+JFb6lZWVEjToR3p4ZxmwKkTe
ROB5pdRsmp68B56McP2ijzFNk435JzleWcGMrEEhN0gPx+Pz8dyeXRwrKmNO9hBmZw9hetYgxnsz
+eH61a2mYL5x5ADfmb2wSzen/7xjE88e2MWZW7+7igvYWniK+xYtY3IX2i1EXxTpoK+A91olKHUv
cFhr/bpS6ingU6AB+IfWem+E6xd9SF5VJf+1aR27igvwmSYuq42FQ4bzy3OX4Aqz+FpL2wtP8evN
Gzja9ACWw2Jh0dCR/Od5S3lo15aAOfcl9XU8vGsLFw0f0+E6oHFtn7eOHqTtXJ/8mmqePrCb/1m0
tMNlCdGXRTToa62/GSTt9y1+/g3wm0jWKfqu+7dsYHvR6ebXjev6HOMP2zfy4/nnt3u+z+/ngW0b
mwM+NK63szb3GA/t2szm07lBz8urqeLtowe5dsKUDrd1Xe4xyt31QfMOhZgmKkQskoezRI84VlHG
jhYBv6XNp/Pw+f1B1785UVnOu8ePAJDqcAYsstZcRkEehXU1IesPlxdMmtMVMi/B1v6yz0LECgn6
Iiif388/9u3gk/yTVDd4GJ2aznVqOrOyzj5vZ5omJsEXbztdW029zxu07OoGDx6/j4Q2Qf+hXVt4
4eCe5vVynNbQs7ZqvV6M4JMTmhoXpnNBLB05lif37eBIRWlA3txBoZeAECLWSNAXQf168wZeyznQ
/Dqnooxdxaf57/OWMSE9kz/t2MS2wnzqvQ2MTxvADZOmsWDI2TXxpw8cxJDEZPJrqwPKHpWSjsva
+q23KT+Xp/fvxNNiiQS3L/RyCePTMsirrqLC4w6an+EKfeV+xqGyEp4/uJu8qipSHE4WDRtBg8/H
ierGh8TthoVzh46Q9fpFvyJBXwQ4WlHGmhNHAtKL6+r4l95DjdfDptNnb6AW1dVysLyEXy1exoym
bwJJdgfLx0zgib3bW5XhsFq5dsLkgDn1a07mtAr44WQnJHH9pBmszz3GwfLA4Z/BiclcPlaFLWNH
0Wl+8ckaCmrPDgM58i3cNHkmaU4XFW43M7MGc87gYTL/X/QrEvRFgI35J6n2NgTN21taSHFdbUB6
SX0tLx3a1xz0AcakpOO0WltdsZt+M+gNU7c3+FAQwNTMLDKSEymtrmFkSjrXqWlMGZDNpIyBHC4v
4bPTec2jORlOF3dNm0uy3RG2j8/s39kq4EPjTeI3jx7i2cu+QFI75wsRqyToiwADXYkh84rravGF
eKAvt7oSj8/H8wf3sKvoNDuK8gOGaBpMP68e3s/nx0/B3mLMfvKALN45frhtkQCsHDORuxctDHgg
xmWz8cCFK3nvxBH2FBeSaLdzzbjJDElOabePh4J8QwAoqK3mw9xjXDZmYrtlCBGLJOiLAEtGjuWp
Azs5GGTmTKiAD5DmcPLD9e/yaYiplGccrSznUHkpUwacXe3y2vFT+DD3WMC+vfMGDeXqcZNClmW1
WLq0UqjDEvwmsUH4mTxCxDrZREUEsFks/HDuIiZ2YvkBG5DicLYb8AFcVivpbQKrw2rl9xeu4NYp
s5ibPYS52UO5depsfnfBilbfCCJldnbwGTkqYyDnDgm/SbsQsUyu9EVQM7IG8/il13DD2y9woqr9
LQ9UZhZeM3Dt+mBmZw9haJAhmASbnW/MPKfTbe2Kb886h7zqCrYWnGp+CndkShrfmb0w7P7BQsQ6
CfoiJJvFwujU9A4FfcNiYDPCf3E0gCkDsrh3znkRamHXJTuc/Oniy1l78ii6tJgMl4urx0+WB7FE
vydBX4S1ZORYPs0/GXQHqpbOzGlfffxwwPo1NsNg+agJLBw6nKUjxgZ9EjcaLIbB0pFjWTpybLSb
IkSvkaAvwlo5egJ5VZW8nnMgYIpjS7Ozh7B81Hi2Fpxi1dFDNDQN9TitVr4wYSrfmb2wt5oshAhD
gr5o153T53L9pOmszz3G/pJC3j12hIqGxidhLcC5Q0dy29Q5GIbBT8+5gGUjx/Fx3gkMCywdMY4Z
WX1zs3Uh4pEEfdEhyXYHl42ZyGVjJnLDpJm8eng/db4GZmUN4eIRY5pvfhqGwYIhw1kwZHiUWxza
GzkHWH38CKV1dQxJTuHKsYoLh4+OdrOE6BUS9EWnDU1O4e5ZvTPLJtKe2LudR/dsbb5HcbiilG0F
p/jR/MVd2nhFiFjTN+6oCdEL6r1e3sjRATela7wNvHRoH5HcOlSIvkqCvogbB8qKyK2uDJqXU1FG
ddOSzkL0ZxL0RdwY4EoMWNL5jGS7A2eIPCH6k269y5VS1wBf1Frf2PR6IfBHwAus1lr/ss3xCcDT
QDZQBdyitS7qThuE6KgRKWnMzh7MxvzApSLmDRqKoweWexCir+nylb5S6o/AfW3K+BtwI7AYWKCU
mt3mtG8Au7XW5wP/AH7W1fqF6IofzF3EjIGDmt+0dsPCuUOGc8+cc6PaLiF6S3eu9D8BXgW+BqCU
SgWcWusjTa/fBZYBLXfRWAzc3/TzKuDn3ahfiE4bnpLGw8s+x4e5xzhRVcGUzCzmDR4W7WYJ0Wva
DfpKqTuAe9ok36a1fl4pdVGLtFSg5V2yKqDt8+2pQEWL/LT26s/ISMRm65mv3VlZ7a+7HoukX+37
YvaMiJUVCfI7iy2x3K92g77W+lHg0Q6UVQm0/JdIAcrDHBMsP0BZWeAuTZGQlZUSsClHfyD9ij39
tW/Sr+gK9cEUsdk7WutKwKOUGqeUMoDlwEdtDvsYuKzp55VB8oUQQvSgSM9R+zrwDGClcfbOJgCl
1GrgCuCvwJNKqQ2Ah8abvkIIIXpJt4K+1nodsK7F60+BgOUUtdaXNv3oAb7YnTqFEEJ0nTycJYQQ
ccSQ9UaEECJ+yJW+EELEEQn6QggRRyToCyFEHJGgL4QQcUSCvhBCxBEJ+kIIEUficteIzu4DEEuU
UmnAc0Ay4Aa+orU+Hd1WdZ9Sygr8HpgHOIH/0Fq/Gd1WRZZSahKwCRikta6Pdnu6q+m9+DSNCy06
gHu11huj26quU0pZgAeBmTT+bd2ptT4c3VZ1Xtxd6XdxH4BYcitn9yx4HvhhdJsTMTcBdq31IuAq
YHyU2xNRTUuT/47GYNJf3Aus0VpfSOP78i/RbU63XQ24tNbnAj+m8fcVc+Iu6NO4D8A3zrxouQ+A
1toEzuwDEKt2c3Yl01SgIYptiaTlQJ5S6i3gEeCNKLcnYpoWKHwY+CnQM8vKRscDwENNP9uAWP/2
shh4B5qXnJkX3eZ0Tb8d3onwPgB9Uog+fhO4VCm1D8gEzu/1hnVTiH4V0Rg0rgAuAB5v+n9MCdG3
48BzWuudSqkotKr7wvy9bVZKDaZxmOd7vd+yiGq5HwiATyll01p7o9Wgrui3QT/C+wD0ScH6qJR6
Gbhfa/2QUmoG8BLQt3YMaUeIfj0HvNn0bexDpdTEqDSum0L07TBwR1PgHAysJsY+0EL9vSmlptN4
j+kHWusPe71hkdU2VlhiLeBDfA7vtNLBfQBiSRlnr0YKabw66Q820LQXg1JqJnAius2JHK31eK31
RVrri4DTwKXtnBITlFJTgBeAG7XWq6Ldngho3g+kafLH7ug2p2v67ZV+JwXdByBG/Rz4u1LqbsAO
3BXl9kTKI8BflVKfAgaNvzPRt90HuIA/Ng1bVWitr4puk7rlFeASpdQnNL4Hb4tye7pEVtkUQog4
EvfDO0IIEU8k6AshRByRoC+EEHFEgr4QQsQRCfpCCBFHJOgLIUQckaAvhBBxRIK+EELEkf8PzkM8
pEwGjhQAAAAASUVORK5CYII=
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">blob_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span><span class="s1">&#39;x&#39;</span><span class="p">:</span><span class="n">blob_data</span><span class="p">[:,</span><span class="mi">0</span><span class="p">],</span><span class="s1">&#39;y&#39;</span><span class="p">:</span><span class="n">blob_data</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span><span class="s1">&#39;Label&#39;</span><span class="p">:</span><span class="n">blob_labels</span><span class="p">})</span>
<span class="n">blob_df</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[3]:</div>


<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>-0.530153</td>
      <td>3.996292</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>-4.974868</td>
      <td>-4.763579</td>
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
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Circles">Circles<a class="anchor-link" href="#Circles">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Create one circle inside of another one.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.datasets</span> <span class="k">import</span> <span class="n">make_circles</span>

<span class="n">circles_data</span><span class="p">,</span> <span class="n">circles_labels</span> <span class="o">=</span> <span class="n">make_circles</span><span class="p">(</span><span class="n">n_samples</span><span class="o">=</span><span class="mi">100</span><span class="p">,</span>
                                      <span class="n">noise</span><span class="o">=.</span><span class="mi">03</span><span class="p">,</span>
                                      <span class="n">factor</span><span class="o">=.</span><span class="mi">5</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">circles_data</span><span class="p">[:,</span><span class="mi">0</span><span class="p">],</span>
            <span class="n">circles_data</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span> 
            <span class="n">c</span> <span class="o">=</span> <span class="n">circles_labels</span><span class="p">,</span>
            <span class="n">cmap</span><span class="o">=</span><span class="s1">&#39;viridis&#39;</span><span class="p">);</span>
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
6XUbp94AAAAASUVORK5CYII=
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">circles_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span><span class="s1">&#39;x&#39;</span><span class="p">:</span><span class="n">circles_data</span><span class="p">[:,</span><span class="mi">0</span><span class="p">],</span><span class="s1">&#39;y&#39;</span><span class="p">:</span><span class="n">circles_data</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span><span class="s1">&#39;Class&#39;</span><span class="p">:</span><span class="n">circles_labels</span><span class="p">})</span>
<span class="n">circles_df</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[5]:</div>


<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.964222</td>
      <td>0.223756</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.775693</td>
      <td>0.734270</td>
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
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Regression">Regression<a class="anchor-link" href="#Regression">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.datasets</span> <span class="k">import</span> <span class="n">make_regression</span>

<span class="n">regression_data</span><span class="p">,</span> <span class="n">regression_values</span> <span class="o">=</span> <span class="n">make_regression</span><span class="p">(</span><span class="n">n_samples</span><span class="o">=</span><span class="mi">100</span><span class="p">,</span>
                                                     <span class="n">n_features</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
                                                     <span class="n">n_informative</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
                                                     <span class="n">noise</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">regression_data</span><span class="p">[:,</span><span class="mi">0</span><span class="p">],</span>
            <span class="n">regression_values</span><span class="p">,</span>
            <span class="n">cmap</span><span class="o">=</span><span class="s1">&#39;viridis&#39;</span><span class="p">);</span>
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
K9MOOwMAAAAASUVORK5CYII=
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">regression_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span><span class="s1">&#39;Feature 1&#39;</span><span class="p">:</span><span class="n">regression_data</span><span class="p">[:,</span><span class="mi">0</span><span class="p">],</span><span class="s1">&#39;Value&#39;</span><span class="p">:</span><span class="n">regression_values</span><span class="p">})</span>
<span class="n">regression_df</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[7]:</div>


<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature 1</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.523895</td>
      <td>26.287656</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.365088</td>
      <td>17.644997</td>
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
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Biclusters">Biclusters<a class="anchor-link" href="#Biclusters">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.datasets</span> <span class="k">import</span> <span class="n">make_biclusters</span>

<span class="n">biclusters_data</span><span class="p">,</span> <span class="n">biclusters_rows</span><span class="p">,</span> <span class="n">biclusters_cols</span> <span class="o">=</span> <span class="n">make_biclusters</span><span class="p">(</span><span class="n">shape</span> <span class="o">=</span> <span class="p">(</span><span class="mi">100</span><span class="p">,</span><span class="mi">2</span><span class="p">),</span>
                                                                   <span class="n">n_clusters</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span>

<span class="n">biclusters_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span><span class="s1">&#39;x&#39;</span><span class="p">:</span><span class="n">biclusters_data</span><span class="p">[:,</span><span class="mi">0</span><span class="p">],</span>
                              <span class="s1">&#39;y&#39;</span><span class="p">:</span><span class="n">biclusters_data</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span>
                              <span class="s1">&#39;Row Class 1&#39;</span><span class="p">:</span><span class="n">biclusters_rows</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span>
                              <span class="s1">&#39;Row Class 2&#39;</span><span class="p">:</span><span class="n">biclusters_rows</span><span class="p">[</span><span class="mi">1</span><span class="p">]})</span>
<span class="n">biclusters_df</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[8]:</div>


<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Row Class 1</th>
      <th>Row Class 2</th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>True</td>
      <td>False</td>
      <td>80.816475</td>
      <td>80.816475</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>False</td>
      <td>80.816475</td>
      <td>80.816475</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">biclusters_cols</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[9]:</div>



<div class="output_text output_subarea output_execute_result">
<pre>array([[ True,  True],
       [False, False]], dtype=bool)</pre>
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
<h3 id="Classification">Classification<a class="anchor-link" href="#Classification">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.datasets</span> <span class="k">import</span> <span class="n">make_classification</span>

<span class="n">classification_data</span><span class="p">,</span> <span class="n">classification_class</span> <span class="o">=</span> <span class="n">make_classification</span><span class="p">(</span><span class="n">n_samples</span><span class="o">=</span><span class="mi">100</span><span class="p">,</span>
                                                                 <span class="n">n_features</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
                                                                 <span class="n">n_informative</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
                                                                 <span class="n">n_redundant</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
                                                                 <span class="n">n_classes</span><span class="o">=</span><span class="mi">3</span><span class="p">)</span>

<span class="n">classification_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span><span class="s1">&#39;Feature 1&#39;</span><span class="p">:</span><span class="n">classification_data</span><span class="p">[:,</span><span class="mi">0</span><span class="p">],</span>
                                  <span class="s1">&#39;Feature 2&#39;</span><span class="p">:</span><span class="n">classification_data</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span>
                                  <span class="s1">&#39;Feature 3&#39;</span><span class="p">:</span><span class="n">classification_data</span><span class="p">[:,</span><span class="mi">2</span><span class="p">],</span>
                                  <span class="s1">&#39;Feature 4&#39;</span><span class="p">:</span><span class="n">classification_data</span><span class="p">[:,</span><span class="mi">3</span><span class="p">],</span>
                                  <span class="s1">&#39;Class&#39;</span><span class="p">:</span><span class="n">classification_class</span><span class="p">})</span>
<span class="n">classification_df</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[10]:</div>


<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>Feature 1</th>
      <th>Feature 2</th>
      <th>Feature 3</th>
      <th>Feature 4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3.062990</td>
      <td>0.117025</td>
      <td>4.274497</td>
      <td>2.940409</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2.680943</td>
      <td>-0.492702</td>
      <td>3.652454</td>
      <td>1.894190</td>
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
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Multilabel-Classification">Multilabel Classification<a class="anchor-link" href="#Multilabel-Classification">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.datasets</span> <span class="k">import</span> <span class="n">make_multilabel_classification</span>

<span class="n">multilabel_classification_data</span><span class="p">,</span> <span class="n">multilabel_classification_classes</span> <span class="o">=</span> <span class="n">make_multilabel_classification</span><span class="p">(</span><span class="n">n_samples</span><span class="o">=</span><span class="mi">100</span><span class="p">,</span>
                                                                                                  <span class="n">n_features</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
                                                                                                  <span class="n">n_classes</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span>
                                                                                                  <span class="n">n_labels</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span>

<span class="n">multilabel_classification_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span><span class="s1">&#39;Feature 1&#39;</span><span class="p">:</span><span class="n">multilabel_classification_data</span><span class="p">[:,</span><span class="mi">0</span><span class="p">],</span>
                                             <span class="s1">&#39;Feature 2&#39;</span><span class="p">:</span><span class="n">multilabel_classification_data</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span>
                                             <span class="s1">&#39;Feature 3&#39;</span><span class="p">:</span><span class="n">multilabel_classification_data</span><span class="p">[:,</span><span class="mi">2</span><span class="p">],</span>
                                             <span class="s1">&#39;Feature 4&#39;</span><span class="p">:</span><span class="n">multilabel_classification_data</span><span class="p">[:,</span><span class="mi">3</span><span class="p">],</span>
                                             <span class="s1">&#39;Class 1&#39;</span><span class="p">:</span><span class="n">multilabel_classification_classes</span><span class="p">[:,</span><span class="mi">0</span><span class="p">],</span>
                                             <span class="s1">&#39;Class 2&#39;</span><span class="p">:</span><span class="n">multilabel_classification_classes</span><span class="p">[:,</span><span class="mi">1</span><span class="p">]})</span>
<span class="n">multilabel_classification_df</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[11]:</div>


<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class 1</th>
      <th>Class 2</th>
      <th>Feature 1</th>
      <th>Feature 2</th>
      <th>Feature 3</th>
      <th>Feature 4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>19.0</td>
      <td>20.0</td>
      <td>4.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>21.0</td>
      <td>14.0</td>
      <td>3.0</td>
      <td>7.0</td>
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
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Moons">Moons<a class="anchor-link" href="#Moons">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.datasets</span> <span class="k">import</span> <span class="n">make_moons</span>

<span class="n">moons_data</span><span class="p">,</span> <span class="n">moons_labels</span> <span class="o">=</span> <span class="n">make_moons</span><span class="p">(</span><span class="n">n_samples</span><span class="o">=</span><span class="mi">100</span><span class="p">,</span><span class="n">noise</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">moons_data</span><span class="p">[:,</span><span class="mi">0</span><span class="p">],</span>
            <span class="n">moons_data</span><span class="p">[:,</span><span class="mi">1</span><span class="p">],</span>
            <span class="n">c</span><span class="o">=</span><span class="n">moons_labels</span><span class="p">,</span>
            <span class="n">cmap</span><span class="o">=</span><span class="s1">&#39;viridis&#39;</span><span class="p">);</span>
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
gg==
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
<h3 id="Spd-Matrix-(Positive-Definite-Matrix)">Spd Matrix (Positive Definite Matrix)<a class="anchor-link" href="#Spd-Matrix-(Positive-Definite-Matrix)">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.datasets</span> <span class="k">import</span> <span class="n">make_spd_matrix</span>

<span class="n">spd_matrix</span> <span class="o">=</span> <span class="n">make_spd_matrix</span><span class="p">(</span><span class="n">n_dim</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span>

<span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">data</span><span class="o">=</span><span class="n">spd_matrix</span><span class="p">,</span> <span class="n">annot</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s1">&#39;viridis&#39;</span><span class="p">);</span>
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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[14]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">spd_matrix</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
<div class="prompt output_prompt">Out[14]:</div>


<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.758847</td>
      <td>0.019034</td>
      <td>0.092914</td>
      <td>0.010491</td>
      <td>0.003805</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.019034</td>
      <td>0.614790</td>
      <td>-0.070875</td>
      <td>0.620051</td>
      <td>0.048334</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.092914</td>
      <td>-0.070875</td>
      <td>0.678010</td>
      <td>-0.178302</td>
      <td>0.189515</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.010491</td>
      <td>0.620051</td>
      <td>-0.178302</td>
      <td>4.917203</td>
      <td>-1.715514</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.003805</td>
      <td>0.048334</td>
      <td>0.189515</td>
      <td>-1.715514</td>
      <td>1.408395</td>
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
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Others-data-generation-functions-available-in-sklearn">Others data generation functions available in sklearn<a class="anchor-link" href="#Others-data-generation-functions-available-in-sklearn">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<ul>
<li>datasets.make_checkerboard()</li>
<li>datasets.make_friedman1()</li>
<li>datasets.make_friedman2()</li>
<li>datasets.make_friedman3()</li>
<li>datasets.make_gaussian_quantiles()</li>
<li>datasets.make_hastie_10_2()</li>
<li>datasets.make_low_rank_matrix()</li>
<li>datasets.make_s_curve()</li>
<li>datasets.make_sparse_coded_signal()</li>
<li>datasets.make_sparse_spd_matrix()</li>
<li>datasets.make_sparse_uncorrelated()</li>
<li>datasets.make_swiss_roll()</li>
</ul>

</div>
</div>
</div>
 
