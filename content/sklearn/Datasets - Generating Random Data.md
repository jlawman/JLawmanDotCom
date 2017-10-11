
Scikit-learn has a number of methods to generate data in its datasets module. Below are some examples of the options available.


```python
#Import pandas and plotting libraries to visualize data
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
```

### Blobs


```python
from sklearn.datasets import make_blobs

blob_data, blob_labels = make_blobs(n_samples=100,
                                    n_features=2,
                                    centers=5,
                                    cluster_std=.8)

plt.scatter(blob_data[:,0],
            blob_data[:,1], 
            c = blob_labels,
            cmap='viridis');
```


![png](output_3_0.png)



```python
blob_df = pd.DataFrame({'x':blob_data[:,0],'y':blob_data[:,1],'Label':blob_labels})
blob_df.head(2)
```




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



### Circles

Create one circle inside of another one.


```python
from sklearn.datasets import make_circles

circles_data, circles_labels = make_circles(n_samples=100,
                                      noise=.03,
                                      factor=.5)

plt.scatter(circles_data[:,0],
            circles_data[:,1], 
            c = circles_labels,
            cmap='viridis');
```


![png](output_7_0.png)



```python
circles_df = pd.DataFrame({'x':circles_data[:,0],'y':circles_data[:,1],'Class':circles_labels})
circles_df.head(2)
```




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



### Regression


```python
from sklearn.datasets import make_regression

regression_data, regression_values = make_regression(n_samples=100,
                                                     n_features=1,
                                                     n_informative=1,
                                                     noise=5)

plt.scatter(regression_data[:,0],
            regression_values,
            cmap='viridis');
```


![png](output_10_0.png)



```python
regression_df = pd.DataFrame({'Feature 1':regression_data[:,0],'Value':regression_values})
regression_df.head(2)
```




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



### Biclusters


```python
from sklearn.datasets import make_biclusters

biclusters_data, biclusters_rows, biclusters_cols = make_biclusters(shape = (100,2),
                                                                   n_clusters=2)

biclusters_df = pd.DataFrame({'x':biclusters_data[:,0],
                              'y':biclusters_data[:,1],
                              'Row Class 1':biclusters_rows[0],
                              'Row Class 2':biclusters_rows[1]})
biclusters_df.head(2)
```




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




```python
biclusters_cols
```




    array([[ True,  True],
           [False, False]], dtype=bool)



### Classification


```python
from sklearn.datasets import make_classification

classification_data, classification_class = make_classification(n_samples=100,
                                                                 n_features=4,
                                                                 n_informative=3,
                                                                 n_redundant=1,
                                                                 n_classes=3)

classification_df = pd.DataFrame({'Feature 1':classification_data[:,0],
                                  'Feature 2':classification_data[:,1],
                                  'Feature 3':classification_data[:,2],
                                  'Feature 4':classification_data[:,3],
                                  'Class':classification_class})
classification_df.head(2)
```




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



### Multilabel Classification


```python
from sklearn.datasets import make_multilabel_classification

multilabel_classification_data, multilabel_classification_classes = make_multilabel_classification(n_samples=100,
                                                                                                  n_features=4,
                                                                                                  n_classes=2,
                                                                                                  n_labels=2)

multilabel_classification_df = pd.DataFrame({'Feature 1':multilabel_classification_data[:,0],
                                             'Feature 2':multilabel_classification_data[:,1],
                                             'Feature 3':multilabel_classification_data[:,2],
                                             'Feature 4':multilabel_classification_data[:,3],
                                             'Class 1':multilabel_classification_classes[:,0],
                                             'Class 2':multilabel_classification_classes[:,1]})
multilabel_classification_df.head(2)
```




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



### Moons


```python
from sklearn.datasets import make_moons

moons_data, moons_labels = make_moons(n_samples=100,noise=0)

plt.scatter(moons_data[:,0],
            moons_data[:,1],
            c=moons_labels,
            cmap='viridis');
```


![png](output_20_0.png)


### Spd Matrix (Positive Definite Matrix)


```python
from sklearn.datasets import make_spd_matrix

spd_matrix = make_spd_matrix(n_dim=5)

sns.heatmap(data=spd_matrix, annot=True, cmap='viridis');
```


![png](output_22_0.png)



```python
pd.DataFrame(spd_matrix)
```




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



### Others data generation functions available in sklearn

 - datasets.make_checkerboard()
 - datasets.make_friedman1()
 - datasets.make_friedman2()
 - datasets.make_friedman3()
 - datasets.make_gaussian_quantiles()
 - datasets.make_hastie_10_2()
 - datasets.make_low_rank_matrix()
 - datasets.make_s_curve()
 - datasets.make_sparse_coded_signal()
 - datasets.make_sparse_spd_matrix()
 - datasets.make_sparse_uncorrelated()
 - datasets.make_swiss_roll()
