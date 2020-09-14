# DecisionTree
## Data

Each data set is divided into three sub-sets: the training set, the validation set and the test set. Data sets are in CSV format. Each line is a training (or test) example that contains a list of attribute values separated by a comma. The last attribute is the class-variable. Assume that all attributes take values from the domain {0,1}. 

The datasets are generated synthetically by randomly sampling solutions and non-solutions (with solutions having class “1” and nonsolutions having class “0”) from a Boolean formula in conjunctive normal form (CNF). I randomly generated five formulas having 500 variables and 300, 500, 1000, 1500 and 1800 clauses (where the length of each clause equals 3) respectively and sampled 100, 1000 and 5000 positive and negative examples from each formula. I am using the following naming convention for the files. Filenames train∗, test∗ and valid∗ denote the training, test and validation data respectively. train c[i] d[j].csv where i and j are integers contains training data having j examples generated from the formula having i clauses. For example, the file with filename train c500 d100 contains 100 examples generated from the formula having 500 clauses.

## How to run

```python
python __main__.py -h

usage: __main__.py [-h] -m {1,2} -c
                   {300,500,1000,1500,1800} -s
                   {100,1000,5000} [-d] [-r]

Instructions:

optional arguments:
  -h, --help            			show this help message and exit
  -m {1,2}              			impurity heuristic: 1 for entrophy, 2 for variance
  -c {300,500,1000,1500,1800}	clause count
  -s {100,1000,5000}    			sample count
  -d                    			depth-based pruning
  -r                    			reduced error pruning
```

For exapmple, Decision tree learner with Entropy as the impurity heuristic and reduced error pruning using data set with 500 clauses and 1000 examples:

```python
python __main__.py -m 1 -c 500 -s 1000 -r
```

##  Results

[id3 full test](/id3.ipynb)

```
+--------+--------+----------+---------------+---------------+----------+---------------+---------------+
| clause | sample | entrophy | entrophy(REP) | entrophy(DBP) | variance | variance(REP) | variance(DBP) |
+--------+--------+----------+---------------+---------------+----------+---------------+---------------+
|  300   |  100   |  0.585   |      0.58     |     0.575     |  0.575   |     0.585     |     0.575     |
|  300   |  1000  |  0.595   |     0.6295    |     0.6235    |  0.5975  |     0.6365    |     0.642     |
|  300   |  5000  |  0.6207  |     0.6422    |     0.632     |  0.6173  |     0.6251    |     0.6327    |
|  500   |  100   |  0.645   |      0.62     |      0.61     |  0.605   |     0.605     |     0.615     |
|  500   |  1000  |  0.638   |     0.683     |     0.6675    |   0.65   |     0.6795    |     0.6775    |
|  500   |  5000  |  0.6719  |     0.6944    |     0.6824    |  0.6735  |     0.6927    |     0.6798    |
|  1000  |  100   |   0.71   |      0.68     |      0.71     |   0.71   |      0.68     |      0.71     |
|  1000  |  1000  |  0.7765  |     0.7935    |     0.7865    |  0.7615  |     0.7935    |     0.791     |
|  1000  |  5000  |  0.7694  |     0.7956    |     0.7821    |  0.7728  |     0.7953    |     0.7838    |
|  1500  |  100   |  0.815   |      0.87     |      0.82     |   0.83   |      0.88     |      0.83     |
|  1500  |  1000  |  0.8925  |     0.9245    |     0.8925    |  0.884   |     0.9195    |     0.884     |
|  1500  |  5000  |  0.9056  |     0.9237    |     0.9056    |  0.9041  |     0.9259    |     0.9041    |
|  1800  |  100   |   0.91   |      0.94     |      0.91     |   0.91   |      0.94     |      0.91     |
|  1800  |  1000  |  0.967   |     0.976     |     0.967     |  0.9595  |     0.9725    |     0.9595    |
|  1800  |  5000  |  0.9793  |     0.9848    |     0.9793    |  0.9757  |     0.9843    |     0.9757    |
+--------+--------+----------+---------------+---------------+----------+---------------+---------------+
```

[id3 vs random forest](/id3_vs_random_forest.ipynb)

```
+--------+--------+--------+---------------+
| clause | sample |  id3   | random forest |
+--------+--------+--------+---------------+
|  300   |  100   | 0.585  |     0.715     |
|  300   |  1000  | 0.595  |     0.821     |
|  300   |  5000  | 0.6207 |     0.8844    |
|  500   |  100   | 0.645  |     0.755     |
|  500   |  1000  | 0.638  |     0.921     |
|  500   |  5000  | 0.6719 |     0.9419    |
|  1000  |  100   |  0.71  |      0.99     |
|  1000  |  1000  | 0.7765 |     0.9845    |
|  1000  |  5000  | 0.7694 |     0.9918    |
|  1500  |  100   | 0.815  |      1.0      |
|  1500  |  1000  | 0.8925 |      1.0      |
|  1500  |  5000  | 0.9056 |     0.9997    |
|  1800  |  100   |  0.91  |      1.0      |
|  1800  |  1000  | 0.967  |      1.0      |
|  1800  |  5000  | 0.9793 |      1.0      |
+--------+--------+--------+---------------+
```

