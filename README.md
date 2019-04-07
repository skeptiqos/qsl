# qsl
q statistical learning

A library for statistical and machine learning implemented in kdb+/q.

## dtl 

* decision tree learning
* random forests

### grow a Tree

let
 
* X and Y be the predictor and predicted variables respectively
* rule be a function or logical operator applied to a set of ordinal numbers
* classes be the classification of the predicted variable

Then to learn a tree 
```
.dtl.learnTree `x`y`rule`classes!(X;Y;rule;classes)
```
A tree is grown by maximising information gain through entropy reduction on each split and results into a treetable, a q datastructure illustrated here: http://archive.vector.org.uk/art10500340

### random forests

#### construction
We generalise the decision tree process by growing a number of trees using the random forest algorithm. 

let m be the number of features that are sampled when selecting a splitting point, we can create a bootstrap sample  
```
.dtl.learnTree `x`y`rule`classes`m!(X;Y;rule;classes;m)
```
For illustration purposes we use the famous iris dataset by R. Fisher
```
iris:("FFFFS";enlist csv)0:`:/path/to/data/iris.csv;
iris
sepal_length sepal_width petal_length petal_width species
---------------------------------------------------------
5.1          3.5         1.4          0.2         setosa 
4.9          3           1.4          0.2         setosa 
4.7          3.2         1.3          0.2         setosa 
4.6          3.1         1.5          0.2         setosa 
5            3.6         1.4          0.2         setosa 
5.4          3.9         1.7          0.4         setosa 
4.6          3.4         1.4          0.3         setosa 
5            3.4         1.5          0.2         setosa 
4.4          2.9         1.4          0.2         setosa 
4.9          3.1         1.5          0.1         setosa
...
```
We now create the input dataset for our tree. All fields apart from last will consist our features. We classify the species (map to integer classes):

```
dataset:()!();
dataset[`x]:value flip delete species from iris;
dataset[`y]:{distinct[x]?x} iris[`species];
params: dataset,`rule`classes!(>;asc distinct dataset`y);

flip params`x
5.1 3.5 1.4 0.2
4.9 3   1.4 0.2
4.7 3.2 1.3 0.2
4.6 3.1 1.5 0.2
5   3.6 1.4 0.2
5.4 3.9 1.7 0.4
4.6 3.4 1.4 0.3
5   3.4 1.5 0.2
4.4 2.9 1.4 0.2
...

distinct params`y
0 1 2
```
visualise
```
flip @[`x`y#params;`x;flip] 
x               y
-----------------
5.1 3.5 1.4 0.2 0
4.9 3   1.4 0.2 0
4.7 3.2 1.3 0.2 0
4.6 3.1 1.5 0.2 0
5   3.6 1.4 0.2 0
5.4 3.9 1.7 0.4 0
4.6 3.4 1.4 0.3 0
5   3.4 1.5 0.2 0
4.4 2.9 1.4 0.2 0
4.9 3.1 1.5 0.1 0
...
```
We grow (learn) a tree and observe its meta . We can see that columns returned for this treetable include the params as well as intermediate steps which are useful for exploration of each node
```
\t tree:.dtl.learnTree params
16

q)meta tree
c          | t f a
-----------| -----
i          | j    
p          | j    
path       | J    
infogains  |      
xi         | j    
j          |      
infogain   | f    
bitmap     | B    
appliedrule|      
rule       |      
rulepath   |      
  
```
We can inspect our fields of choise, e.g. the full rullpath along with the associated applied rules
```
select i,p,path,xi,j,rulepath from tree
x  p  path         xi j   rulepath                                                                                                                                          
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0  0  ,0              0N  ()                                                                                                                                                
1  0  1 0          2  1.9 ,(~:;(`.dtl.runRule;>;2;1.9))                                                                                                                     
2  0  2 0          2  1.9 ,(::;(`.dtl.runRule;>;2;1.9))                                                                                                                     
3  2  3 2 0        3  1.7 ((::;(`.dtl.runRule;>;2;1.9));(~:;(`.dtl.runRule;>;3;1.7)))                                                                                       
4  3  4 3 2 0      2  4.9 ((::;(`.dtl.runRule;>;2;1.9));(~:;(`.dtl.runRule;>;3;1.7));(~:;(`.dtl.runRule;>;2;4.9)))                                                          
5  4  5 4 3 2 0    3  1.6 ((::;(`.dtl.runRule;>;2;1.9));(~:;(`.dtl.runRule;>;3;1.7));(~:;(`.dtl.runRule;>;2;4.9));(~:;(`.dtl.runRule;>;3;1.6)))                             
6  4  6 4 3 2 0    3  1.6 ((::;(`.dtl.runRule;>;2;1.9));(~:;(`.dtl.runRule;>;3;1.7));(~:;(`.dtl.runRule;>;2;4.9));(::;(`.dtl.runRule;>;3;1.6)))                             
7  3  7 3 2 0      2  4.9 ((::;(`.dtl.runRule;>;2;1.9));(~:;(`.dtl.runRule;>;3;1.7));(::;(`.dtl.runRule;>;2;4.9)))                                                          
8  7  8 7 3 2 0    3  1.5 ((::;(`.dtl.runRule;>;2;1.9));(~:;(`.dtl.runRule;>;3;1.7));(::;(`.dtl.runRule;>;2;4.9));(~:;(`.dtl.runRule;>;3;1.5)))                             
9  7  9 7 3 2 0    3  1.5 ((::;(`.dtl.runRule;>;2;1.9));(~:;(`.dtl.runRule;>;3;1.7));(::;(`.dtl.runRule;>;2;4.9));(::;(`.dtl.runRule;>;3;1.5)))                             
10 9  10 9 7 3 2 0 0  6.7 ((::;(`.dtl.runRule;>;2;1.9));(~:;(`.dtl.runRule;>;3;1.7));(::;(`.dtl.runRule;>;2;4.9));(::;(`.dtl.runRule;>;3;1.5));(~:;(`.dtl.runRule;>;0;6.7)))
11 9  11 9 7 3 2 0 0  6.7 ((::;(`.dtl.runRule;>;2;1.9));(~:;(`.dtl.runRule;>;3;1.7));(::;(`.dtl.runRule;>;2;4.9));(::;(`.dtl.runRule;>;3;1.5));(::;(`.dtl.runRule;>;0;6.7)))
...

```
To find all leaf nodes
```
.dtl.leaves tree
i  p  path         infogains                                              xi j   infogain   x                                                                               ..
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------..
1  0  1 0          `s#0 1 2 3!0.5572327 0.2679114 0.9182958 0.9182958     2  1.9 0.9182958  5.1 4.9 4.7 4.6 5   5.4 4.6 5   4.4 4.9 5.4 4.8 4.8 4.3 5.8 5.7 5.4 5.1 5.7 5.1 ..
5  4  5 4 3 2 0    `s#0 1 2 3!0.1044276 0.03781815 0.02834482 0.1460943   3  1.6 0.1460943  7   6.4 6.9 5.5 6.5 5.7 6.3 4.9 6.6 5.2 5   5.9 6   6.1 5.6 6.7 5.6 5.8 6.2 5.6 ..
6  4  6 4 3 2 0    `s#0 1 2 3!0.1044276 0.03781815 0.02834482 0.1460943   3  1.6 0.1460943  4.9                                                                             ..
8  7  8 7 3 2 0    `s#0 1 2 3!0.1091703 0.2516292 0.2516292 0.4591479     3  1.5 0.4591479  6   6.3 6.1                                                                     ..
10 9  10 9 7 3 2 0 `s#0 1 2 3!0.9182958 0.2516292 0.9182958 0.2516292     0  6.7 0.9182958  6.7 6                                                                           ..
11 9  11 9 7 3 2 0 `s#0 1 2 3!0.9182958 0.2516292 0.9182958 0.2516292     0  6.7 0.9182958  7.2                                                                             ..
14 13 14 13 12 2 0 `s#0 1 2 3!0.9182958 0.9182958 0 0                     0  5.9 0.9182958  5.9                                                                             ..
15 13 15 13 12 2 0 `s#0 1 2 3!0.9182958 0.9182958 0 0                     0  5.9 0.9182958  6.2 6                                                                           ..
16 12 16 12 2 0    `s#0 1 2 3!0.06105981 0.03811322 0.09120811 0.04314475 2  4.8 0.09120811 6.3 5.8 7.1 6.3 6.5 7.6 7.3 6.7 7.2 6.5 6.4 6.8 5.7 5.8 6.4 6.5 7.7 7.7 6.9 5.6 ..
...
```
Let's predict a value. We start by looking at the average values for each feature , groupping by the predicted variable, and then attempt to predict:
```
select avg sepal_length,avg sepal_width,avg petal_length,avg petal_width by species from iris
species   | sepal_length sepal_width petal_length petal_width
----------| -------------------------------------------------
setosa    | 5.006        3.418       1.464        0.244      
versicolor| 5.936        2.77        4.26         1.326      
virginica | 6.588        2.974       5.552        2.026    

{distinct x where y}[params`y] each exec bitmap from .dtl.predictOnTree[tree] 5.936 2.77 4.26 1.326
1

q)select i:i,path,y:{distinct x where y}[params`y]each bitmap from .dtl.predictOnTree[tree] 5.936 2.77 4.26 1.326
i path      y
-------------
0 5 4 3 2 0 1


q).dtl.predictOnTree[tree] 6.3 2.8 5 1
i p path      infogains                                          xi j   infogain  bitmap         ..
-------------------------------------------------------------------------------------------------..
8 7 8 7 3 2 0 `s#0 1 2 3!0.1091703 0.2516292 0.2516292 0.4591479 3  1.5 0.4591479 000000000000000..

```
Are all features predicted correctly?
```
/ all features predict correctly?
all {[x;y;i] 
     predicted: .dtl.predictOnTree[tree] flip[x] i ;
     y[i]=first distinct y where first predicted`bitmap
    }[dataset`x;dataset`y]each til count iris
1b
```

We then proceed to create a random forest repeating the random sampling bootstrap method and then bagging everything to one number.
```
/ create a rf with 50 trees using 
/  sampling size = count of data sample
/  sampling 2/4 features (sqrt num features) in every breakpoint search (random forest)
q)\t forest50: .dtl.randomForest params,`m`n`B!(2;150;50)
445
rf:forest50`tree
```
Note the out of bag samples are stored as a separate key and used to validate the prediction ability of the learnt tree by running the prediction on them.

Show nodes for each tree in the forest
```
q)show each {[rf;b] .dtl.leaves select from rf where B=b}[rf]each exec distinct B from rf
B i  p  path           infogains                    xi j   infogain   x                                                                                                     ..
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------..
0 3  2  3 2 1 0        `s#0 1!0.9709506 0.9709506   0  4.9 0.9709506  4.9 4.9                                                                                               ..
0 4  2  4 2 1 0        `s#0 1!0.9709506 0.9709506   0  4.9 0.9709506  5   5   5                                                                                             ..
0 6  5  6 5 1 0        `s#0 3!0.1643899 0.2460226   1  0.6 0.2460226  5.2 4.8 4.8 4.4 4.8 4.6 5.1 4.9 5   4.8 4.3 4.9 5.1 4.8 4.4 4.4 4.4 5.1 4.8 4.9 5.1 5   5.1 5.1 5.4 5 ..
0 7  5  7 5 1 0        `s#0 3!0.1643899 0.2460226   1  0.6 0.2460226  5.4 5.4                                                                                               ..
0 10 9  10 9 8 0       `s#0 2!0.1309469 0.5095157   1  1.7 0.5095157  5.5 5.5 5.8 5.7 5.7 5.8                                                                               ..
0 13 12 13 12 11 9 8 0 `s#2 3!0.3912436 0.3912436   0  4.4 0.3912436  5.5 5.8 5.5 5.5 5.5 5.5 5.5 5.5 6.3 6   6   6                                                         ..
0 14 12 14 12 11 9 8 0 `s#2 3!0.3912436 0.3912436   0  4.4 0.3912436  6.1                                                                                                   ..
0 15 11 15 11 9 8 0    `s#1 3!0.04033319 0.02667844 0  2.6 0.04033319 5.6 6.7 5.9 6   5.6 6.9 5.6 5.7 6.1 6.9 6.6 6.4 5.7 5.6 6.4 5.7 6.4 5.7 6.9 6.1 5.8 5.7 5.8 5.7 6.4 5...
0 18 17 18 17 16 8 0   `s#1 2!0.9182958 0           0  3   0.9182958  6.2 6                                                                                                 ..
0 19 17 19 17 16 8 0   `s#1 2!0.9182958 0           0  3   0.9182958  5.9                                                                                                   ..
0 20 16 20 16 8 0      `s#1 2!0.04386629 0.09528291 1  4.8 0.09528291 6.7 7.7 6.3 6.7 6.7 6.2 6.9 6.3 5.8 6.3 6.5 6.5 6.7 6.4 6.7 6.7 7.2 7.7 6.9 7.6 5.8 5.8 6.5 6.7 5.7 7...
B i p path      infogains                   xi j   infogain  x                                                                                                              ..
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------..
1 1 0 1 0       `s#2 3!0.9531972 0.9531972  0  1.7 0.9531972 5.1 5   5   5.2 4.9 4.9 5.1 4.6 4.7 5.1 5.7 4.4 4.6 4.8 5.1 4.7 5.8 4.3 5.7 5.1 5.1 4.9 4.5 4.5 5   5.2 5.1 4.8..
1 4 3 4 3 2 0   `s#2 3!0.2338271 0.3162862  1  1.6 0.3162862 5.7 5.9 6.1 6.8 5.6 6.3 5.7 6.7 5.7 6.6 6.1 5.1 6.1 6.6 6.9 6.3 6.5 5.2 5.2 5.7 5.5 5.7 6.3 6.5 5.5 5.5 6.3 5.7..
1 6 5 6 5 3 2 0 `s#0 3!0.954434 0           0  5.9 0.954434  5.9 5.9 5.9                                                                                                    ..
1 7 5 7 5 3 2 0 `s#0 3!0.954434 0           0  5.9 0.954434  6.3 6.3 6.2 6.3 6                                                                                              ..
1 8 2 8 2 0     `s#1 2!0.09669115 0.7492277 1  4.9 0.7492277 6.7 6.3 7.7 6.7 6.9 7.9 6.7 6.3 7.3 5.8 6.7 6.3 5.9 7.7 6.3 6.7 6.4 7.6 6.9 6.7 6.3 6.8 6.4 5.8 6.8 6.7 6.9 6.8..
B i  p  path            infogains                    xi j   infogain   x                                                                                                    ..
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------..
2 1  0  1 0             `s#2 3!0.9182958 0.9182958   0  1.9 0.9182958  4.8 5.4 5   4.8 5.1 5.1 5.4 4.4 5   5   5.5 4.8 5.4 4.8 5.2 5.2 5.1 5.7 4.9 5.1 4.9 4.9 5.1 5.1 4.9 4..
2 5  4  5 4 3 2 0       `s#0 3!0.04543042 0.07040292 1  1.4 0.07040292 5.6 5.8 5.5 6   5.5 5.5 5.5 6.1 6.1 5.7 6.1 6.1 5.7 5   6.1 6.1 5.7 5.6 5.7 6   6   5.6 5.7 6.1 5.6 5..
2 7  6  7 6 4 3 2 0     `s#1 3!0.5916728 0.0345107   0  2.2 0.5916728  6                                                                                                    ..
2 8  6  8 6 4 3 2 0     `s#1 3!0.5916728 0.0345107   0  2.2 0.5916728  5.4 5.4 5.6 6   5.9 5.4                                                                              ..
2 10 9  10 9 3 2 0      `s#1 3!0.5435644 0.2935644   0  3   0.5435644  5.6 5.8 5.6 5.9 5.7 5.6 5.8                                                                          ..
2 11 9  11 9 3 2 0      `s#1 3!0.5435644 0.2935644   0  3   0.5435644  5.9                                                                                                  ..
2 15 14 15 14 13 12 2 0 `s#1 2!0.1935068 0.4689956   1  5   0.4689956  6.7 6.8 7   7   6.6 6.3 6.3 7   6.9 6.8 6.7 6.6 6.7 6.9 6.6 6.9 7   6.3                              ..
2 16 14 16 14 13 12 2 0 `s#1 2!0.1935068 0.4689956   1  5   0.4689956  6.3 6.3                                                                                              ..
2 17 13 17 13 12 2 0    `s#0 3!0.213605 0.7625081    1  1.7 0.7625081  6.3 6.3 6.3 6.4 6.7 6.7 6.4 6.5 6.3 6.4 6.3 6.4 6.7 6.5 6.8 6.9 6.3 6.4 6.5 6.4 6.8 6.4 6.3 6.7 6.8  ..
2 18 12 18 12 2 0       `s#0 1!0.1468624 0.05697338  0  7   0.1468624  7.4 7.2 7.7 7.7 7.7 7.7 7.7 7.2 7.2 7.6 7.7 7.7 7.4 7.9                                              ..
...
```
#### prediction
We use the built random forest to predict the classification of a new data point. The data point will traverse every tree in the random forest.
The final classification will be based on majority vote
```
q).dtl.predictOnRF[forest50]  6.3 2.8 5 1
prediction| 2
mean_error| +(,`pred_error)!,,0.9878766

```
The mean error is produced by taking the average mean error across all out of bag samples

#### feature selection

At every iteration of each individual tree, we keep track of the information gain for the features which have been selected. 
We store this under infogains column as a list of dictionaries. 
We can compute the average information gain per feature and sort in descending order - this allows us to discard the less important features:
```
q)desc avg each  (,'/)exec infogains from 1_ rf
2| 0.4640686
3| 0.4230697
0| 0.3604106
1| 0.2797944
```

### Summary and running a model

The code is constructed from basic principles to illustrate the creation of a random forest of decision trees using a q treetable. At each stage of the algorithm one can observe all data points, the vector of split features, the split point itself, the information gain, the predictor (label) and the full rulepath of decicions.

The random forest generation is a great candidate for parallelization and therefore uses the 'peach' keyword. To take advantage of running the random forest generation on multiple cores, start q with a number of slaves (q -s 4)

## regression

* linear regression

## multinormdist

* matrix functions for multi normal distribution
* cholesky decomposition
