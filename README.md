# qsl
q statistical learning 

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
x          |      
y          | J    
appliedrule|      
rule       |      
rulepath   |      
classes    | J    
m          | j    
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

exec distinct each y from .dtl.predictOnTree[tree] 5.936 2.77 4.26 1.326
1

`y xcols .dtl.predictOnTree[tree] 6.3 2.8 5 1
y     i p path      infogains                                          xi j   infogain  x                                               appliedrule rule rulepath classes m
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
2 2 2 8 7 8 7 3 2 0 `s#0 1 2 3!0.1091703 0.2516292 0.2516292 0.4591479 3  1.5 0.4591479 6   6.3 6.1 2.2 2.8 2.6 5   5.1 5.6 1.5 1.5 1.4 ~:          >             0 1 2   4
```
Are all features predicted correctly?
```
/ all features predict correctly?
all {[x;y;i] predicted: .dtl.predictOnTree[tree] flip[x] i ; y[i]=first distinct first predicted`y}[dataset`x;dataset`y]each til count iris
1b
```

We then proceed to create a random forest repeating the random sampling bootstrap method and then bagging everything to one number.
```
/ create a rf with 50 trees using sampling size = data size and 3/4 features in every iteration
\ts forest50: .dtl.randomForest params,`p`n`B!( til 3;count iris;50)
```
Note the out of bag samples are stored as a separate key and used to validate the prediction ability of the learnt tree by running the prediction on them.

Show nodes for each tree in the forest
```
q)show each {[rf;b] .dtl.leaves select from rf where B=b}[rf]each exec distinct B from rf
B i  p  path            infogains                                            xi j   infogain  x                                                                             ..
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------..
0 1  0  1 0             `s#0 1 2 3!0.5256439 0.351191 0.9370098 0.9370098    2  1.9 0.9370098 5.5 5.2 5.5 4.8 4.8 4.4 5.8 4.8 4.6 5.1 5.7 4.9 5   4.8 4.3 4.9 5.1 4.8 4.4 4...
0 4  3  4 3 2 0         `s#0 1 2 3!0.02998238 0.03341651 0.139233 0.02550547 2  5.1 0.139233  5.6 5.9 6   5.6 6.9 5.6 5.7 5   6.1 5.5 5   6.9 5.4 6.6 6.4 5.7 5.8 5.5 5.5 5...
0 5  3  5 3 2 0         `s#0 1 2 3!0.02998238 0.03341651 0.139233 0.02550547 2  5.1 0.139233  6.1                                                                           ..
0 8  7  8 7 6 2 0       `s#0 1 2 3!0.2810361 0.4581059 0.09288851 0.09288851 1  2.8 0.4581059 6.3 6.2 4.9 5.7 6.3 4.9                                                       ..
0 10 9  10 9 7 6 2 0    `s#0 1 2 3!0.2516292 0.2516292 0.2516292 0.2516292   0  5.9 0.2516292 5.9                                                                           ..
0 12 11 12 11 9 7 6 2 0 `s#0 1 2 3!1 0 1 1f                                  0  6   1         6                                                                             ..
0 13 11 13 11 9 7 6 2 0 `s#0 1 2 3!1 0 1 1f                                  0  6   1         6.7                                                                           ..
0 14 6  14 6 2 0        `s#0 1 2 3!0.01845374 0.036881 0.1085004 0.08297587  2  5   0.1085004 6.7 7.7 6.7 6.7 6.2 6.9 6.3 5.8 6.3 6.5 6.5 6.7 6.4 6.7 6.7 7.2 7.7 6.9 7.6 5...
B i p path      infogains                                          xi j   infogain  x                                                                                       ..
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------..
1 1 0 1 0       `s#0 1 2 3!0.6527083 0.3455988 0.9531972 0.9531972 2  1.7 0.9531972 5.1 5   5   5.2 4.9 4.9 5.1 4.6 4.7 5.1 5.7 4.4 4.6 4.8 5.1 4.7 5.8 4.3 5.7 5.1 5.1 4.9 ..
1 3 2 3 2 0     `s#0 1 2 3!0.2098285 0.1063728 0.7486074 0.8245606 3  1.6 0.8245606 6.1 6.8 5.6 6.3 5.7 6.7 5.7 6.6 6.1 5.1 6.1 6.6 6.9 6.3 6.5 5.2 5.2 5.7 5.5 5.7 6.3 6.5 ..
1 6 5 6 5 4 2 0 `s#0 1 2 3!0.9709506 0.9709506 0 0                 0  5.9 0.9709506 5.9 5.9 5.9                                                                             ..
1 7 5 7 5 4 2 0 `s#0 1 2 3!0.9709506 0.9709506 0 0                 0  5.9 0.9709506 6.2 6                                                                                   ..
1 8 4 8 4 2 0   `s#0 1 2 3!0.1730418 0.131868 0.2275657 0.09333728 2  4.8 0.2275657 7.7 6.3 6.7 6.9 6.3 7.9 6.7 6.3 7.3 5.8 6.7 6.3 5.9 7.7 6.3 6.7 6.4 7.6 6.9 6.7 6.3 6.8 ..
B i  p path      infogains                                              xi j   infogain   x                                                                                 ..
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------..
2 1  0 1 0       `s#0 1 2 3!0.6277119 0.2542153 0.9114844 0.9114844     2  1.9 0.9114844  4.8 5.1 5.1 5.4 4.4 5   5   5.5 4.8 5.4 4.8 5.2 5.2 5.1 5.7 4.9 5.1 4.9 4.9 5.1 5...
2 4  3 4 3 2 0   `s#0 1 2 3!0.07177718 0.03739419 0.1292362 0.2351934   3  1.8 0.2351934  5.4 5.9 5.5 6   5.5 5.5 5.4 7   7   6.6 5.6 6.3 6.3 5.5 6.1 7   6.1 5.7 6.9 6.1 6...
2 5  3 5 3 2 0   `s#0 1 2 3!0.07177718 0.03739419 0.1292362 0.2351934   3  1.8 0.2351934  5.6 5.6                                                                           ..
2 8  7 8 7 6 2 0 `s#0 1 2 3!0.6500224 0.6500224 0 0.3166891             0  6.3 0.6500224  6.3 5.7 6   6.3 6.3                                                               ..
2 9  7 9 7 6 2 0 `s#0 1 2 3!0.6500224 0.6500224 0 0.3166891             0  6.3 0.6500224  6.7                                                                               ..
2 10 6 10 6 2 0  `s#0 1 2 3!0.02133484 0.02010771 0.06413159 0.06413159 2  5   0.06413159 6.3 5.8 6.3 6.4 6.7 6.7 6.4 6.5 6.3 6.4 7.4 7.2 6.4 6.7 7.7 5.9 7.7 7.7 7.7 7.7 7...
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

### Summary and running a model

The code is constructed from basic principles to illustrate the creation of a random forest of decision trees using a q treetable. At each stage of the algorithm one can observe all data points, the vector of split features, the split point itself, the information gain, the predictor (label) and the full rulepath of decicions.

The random forest generation is a great candidate for parallelization and therefore uses the 'peach' keyword. To take advantage of running the random forest generation on multiple cores, start q with a number of slaves (q -s 4)

## regression

* linear regression

## multinormdist

* matrix functions for multi normal distribution
* cholesky decomposition
