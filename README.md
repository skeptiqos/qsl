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
B i  p  path                infogains                              xi j      infogain  bitmap                                                      ..
---------------------------------------------------------------------------------------------------------------------------------------------------..
0 5  4  5 4 3 2 1 0         `s#2 3 7!0.5665095 0.2161579 0.1344008 0  0.02   0.5665095 000000000000010000000000000000000000000000000000000000000000..
0 8  7  8 7 6 4 3 2 1 0     `s#1 7 8!0.291692 0 0                  0  0.09   0.291692  000000000000000000000000000000000000000000000000000000000000..
0 10 9  10 9 7 6 4 3 2 1 0  `s#0 2 5!0.9709506 0.9709506 0.9709506 0  0.14   0.9709506 000000000000000000000000000000000000000000000000000000000000..
0 11 9  11 9 7 6 4 3 2 1 0  `s#0 2 5!0.9709506 0.9709506 0.9709506 0  0.14   0.9709506 000000000000000000000000000000000000000000000000000000000000..
0 12 6  12 6 4 3 2 1 0      `s#0 7 8!0.2488424 0.1014338 0         0  0.15   0.2488424 000000000000000000000000000000000000000000000000000000000000..
0 16 15 16 15 14 13 3 2 1 0 `s#0 2 5!0.1225562 0.8112781 0.3112781 1  0.025  0.8112781 000000000000000000000000000000000000000000000000000000000000..
0 17 15 17 15 14 13 3 2 1 0 `s#0 2 5!0.1225562 0.8112781 0.3112781 1  0.025  0.8112781 000000000000000000000000000000000000000000000000000000000000..
0 19 18 19 18 14 13 3 2 1 0 `s#3 8 9!0.9709506 0 0                 0  0.0235 0.9709506 000000000000000000000000000000000000000000000000000000000000..
0 20 18 20 18 14 13 3 2 1 0 `s#3 8 9!0.9709506 0 0                 0  0.0235 0.9709506 000000000000000000000000000000000000000000000000000000000000..
0 23 22 23 22 21 13 3 2 1 0 `s#2 6 9!0.8453509 0.8453509 0.2515265 0  0.045  0.8453509 000000000000000000000000000000000000000000000000000000000000..
..
B i  p  path                           infogains                              xi j      infogain  bitmap                                           ..
---------------------------------------------------------------------------------------------------------------------------------------------------..
1 6  5  6 5 4 3 2 1 0                  `s#3 5 8!0.5435644 0.5435644 0         0  0.002  0.5435644 0000000000000000000000000000000000000000000000000..
1 8  7  8 7 5 4 3 2 1 0                `s#0 5 8!0.5916728 0.1280853 0         0  0.155  0.5916728 0000000000000000000000000000000000000000000000000..
1 9  7  9 7 5 4 3 2 1 0                `s#0 5 8!0.5916728 0.1280853 0         0  0.155  0.5916728 0000000000000000000000000000000000000000000000000..
1 13 12 13 12 11 10 4 3 2 1 0          `s#3 4 5!0.4543243 0.3217155 0.1700099 0  0.018  0.4543243 0000000000000000000000000000000000000000000000000..
1 17 16 17 16 15 14 12 11 10 4 3 2 1 0 `s#0 5 9!0.8112781 0.8112781 0         0  0.16   0.8112781 0000000000000000000000000000000000000000000000000..
1 18 16 18 16 15 14 12 11 10 4 3 2 1 0 `s#0 5 9!0.8112781 0.8112781 0         0  0.16   0.8112781 0000000000000000000000000000000000000000000000000..
1 19 15 19 15 14 12 11 10 4 3 2 1 0    `s#0 1 7!0.1169268 0.0884476 0         0  0.17   0.1169268 0000000000000000000000000000000000000000000000000..
1 21 20 21 20 14 12 11 10 4 3 2 1 0    `s#0 5 7!0 1 0f                        1  0.006  1         0000000000000000000000000000000000000000000000000..
1 22 20 22 20 14 12 11 10 4 3 2 1 0    `s#0 5 7!0 1 0f                        1  0.006  1         0000000000000000000000000000000000000000000000000..
1 26 25 26 25 24 23 11 10 4 3 2 1 0    `s#3 5 6!0.9182958 0 0.9182958         0  0.0315 0.9182958 0000000000000000000000000000000000000000000000000..
..
B i  p  path                      infogains                                xi j      infogain  bitmap                                              ..
---------------------------------------------------------------------------------------------------------------------------------------------------..
2 6  5  6 5 4 3 2 1 0             `s#0 4 7!0.7219281 0.7219281 0           0  0.15   0.7219281 0000000000000000000000000000000000000000000000000000..
2 7  5  7 5 4 3 2 1 0             `s#0 4 7!0.7219281 0.7219281 0           0  0.15   0.7219281 0000000000000000000000000000000000000000000000000000..
2 10 9  10 9 8 4 3 2 1 0          `s#4 7 8!0.9940302 0 0                   0  0.0045 0.9940302 0000000000000000000000000000000000000000000000000000..
2 11 9  11 9 8 4 3 2 1 0          `s#4 7 8!0.9940302 0 0                   0  0.0045 0.9940302 0000000000000000000000000000000000000000000000000000..
2 14 13 14 13 12 8 4 3 2 1 0      `s#2 4 7!0.9182958 0 0                   0  0.035  0.9182958 0000000000000000000000000000000000000000000000000000..
2 15 13 15 13 12 8 4 3 2 1 0      `s#2 4 7!0.9182958 0 0                   0  0.035  0.9182958 0000000000000000000000000000000000000000000000000000..
2 16 12 16 12 8 4 3 2 1 0         `s#5 7 9!0.1981174 0.07600985 0.07600985 0  0.0045 0.1981174 0000000000000000000000000000000000000000000000000000..
2 21 20 21 20 19 18 17 3 2 1 0    `s#4 6 9!0 0.2516292 0.2516292           1  0.0075 0.2516292 0000000000000000000000000000000000000000000000000000..
2 23 22 23 22 20 19 18 17 3 2 1 0 `s#1 4 7!1 0 0f                          0  0.125  1         0000000000000000000000000000000000000000000000000000..
2 24 22 24 22 20 19 18 17 3 2 1 0 `s#1 4 7!1 0 0f                          0  0.125  1         0000000000000000000000000000000000000000000000000000..
..
B i  p  path                      infogains                      xi j      infogain  bitmap                                                        ..
---------------------------------------------------------------------------------------------------------------------------------------------------..
3 8  7  8 7 6 5 4 3 2 1 0         `s#0 2 8!0.9182958 0.9182958 0 0  0.155  0.9182958 00000000000000000000000000000000000000000000000000000000000000..
3 9  7  9 7 6 5 4 3 2 1 0         `s#0 2 8!0.9182958 0.9182958 0 0  0.155  0.9182958 00000000000000000000000000000000000000000000000000000000000000..
3 11 10 11 10 6 5 4 3 2 1 0       `s#3 6 8!0.5567796 0.4689956 0 0  0.023  0.5567796 00000000000000000000000000000000000000000000000000000000000000..
3 13 12 13 12 10 6 5 4 3 2 1 0    `s#1 5 7!0.8631206 0.5916728 0 0  0.14   0.8631206 00000000000000000000000000000000000000000000000000000000000000..
3 15 14 15 14 12 10 6 5 4 3 2 1 0 `s#1 6 7!1 1 0f                0  0.145  1         00000000000000000000000000000000000000000000000000000000000000..
3 16 14 16 14 12 10 6 5 4 3 2 1 0 `s#1 6 7!1 1 0f                0  0.145  1         00000000000000000000000000000000000000000000000000000000000000..
3 18 17 18 17 5 4 3 2 1 0         `s#4 5 8!1 1 0f                0  0.017  1         00000000000000000000000000000000000000000000000000000000000000..
3 19 17 19 17 5 4 3 2 1 0         `s#4 5 8!1 1 0f                0  0.017  1         00000000000000000000000000000000000000000000000000000000000000..
3 22 21 22 21 20 4 3 2 1 0        `s#4 8 9!0.8112781 0 0         0  0.0315 0.8112781 00000000000000000000000000000000000000000000000000001000000000..
3 23 21 23 21 20 4 3 2 1 0        `s#4 8 9!0.8112781 0 0         0  0.0315 0.8112781 00000000000000000000000000000000000000000000000000000000000000..
..

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

## shape

* Non parametrics methods for identifying unusual shapes in timeseries (aka collective outliers), using Discord/Motif
* These look at sub series of pre-specified window length ``m``

### Discord

The idea consists of detecting the most unusual subsequence in a time series (denominated time series discord) [Keogh et al. 2005; Lin et al. 2005], by comparing each subsequence with the others; that is, `D` is a discord of time series `X` if
>         ∀S ∈ A, min (d(D,D′)) > min (d(S,S′)),  D′∈A,D∩D′=∅ S′∈A,S∩S′=∅
where `A` is the set of all subsequences of X extracted by a sliding window, `D ′` is a subsequence in `A` that does not overlap with `D` (non-overlapping subsequences), `S ′` in `A` does not overlap with `S` (non-overlapping subsequences), and `d` is the Euclidean distance between two subsequences of equal length

See: https://arxiv.org/pdf/2002.04236.pdf.

See also: Discord and Motif (See Neighbor Profile: Bagging Nearest Neighbors for Unsupervised Time Series Mining)
https://www.researchgate.net/profile/Yuanduo-He/publication/340663191_Neighbor_Profile_Bagging_Nearest_Neighbors_for_Unsupervised_Time_Series_Mining/links/5e97d607a6fdcca7891c2a0b/Neighbor-Profile-Bagging-Nearest-Neighbors-for-Unsupervised-Time-Series-Mining.pdf

Example using cosine and introducing anomalies of various sizes:
```
pi:acos -1;
x:cos til[1000]%10*pi;
x1:@[x;694 695 696;:;-1.1 -1.25 -1.2];   / add artificial anomaly dip
x2:@[x1;594 595 596;:;0.8 0.5 0.8];      / add a second, larger dip
x3:@[x1;594 595 596;:;0.9 0.85 0.9];     / make the second dip smaller than the first
/ visualise
select i,x from ([]x:x3)

D1:.ushape.discordMotif[x1;50;0b];        / detect the starting index of the subseries at the dip
D2:.ushape.discordMotif[x2;50;0b];        / detect the larger dip
D3:.ushape.discordMotif[x3;50;0b];        / detect the larger dip
```
## ht - hypothesis testing

Non-parametrics hypothesis testing.

### KS-Test

The Kolmogorov-Smirnov Test is a non-parametric test. The null hypothesis is that both groups were sampled from populations with identical distributions. It tests for any violation of that null hypothesis -- different medians, different variances, or different distributions.
The 1-sample tests for significant difference from normality. The 2-sample tests the two sets for significantly different CDFs. It has the power to detect changes in the shape of the distributions (Lehmann, page 39). It is not tailed since it just generally checks the difference bertween 2 distributions.

```
\S 42
q)s1:250?100f;s2:55?95f
.ht.KSTest[s1;s2;`twoside;1%100]
KSD     | 0.1229091
KSThresh| 0.2424111

q)s1:250?100f;s2:55?50f
q).ht.KSTest[s1;s2;`less;1%1000]
KSD     | 0.544
KSThresh| 0.2767911

/ check for normality
n:-1.21892513 -1.02139816  0.79473819 -0.21667778  1.48445971  0.79481571 -1.28311514  1.61096972  2.06241178  0.91405552  0.66413083  2.63863206   0.79233555 -1.32763874 -0.76997982 -1.76017473 -0.60255683 -0.14036253  1.43857703 -0.06168719 -0.79564272 -1.21610969 -1.24676392  0.93900982 -0.95271042
q).ht.KSTest[n;();`;0.1]
KSD     | 0.1793501
KSThresh| 0.3461637

```

### Neural Networks

We implement a Multilayer Perceptron (MLP). This is a simple Neural Network (NN) and we will make the number of hidden layers `l` configurable.

#### Background

A NN is a linear model `z=b+Wa`, where `b` is a bias vector, `W` a weights matrix and `a` is the input/activation.

Importantly, a NN includes and activation funciton `f[z]` to allow for non-linear relationships in the data.

For building the NN we require two steps:
1. A feed-forward mechanism to implement the actual network prediction
2. Backpropagation, to calculate the Gradient and estimate the network Weights

#### How does a network learn? 

By minimising the cost function, which is the prediction error.

Example Cost functions: MSE (mean square error) for regression or cross-entropy/AUROC for classification

The Gradient of the cost function gives us which changes to the weights matter most to minimise the error.
So if `C(w)` and the Gradient `dCdw (dC/dw)` is negative we shift the input `w` to the right (towards the local minimum).
 `w` represents the vector of weights and biases across all layers.
Backpropagation lets us compute that negative gradient.

How to adjust/emphasize a node's prediction to nudge it in the right direction?
  - increase bias
  - increase weight ( in proportion to previous activation, node value n)
  - change the node value (in proportion to their weights)

We end up with calculating delta changes for the weights matrix w at each layer. Each of these partial derivatives/changes are the components of the Gradient vector.

The new value of the weight will then be 

`w = w -r dCdw` , `r` is the learning rate < 1, `dCdw` is the derivative of `C(w)`

ie compute the gradient and then take a small step in the negative direction, to reach the local minimum

So if C is cost function, n the value of the node in previous layer and y the expected output (correct label)
```
 z= b+w$a
 a=f[z] / f is relu,sigmoid or other activation function
```
We want to calculate the rate of cost change when changing the weight:

```
 dCdw(l): dCda * dadz * dzdw (chain rule)
 -> del of Cost function wrt activation a at layer l * del of activation function wrt to linear output z at layer l * d of z wrt w
```
 Let delta be the approximate error wrt to activation at layer l
```
 delta(L): dCdz: dCda * dadz    / L is output layer
 dCda: 2(a-y) for c=(a-y)xexp 2 / y is output
 delta(l): dCdz: (T(W(l+1)) * delta(l+1)) * dadz / T(W) is weights transpose of the next layer. ie apply the transposte of weights to the error d(l) to get the error at d(l-1)
 dadz: derivative of activation func, da
 dzdw: a[l-1], a of previous layer
 ```
 Therefore: ```dCdw = a[l-1] mmu delta[l]```

reference: http://neuralnetworksanddeeplearning.com/chap2.html?utm_source=perplexity

#### Minibatch Stochastic Gradient Descent

Processes batches of data obtained by a random permutation of the training data. Each observation is processed only once per epoch, albeit in random order. An epoch is one complete pass of the training dataset.

This enables us to average the gradients computed for each batch, which is less noisy than SGD - which applies on every sample - but also faster and less expensive than a Batch Gradient Descent which uses the whole training sample.

#### Training on the MNIST dataset

Let's now train our model on the famous MNIST dataset of handwritten digits. We will use one hidden layer of size 32, along with the mini-batch SGD. We will only use a single epoch:

```
S:readMNIST[`train;0];        / sample train data
Y:@[10#0;;:;1]each Y:first S; / 60k records
X:flip 1_ S;X:X%max over X;   / each of the 60k X is 28x28=784 pixels
-1 "Train on ",string[count X]," MNIST images";
-1 "Model Params:";
show `k`l`eta`batchsize`numepochs`history#pm0:([x:X; y:Y; k:32; l:1; e:0f; eta:1%3; hactivf:relu; hactivf_d: >[;0]; activf:softmax; cost:xentropy; cost_d:xentropy_d; batchsize:64; numepochs:1;history:0b]);
pm:.neuralnet.init[pm0];
-1 ".neuralnet.train:";
\t nn:.neuralnet.train[pm]

T:readMNIST[`test;0];         / test data
Y:@[10#0;;:;1]each Y:first T; / 10k records
X:flip 1_ T;X:X%max over X;
-1 ".neuralnet.validate prediction on ",string[count X]," test MNIST images";
\t predict:.neuralnet.validate[([x:X; y:Y; hactivf:relu; activf:softmax; nn: nn; history:pm`history])]

-1"prediction accuracy:",string {100*sum[x]%count x}predict;

```

The training gives the following results:

```
Train on 60000 MNIST images
Model Params:
k        | 32
l        | 1
eta      | 0.3333333
batchsize| 64
numepochs| 1
history  | 0b
.neuralnet.train:
1517
.neuralnet.validate prediction on 10000 test MNIST images
74
prediction accuracy:93.84
```
So we achieve ~94% accuracy with a training time of 1.5 seconds.

Let's try now and increase the number of epochs to 5

```
 Train on 60000 MNIST images
Model Params:
k        | 32
l        | 1
eta      | 0.3333333
batchsize| 64
numepochs| 5
history  | 0b
.neuralnet.train:
7654
.neuralnet.validate prediction on 10000 test MNIST images
74
prediction accuracy:95.86
```

The prediction accuracy improved to almost 96%, but at a cost of taking almost 8 seconds to train the network.
