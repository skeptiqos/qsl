# qsl
q statistical learning

## dtl
decision tree learning

### grow a Tree

let
 
* X and Y be the predictor and predicted variables respectively
* rule be a function or logical operator applied to a set of ordinal numbers
* classes be the classification of the predicted variable

Then to learn a tree 
```
.dtl.learnTree `x`y`rule`classes!(X;Y;rule;classes)
```
A tree is grown by maximising information entropy on each split and results into a treetable, as illustrated here: http://archive.vector.org.uk/art10500340

### random forests

#### construction
We generalise the decision tree process by growing a number of trees using the random forest algorithm. 

let m be the number of features that are sampled when selecting a splitting point, we can create a bootstrap sample  
```
.dtl.learnTree `x`y`rule`classes`m!(X;Y;rule;classes;m)
```
For illustration purposes, assume we have the following dataset with 3 features and a classified predicted variable y
```
n:10;
x:flip (n?100f;n?1000f;-25+n?50;n?5.5 257.7 3.3 -4);
y:-50f+n?100f;
s:`x`y!(x;y);
s:@[s;`y;.dtl.classify -50 -25 0 25 50f];
params:s,`rule`classes!(>;asc distinct s`y);

params`x
27.82122 230.6385 -5  5.5  
23.92341 949.975  4   5.5  
15.08133 439.081  13  3.3  
15.67317 575.9051 8   257.7
97.85    591.9004 -1  3.3  
70.43314 848.1567 12  257.7
94.41671 389.056  -21 3.3  
78.33686 391.543  20  -4f  
40.99561 81.23546 16  3.3  
61.08817 936.7503 -23 257.7

params`y
2 2 2 1 3 1 1 3 4 2
```
One run of the bootsrap algorithm with m=2 will give us 
```
params:s,`rule`classes`m!(>;asc distinct s`y;2);
boottree:.dtl.bootstrapTree[params;til count flip s`x;count x;0];

q)delete x from boottree`tree
B i  p  path        xi j        infogain  y                   appliedrule rule rulepath                                                                                     ..
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------..
0 0  0  ,0             0N                 1 1 3 2 1 1 3 4 2 2 ::          >    ()                                                                                           ..
0 1  0  1 0         0  23.92341 0.6       2 1 1 2 2           ~:          >    ,(~:;(`.dtl.runRule;>;0;23.92341))                                                           ..
0 2  1  2 1 0       0  15.67317 0.4199731 1 1 2               ~:          >    ((~:;(`.dtl.runRule;>;0;23.92341));(~:;(`.dtl.runRule;>;0;15.67317)))                        ..
0 3  2  3 2 1 0     0  15.08133 0.9182958 ,2                  ~:          >    ((~:;(`.dtl.runRule;>;0;23.92341));(~:;(`.dtl.runRule;>;0;15.67317));(~:;(`.dtl.runRule;>;0;1..
0 4  2  4 2 1 0     0  15.08133 0.9182958 1 1                 ::          >    ((~:;(`.dtl.runRule;>;0;23.92341));(~:;(`.dtl.runRule;>;0;15.67317));(::;(`.dtl.runRule;>;0;1..
0 5  1  5 1 0       0  15.67317 0.4199731 2 2                 ::          >    ((~:;(`.dtl.runRule;>;0;23.92341));(::;(`.dtl.runRule;>;0;15.67317)))                        ..
0 6  0  6 0         0  23.92341 0.6       1 1 3 3 4           ::          >    ,(::;(`.dtl.runRule;>;0;23.92341))                                                           ..
0 7  6  7 6 0       0  40.99561 0.7219281 ,4                  ~:          >    ((::;(`.dtl.runRule;>;0;23.92341));(~:;(`.dtl.runRule;>;0;40.99561)))                        ..
0 8  6  8 6 0       0  40.99561 0.7219281 1 1 3 3             ::          >    ((::;(`.dtl.runRule;>;0;23.92341));(::;(`.dtl.runRule;>;0;40.99561)))                        ..
0 9  8  9 8 6 0     0  -21      0.3112781 ,1                  ~:          >    ((::;(`.dtl.runRule;>;0;23.92341));(::;(`.dtl.runRule;>;0;40.99561));(~:;(`.dtl.runRule;>;0;-..
0 10 8  10 8 6 0    0  -21      0.3112781 1 3 3               ::          >    ((::;(`.dtl.runRule;>;0;23.92341));(::;(`.dtl.runRule;>;0;40.99561));(::;(`.dtl.runRule;>;0;-..
0 11 10 11 10 8 6 0 0  70.43314 0.9182958 ,1                  ~:          >    ((::;(`.dtl.runRule;>;0;23.92341));(::;(`.dtl.runRule;>;0;40.99561));(::;(`.dtl.runRule;>;0;-..
0 12 10 12 10 8 6 0 0  70.43314 0.9182958 3 3                 ::          >    ((::;(`.dtl.runRule;>;0;23.92341));(::;(`.dtl.runRule;>;0;40.99561));(::;(`.dtl.runRule;>;0;-..
```

To view all end nodes, aka leaves:
```
q).dtl.leaves delete x from boottree`tree
B i  p  path        xi j        infogain  y   appliedrule rule rulepath                                                                                                     ..
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------..
0 3  2  3 2 1 0     0  15.08133 0.9182958 ,2  ~:          >    ((~:;(`.dtl.runRule;>;0;23.92341));(~:;(`.dtl.runRule;>;0;15.67317));(~:;(`.dtl.runRule;>;0;15.08133)))      ..
0 4  2  4 2 1 0     0  15.08133 0.9182958 1 1 ::          >    ((~:;(`.dtl.runRule;>;0;23.92341));(~:;(`.dtl.runRule;>;0;15.67317));(::;(`.dtl.runRule;>;0;15.08133)))      ..
0 5  1  5 1 0       0  15.67317 0.4199731 2 2 ::          >    ((~:;(`.dtl.runRule;>;0;23.92341));(::;(`.dtl.runRule;>;0;15.67317)))                                        ..
0 7  6  7 6 0       0  40.99561 0.7219281 ,4  ~:          >    ((::;(`.dtl.runRule;>;0;23.92341));(~:;(`.dtl.runRule;>;0;40.99561)))                                        ..
0 9  8  9 8 6 0     0  -21      0.3112781 ,1  ~:          >    ((::;(`.dtl.runRule;>;0;23.92341));(::;(`.dtl.runRule;>;0;40.99561));(~:;(`.dtl.runRule;>;0;-21)))           ..
0 11 10 11 10 8 6 0 0  70.43314 0.9182958 ,1  ~:          >    ((::;(`.dtl.runRule;>;0;23.92341));(::;(`.dtl.runRule;>;0;40.99561));(::;(`.dtl.runRule;>;0;-21));(~:;(`.dtl...
0 12 10 12 10 8 6 0 0  70.43314 0.9182958 3 3 ::          >    ((::;(`.dtl.runRule;>;0;23.92341));(::;(`.dtl.runRule;>;0;40.99561));(::;(`.dtl.runRule;>;0;-21));(::;(`.dtl...
```
We then proceed to create a random forest repeating the random sampling bootstrap method and then bagging everything to one number.
```
q)ensemble:.dtl.randomForest params,`p`n`B!(til count flip params`x;count params`x;5)
q)ensemble
tree| +`B`i`p`path`xi`j`infogain`x`y`appliedrule`rule`rulepath`classes`m`oob!(0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 2 2 2 2 2 2 2 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4;0 1 2 3 4 5 6 7 ..
oob | +`B`i`p`path`xi`j`infogain`x`y`appliedrule`rule`rulepath`classes`m`oob`obs_y`pred_error!(0 0 0 1 1 1 1 1 2 2 2 3 3 3 3 4 4 4 4;3 3 3 2 2 2 2 3 3 3 5 5 6 6 5 1 1 1 1;2..
q)
q)ensemble`tree
B i p path      xi j        infogain  x                                                                                                                                     ..
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------..
0 0 0 ,0           0N                 ((40.99561;81.23546;16;3.3);(94.41671;389.056;-21;3.3);(61.08817;936.7503;-23;257.7);(27.82122;230.6385;-5;5.5);(27.82122;230.6385;-5;..
0 1 0 1 0       1  81.23546 0.4689956 ,(40.99561;81.23546;16;3.3)                                                                                                           ..
0 2 0 2 0       1  81.23546 0.4689956 ((94.41671;389.056;-21;3.3);(61.08817;936.7503;-23;257.7);(27.82122;230.6385;-5;5.5);(27.82122;230.6385;-5;5.5);(27.82122;230.6385;-5;..
0 3 2 3 2 0     0  230.6385 0.2516292 ((27.82122;230.6385;-5;5.5);(27.82122;230.6385;-5;5.5);(27.82122;230.6385;-5;5.5))                                                    ..
0 4 2 4 2 0     0  230.6385 0.2516292 ((94.41671;389.056;-21;3.3);(61.08817;936.7503;-23;257.7);(61.08817;936.7503;-23;257.7);(70.43314;848.1567;12;257.7);(15.08133;439.081..
0 5 4 5 4 2 0   0  848.1567 0.4591479 ((94.41671;389.056;-21;3.3);(70.43314;848.1567;12;257.7);(15.08133;439.081;13;3.3);(15.67317;575.9051;8;257.7))                       ..
0 6 5 6 5 4 2 0 1  12       0.8112781 ((94.41671;389.056;-21;3.3);(70.43314;848.1567;12;257.7);(15.67317;575.9051;8;257.7))                                                 ..
0 7 5 7 5 4 2 0 1  12       0.8112781 ,(15.08133;439.081;13;3.3)                                                                                                            ..
0 8 4 8 4 2 0   0  848.1567 0.4591479 ((61.08817;936.7503;-23;257.7);(61.08817;936.7503;-23;257.7))                                                                         ..
1 0 0 ,0           0N                 ((78.33686;391.543;20;-4f);(94.41671;389.056;-21;3.3);(94.41671;389.056;-21;3.3);(97.85;591.9004;-1;3.3);(78.33686;391.543;20;-4f);(40..
1 1 0 1 0       0  389.056  0.9709506 ((94.41671;389.056;-21;3.3);(94.41671;389.056;-21;3.3);(40.99561;81.23546;16;3.3);(94.41671;389.056;-21;3.3))                         ..
1 2 1 2 1 0     0  40.99561 0.8112781 ,(40.99561;81.23546;16;3.3)                                                                                                           ..
1 3 1 3 1 0     0  40.99561 0.8112781 ((94.41671;389.056;-21;3.3);(94.41671;389.056;-21;3.3);(94.41671;389.056;-21;3.3))                                                    ..
1 4 0 4 0       0  389.056  0.9709506 ((78.33686;391.543;20;-4f);(97.85;591.9004;-1;3.3);(78.33686;391.543;20;-4f);(97.85;591.9004;-1;3.3);(97.85;591.9004;-1;3.3);(61.08817..
```
Note the out of bag samples are stored as a separate key and used to validate the prediction ability of the learnt tree by running the prediction on them.

#### prediction
We the proceed to use the built random forest to predict the classification of a new data point. The data point will traverse every tree in the random forest.
The final classification will be based on majority vote
```
q).dtl.predictOnRF[ensemble;(5.5 0 -2 10f)]
prediction| 4
mean_error| +(,`pred_error)!,,1.473684

```
The mean error is produced by taking the average mean error across all out of bag samples

### Summary and running a model

The code is constructed from basic principles to illustrate the creation of a random forest of decision trees using a q treetable. At each stage of the algorithm one can observe all data points, the vector of split features, the split point itself, the information gain, the predictor (label) and the full rulepath of decicions.

The random forest generation is a great candidate for parallelization and therefore uses the 'peach' keyword. To take advantage of running the random forest generation on multiple cores, start q with a number of slaves (q -s 4)

## regression
linear regression

## multinormdist
matrix functions for multi normal distribution
