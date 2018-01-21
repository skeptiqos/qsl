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

### random forests

We generalise the decision tree process by growing a number of tress using the random forest algorithm. 

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
delete x from boottree`tree
i p path  xi j        infogain  y                   appliedrule rule rulepath                                                             classes m oob
-------------------------------------------------------------------------------------------------------------------------------------------------------
0 0 ,0       0N                 3 1 1 3 3 4 3 3 1 2 ::          >    ()                                                                   1 2 3 4 2 () 
1 0 1 0   0  389.056  0.9709506 1 1 4 1             ~:          >    ,(~:;(`.dtl.runRule;>;0;389.056))                                    1 2 3 4 2 0  
2 1 2 1 0 0  40.99561 0.8112781 ,4                  ~:          >    ((~:;(`.dtl.runRule;>;0;389.056));(~:;(`.dtl.runRule;>;0;40.99561))) 1 2 3 4 2 1  
3 1 3 1 0 0  40.99561 0.8112781 1 1 1               ::          >    ((~:;(`.dtl.runRule;>;0;389.056));(::;(`.dtl.runRule;>;0;40.99561))) 1 2 3 4 2 3  
4 0 4 0   0  389.056  0.9709506 3 3 3 3 3 2         ::          >    ,(::;(`.dtl.runRule;>;0;389.056))                                    1 2 3 4 2 2  
5 4 5 4 0 0  61.08817 0.6500224 ,2                  ~:          >    ((::;(`.dtl.runRule;>;0;389.056));(~:;(`.dtl.runRule;>;0;61.08817))) 1 2 3 4 2 2  
6 4 6 4 0 0  61.08817 0.6500224 3 3 3 3 3           ::          >    ((::;(`.dtl.runRule;>;0;389.056));(::;(`.dtl.runRule;>;0;61.08817))) 1 2 3 4 2 3  
```
Note the out of bag samples are used to validate the prediction ability of the learnt tree by running the prediction on them:
```
delete x from boottre`ooberror

```

## regression
linear regression

## multinormdist
matrix functions for multi normal distribution
