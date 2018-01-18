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
.dtl.learnTree `x`y`rule`classes(X;Y;rule;classes)
```

### random forests

We generalise the decision tree process by growing a number of tress using the random forest algorithm. 

let m be the number of features that are sampled when selecting a splitting point, we can create a bootstrap sample  
```
.dtl.learnTree `x`y`rule`classes`m(X;Y;rule;classes;m)
```


## regression
linear regression

## multinormdist
matrix functions for multi normal distribution
