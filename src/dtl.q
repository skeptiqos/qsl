/
 * Created by aris on 12/23/17.
 Decision tree learning and random forests
 tree construction based on http://archive.vector.org.uk/art10500340
\

/
 SampleTree: used for bootstrapping, samples a subset of the dataset to grow a tree from
 @param
  s: dataset `x`y!(predictors;predicted)
  p: index vector of features to sample (subset of 0...p-1)
  n: sample size
 @return a dictionary of
          `x`y : subsets of predictors and corresponding predicted values
          `oobi: indices of data sample n which were not included (out of bag sample)
 @example
 .dtl.sampleTree[`x`y!(x;y);til count x;count y]
 .dtl.sampleTree[s;til count s`x;count y]
\
.dtl.sampleTree:{[s;p;n]
 z:`x`y!(s[`x][p;i];s[`y]i:n?n);
 z,enlist[`oobi]!enlist (til n) except distinct i}

/
 Classify Y observation (predicted). use for classification tree
 @param
  breakpoints: sorted list to bucket (classify) predicted variable sample
  y          : vector of sampled predicted variable
 @return
  vector of classified predicted variable
 @example:
  .dtl.classify[-50 0 50f;y]
  1 1 1 0 0 0
\
.dtl.classify:{[breakpoints;y] asc[breakpoints] binr y}

/
 Entropy Gain for given classification
 @param
  y       : vector of sampled predicted variable
  classes : domain we wish to classify y, the distinct classes
 @return
  entropy as a float atom
 @example
  .dtl.entropy[.dtl.classify[-50 0 50f;y];0 1]
\
.dtl.entropy:{[y;classes]
 p:{sum[x=y]%count x}[y]each classes;
 neg p wsum 2 xlog p
 }

/
 Information gain for a given classification (split)
 https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
 @param
  yp     : the parent node classification set of attributes
  ysplit : the child nodes classifications set of attributes after the split
  classes: the set of classes
 @return
  information gain as a float atom
\
.dtl.infogain:{[yp;ysplit;classes].dtl.entropy[yp;classes] - (wsum). flip ({count[y]%count x}[yp];.dtl.entropy[;classes])@\:/:ysplit}

/
 Apply rule on a split of x at j to categorise classes
 @param
  y    : vector of sampled predicted variable
  x    : matrix of predictor variables
  xi   : the ith predicted variable
  rule : rule to split on
  j    : split xi at point j based on rule
\
.dtl.applyRule:{[y;x;xi;rule;j]
 (`x`y`appliedrule!(x[::;i];y i:where not b; not);
  `x`y`appliedrule!(x[::;i];y i:where b:eval (rule;xi;j); (::)))}

.dtl.runRule:{[rule;i;j;x] rule[x i;j]}

.dtl.chooseSplitXi:{[y;x;rule;classes;xi]
  j:asc distinct xi;
  info:{[y;x;xi;rule;classes;j]
   split:.dtl.applyRule[y;x;xi;rule;j];
   (.dtl.infogain[y;split`y;classes];split)
   }[y;x;xi;rule;classes] each j;
   (j;info)@\:first idesc flip[info]0
 }

/
 Choose optimal split for a node
 find the split which maximizes information gain by iterating over all splits
 @param  x:       predictor
         y:       predicted
         rule:    the logical rule to apply
         classes: the k classification classes for y
 @return xi:       the index of the predictor to split on
         j:        the position of the rule split ( x[i]>j ) ( x[i]<=j )
         infogain: how much information is gained by splitting at x[i]>j
         x,y:      the two child nodes with the corresponding splitted x and y
         rulepath: the full path of rules applied to node
 @example .dtl.chooseSplit[x;.dtl.classify[-50 0 50f;y];>;0 1]
\
.dtl.chooseSplit:{[treeinput]
 x: treeinput`x;y: treeinput`y;rule: treeinput`rule;classes: treeinput`classes;rulepath: treeinput`rulepath;m: treeinput`m;
 cx:count x;
 info: .dtl.chooseSplitXi[y;x;rule;classes]each x@sampled:asc neg[m]?cx;
 summary: (`xi`j`infogain!i,(-1_r)),/:last r:raze info i:first idesc info[;1;0];
 cnt: count summary;
 res: update rule:rule,rulepath:{[rp;ar;r;i;j] rp,enlist (ar;(`.dtl.runRule;r;i;j))}[rulepath]'[appliedrule;rule;xi;j],classes:cnt#enlist classes from summary;
 update m from res
 }

.dtl.growTree:{[r]
 {if[1>=count distinct x`y;:x];
  enlist[x],$[98h<>type rr:.dtl.growTree[x];raze @[rr;where 99h=type each rr;enlist];rr]
 }each  r:.dtl.chooseSplit[r]}

/
 Learn a tree: for each of the records in the initial split, we iterate until we reach pure nodes (leaves)
 when we reach a leaf we return flattened result and then recurse over the next split record until there are none left
 the flattened tree should contain all the paths and a tree like structure with all indices i and parents p
 @param
  dictionary with  keys
  `xi       : index of x predictor. initialise i of predictor xi as null, we will iterate over all of them
  `j        : initialise j split as null, we will iterate overa all of them
  `infogain : initialise info gain to null, this will be populated with the information gain at each split
  `x`y`rule`classes: these are input params with `x`y denoting initial sampled z set
 @return
  a table tree structure
\
.dtl.learnTree:{[params]
 rfparams:enlist[`m]!enlist count params`x;
 r0:`xi`j`infogain`x`y`appliedrule`rule`rulepath`classes`m#rfparams,params,`xi`j`infogain`appliedrule`rulepath!(0N;0N;0n;(::);());
 tree: enlist[r0],$[98<>type r:.dtl.growTree r0;raze @[r;where 99h=type each r;enlist];r];
 tree: update p:{x?-1_'x} rulepath  from tree;
 `i`p`path xcols update path:{(x scan)each til count x}p,i:i from tree}

/
 Return a subtree containing only the leaves
\
.dtl.leaves:{[tree] select from tree where i in til[count p]except p}

/
 predict Y (classify) given a tree and an input Xi
 @param
  x    : a tuple of the features at a data point i, ie X[i]
  tree : a previously grown tree
 @return
     apply the rules of tree to X[i] using the previously constructed `ruelpath field and return the tree record containing the classification
\
.dtl.predictOnTree:{[x;tree]
 ({[x;tree]
  if[1=count tree;:tree];
  @[tree;`rulepath;1_'] where {value y[0],value[y 1]x}[x]each tree[`rulepath][;0]
  }[x]over)[.dtl.leaves tree]
 };

/
 Draw a bootstrap sample of size N from the training data using features specified by vector p (til count p for all features to be included)
 @param
  params: dictionary with tree input params. see: .dtl.learnTree
  p: index vector of features to sample (subset of 0...p-1)
  n: sample size
 @return

\
.dtl.bootstrapTree:{[params;p;n;B]
 z: .dtl.sampleTree[`x`y#params;p;n];
 tree_b:   .dtl.learnTree @[params;`x`y;:;z`x`y];
 tree_oob: raze .dtl.predictOnTree[;tree_b]each params[`x]z`oobi;
 tree_oob: update pred_error:abs obs_y-first each y from update obs_y:params[`y]z`oobi from tree_oob;
 `tree`oob!(`B xcols update B from tree_b;`B xcols update B from tree_oob)
 }

/
 Random Forest
 @param
  params: dictionary with keys
     x       : predictor
     y       : predicted
     rule    : the logical rule to apply
     classes : the k classification classes for y
     m       : the number of random features to sample at each split point (for classification m is usually set to sqrt p, where p=count features)
     p       : index vector of features to sample (subset of 0...p-1)
     n       : sample size for bootstrapping
     B       : number of bootstrap trees
\
.dtl.randomForest:{[params]
 ensemble: .dtl.bootstrapTree[params;params`p;params`n] peach til params`B;
 raze each flip ensemble}

/
 Predict classification of x (data), given an ensemble
 @param
     x        : datapoint to predict classification on: vector of m features
     ensemble : a random forest: a table of treetables
 @return
 classification of x based on majority rule
\
.dtl.predictOnRF:{[ensemble;data]
 rf:{[data;tree;b].dtl.predictOnTree[data]select from tree where B=b}[data;tree]each exec distinct B from tree:ensemble`tree;
 prediction: {first where x=max x}count each group exec first each y from raze rf;
 `prediction`mean_error!( prediction; select avg pred_error from ensemble`oob )
 }

\

x:1f*((1;1);(2;2);(3;3);(4;5);(5;6);(8;7));
.[`x;(::;1);+;-0.05+count[x]?0.1]; / make some noise
y:-8 -16 -24 -42 -50 -54f;
s:`x`y!(flip x;y);
z1:.dtl.sampleTree[s;til count s`x;count y];
tree:.dtl.learnTree @[s;`y;.dtl.classify -50 -25 0 25 50f],`rule`classes!(>;0 1 2);

/ all leaf nodes
select from tree where i in til[count p]except p

/ learn tree
n:10;
x:(n?100f;n?1000f;-25+n?50;n?5.5 257.7 3.3 -4);
y:-50f+n?100f;
s:`x`y!(x;y);
s:@[s;`y;.dtl.classify -50 -25 0 25 50f];
params:s,`rule`classes!(>;asc distinct s`y);
.dtl.learnTree params

/ bootstrapping
params:s,`rule`classes!(>;asc distinct s`y);
\ts b:.dtl.bootstrapTree[params;til count params`x;count params`y;0]

/ bootstrapping with random feature selection at each node -> random forest
params:s,`rule`classes`m!(>;asc distinct s`y;2);
\ts b:.dtl.bootstrapTree[params;til count params`x;n;0]
\ts ensemble:.dtl.randomForest params,`p`n`B!(til count params`x;n;5)


