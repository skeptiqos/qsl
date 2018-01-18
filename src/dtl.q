/
 * Created by aris on 12/23/17.
 Decision tree learning
 tree construction based on http://archive.vector.org.uk/art10500340
\

/
 SampleTree: used for bootstrapping, samples a subset of the dataset to grow a tree from
 @param
  s: dataset `x`y!(predictors;predicted)
  p: index vector of features to sample (subset of 0...p-1)
  n: sample size
 @return a dictionary `x`y!(x;y) subsets of predictors and corresponding predicted values
 @example
 .dtl.sampleTree[`x`y!(x;y);til count flip x;count x]
 .dtl.sampleTree[s;til count flip s`x;count x]
\
.dtl.sampleTree:{[s;p;n] @[s;`x;{flip flip[x]y}[;p]]@\:n?til n}

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
 applyRule
 applies rule on a split of x at j to categorise classes
 @param
  y    : vector of sampled predicted variable
  x    : matrix of predictor variables
  xi   : the ith predicted variable
  rule : rule to split on
  j    : split xi at point j based on rule
\
.dtl.applyRule:{[y;x;xi;rule;j]
 (`x`y`appliedrule!((x;y)@\:where not b),enlist (not;rule);
  `x`y`appliedrule!((x;y)@\:where b:eval (rule;xi;j)),enlist rule)}

/
 choose optimal split for a node
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
 cx:count fx:flip x;
 info: .dtl.chooseSplitXi[y;x;rule;classes]peach fx@sampled:asc neg[m]?cx;
 oob:  til[cx] except sampled;
 summary: (`xi`j`infogain!i,(-1_r)),/:last r:raze info i:first idesc info[;1;0];
 cnt: count summary;
 res: update rule:rule,rulepath:{[rp;r;i;j] rp,enlist (r;`$"x",string i;j)}[rulepath]'[appliedrule;xi;j],classes:cnt#enlist classes from summary;
 update m,oob from res
 }

.dtl.chooseSplitXi:{[y;x;rule;classes;xi]
  j:asc distinct xi;
  info:{[y;x;xi;rule;classes;j]
   split:.dtl.applyRule[y;x;xi;rule;j];
   (.dtl.infogain[y;split`y;classes];split)
   }[y;x;xi;rule;classes] each j;
   (j;info)@\:first idesc flip[info]0
 }

/
 for each of the records in the initial split, we iterate until we reach pure nodes (leaves)
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
.dtl.growTree:{[r]
 {if[1=count distinct x`y;:x];
  enlist[x],$[98h<>type rr:.dtl.growTree[x];raze @[rr;where 99h=type each rr;enlist];rr]
 }each  r:.dtl.chooseSplit[r]}


.dtl.learnTree:{[params]
 rfparams:`m`oob!(count flip params`x;());
 r0:`xi`j`infogain`x`y`appliedrule`rule`rulepath`classes`m`oob#rfparams,params,`xi`j`infogain`appliedrule`rulepath!(0N;0N;0n;(::);());
 tree: enlist[r0],raze @[r;where 99h=type each r:.dtl.growTree r0;enlist];
 tree: update p:{x?-1_'x} rulepath  from tree;
 `i`p`path xcols update path:{(x scan)each til count x}p,i:i from tree}

.dtl.randomForest:{[params]
 /{.dtl.learnTree params}peach x;

 }

\

x:1f*((1;1);(2;2);(3;3);(4;5);(5;6);(8;7));
.[`x;(::;1);+;-0.05+count[x]?0.1]; / make some noise
y:-8 -16 -24 -42 -50 -54f;
s:`x`y!(x;y);
z1:.dtl.sampleTree[s;til count flip s`x;count x];
tree:.dtl.learnTree @[s;`y;.dtl.classify -50 -25 0 25 50f],`rule`classes!(>;0 1 2);

/ all leaf nodes
select from tree where i in til[count p]except p

/ random forest
n:100;
x:flip (n?100f;n?1000f;-25+n?50);
y:-50f+n?100f;
s:`x`y!(x;y);
s:@[s;`y;.dtl.classify -50 -25 0 25 50f];
\ts tree:.dtl.learnTree s,`rule`classes!(>;asc distinct s`y)
\ts tree:.dtl.learnTree s,`rule`classes`m!(>;asc distinct s`y;2)


