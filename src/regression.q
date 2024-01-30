/
 * Created by aris on 12/18/17.
 Linear regression with bootstrap
\

/
 Least Squares model summary
 let the model be Y = X $ b, where Y is a vector[n], X is a matrix[m;n] and b is a vector[m]
 @params  x: predictor matrix[m;n]
          y: predicted vector[n]
 @return  dictionary of keys
          `b:      estimated coefficient vector[m]
          `Rsq:    R squared of regression (1-SSE%SSY)
          `adjRsq: adjusted R squared of regression, dealing with natural error increase as m grows
          `y:      predicted vector[n]
          `ybar:   estimated of predicted vector[n]
          `e:      error vector[n]
     `bTstat:      slope tstat
 @example
x:"f"$19,34,21,21,21,21,21,22,62,31,24,24;
y:"f"$155,166,168,185,173,186,177,188,165,173,167,165;
.lsq.model[(count[x]#1f;x);y]
x:flip 1f*((1;1);(2;2);(3;3);(4;5);(5;6);(8;7));
y:-8 -16 -24 -42 -50 -54f;
.lsq.model[x;y]
 https://en.wikipedia.org/wiki/Coefficient_of_determination
\
.lsq.model:{
 b:first enlist[y] lsq x; / inv[x mmu flip x]mmu x$y
 ybar: flip[x]$b;
 e: y-ybar;
 sse:e$e;
 n:count y;
 Rsq: 1-sse%n*var y;
 p: count x;
 bx:b;
 if[all 1=first x;p-:1;x:1_ x;bx:1_ b];
 adjRsq: 1-(1-Rsq)*(n-1)%(n-p)-1;
 se:sqrt sse%(n-p)-1;
 ssx:n*var each x;
 bTstat:bx%se%sqrt ssx ;
 `b`Rsq`adjRsq`y`ybar`e`bTstat!(b;Rsq;adjRsq;y;ybar;e;bTstat)
 }

/
 bootstrap linear regression (residual bootstrap)
 randomly select errors of first fit and create new ybar y~ = y + e iteratively
 @params  x: predictor matrix[m;n]
          y: predicted vector[n]
          z: repeat process z times
 @return a table where each record is the result of .lsq.model
 @example
x:"f"$19,34,21,21,21,21,21,22,62,31,24,24;
y:"f"$155,166,168,185,173,186,177,188,165,173,167,165;
.lsq.bootstrap[(count[x]#1f;x);y;3
\
.lsq.bootstrap:{
 model: .lsq.model[x;y];
 {[x;model;i]
  etilde: count[e]?e:model`e;
  ytilde: model[`ybar] + etilde;
  .lsq.model[x;ytilde]
 }[x;model]each til z
 }

