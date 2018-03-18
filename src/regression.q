/
 * Created by aris on 12/18/17.
 Linear regression with bootstrap
\

/
 Least Squares model summary
 let the model be Y = X $ b, where Y is a vector[n], X is a matrix[n;m] and b is a vector[m]
 @params  x: predictor matrix[n;m]
          y: predicted vector[n]
 @return  dictionary of keys
          `b:      estimated coefficient vector[m]
          `Rsq:    R squared of regression (1-SSE%SSY)
          `adjRsq: adjusted R squared of regression, dealing with natural error increase as m grows
          `y:      predicted vector[n]
          `ybar:   estimated of predicted vector[n]
          `e:      error vector[n]
 @example
  x:1f*((1;1);(2;2);(3;3);(4;5);(5;6);(8;7));
  y:-8 -16 -24 -42 -50 -54f;
  .lsq.model[x;y]
  x:1f,'x;
  .lsq.model[x;y]
  -1.051603e-12 2 -10
  y+:3f;
  .lsq.model[x;y]
 https://en.wikipedia.org/wiki/Coefficient_of_determination
\
.lsq.model:{
 b:inv[flip[x] mmu x]mmu flip[x]$y; / Equivalent to: enlist[y] lsq flip x
 ybar: x $ b;
 e: y-ybar;
 Rsq: 1-sum[e*e]%count[y]*var y;
 p: count first x;
 if[all 1=x[;0];p-:1];
 n: count x;
 adjRsq: 1-(1-Rsq)*(n-1)%n-p-1;
 `b`Rsq`adjRsq`y`ybar`e!(b;Rsq;adjRsq;y;ybar;e)
 }

/
 bootstrap linear regression
 randomly select errors of first fit and refit y~ = y + e iteratively
 @params  x: predictor matrix[n;m]
          y: predicted vector[n]
          z: repeat process z times
 @return a table where each record is the result of .lsq.model
 @example
   x:1f*((1;1);(2;2);(3;3);(4;5);(5;6);(8;7));
   .[`x;(::;1);+;-0.05+count[x]?0.1]; / make some noise
   y:-8 -16 -24 -42 -50 -54f;
   .lsq.bootstrap[x;y;3]
\
.lsq.bootstrap:{
 model: .lsq.model[x;y];
 {[x;model]
  etilde: count[e]?e:model`e;
  ytilde: model[`y] + etilde;
  .lsq.model[x;ytilde]
 }[x]\[z;model]
 }

