\d .neuralnet

/ feed forward: each layer is derived as a linear comb of the prev layer, with an activation func applied on top of that
/  to transform/squeeze the output into the desired one (eg sigmoid to convert to 0-1 probability)
/ w: weights matrix, length k x m, k: nodes in next layer, m: nodes in previous layer
/ n: input activation nodes vector, length m
/ b: bias vector, lenght n
/ a: activation function, eg sigmoid (for probability 0-1), ReLU (good for learning one class at a time) , etc
z:{[b;w;n] b+w$n};

feedfwd:{[b;w;n;a] a z[b;w;n] };  / ->n node at layer L. Note when forward feeding we should also calculate the gradient (da)

/ how does a network learn? by minimising the cost function which is the prediction error
/ the gradient of the cost function gives us which changes to the weights matter most to minimise error
/ backpropagation lets us compute that negative gradient
/ how to adjust/emphasize a node's prediction to nudge it in the right direction?
/ we either
/ - increase bias
/ - increase weight ( in proportion to node value n)
/ - change the node value (in proportion to their weights)
/ we end up with calculating delta changes for the weights matrix w at each layer
/ Each of these partial derivatives/changes are the components of the Gradient vector
/ The new value of the weight will then be w = w -r dCdw , r is the learning rate < 1

/ if C is cost function, n the value of the node in previous layer and y the expected output, then
/ c: 2 xexp n-y
/ n: feedfwd
/ We want to calculate

/ 1. the rate of cost change when changing the weight:
/ dCdw: dCdnL * dnLdz * dzdw (chaing rule)
/ z= b+w$n
/ n=a[z]
/ dCdnL: 2(n-y)
/ dnLdz: derivative of activation func, da (sigmoid func etc)
/ dzdw: n (n of L minus 1, ie of previous layer)
dCdw1:{[b;w;n;a;da;y]
 nL: a zL:z[b;w;n]; / nL: feedfwd[b;w;n;a]; nL: new node at L
 2*(nL-y)*da[zL]*n
 };
/ we then average over all training examples: avg dCdw

/ 2. the rate of cost change wrt to the bias
/ dCdb; dCdnL * dnLdz * dzdb
/ dzdb: 1
dCdb1:{[b;w;n;a;da;y]
 nL: a zL:z[b;w;n]; / nL: feedfwd[b;w;n;a];
 2*(nL-y)*da[zL]
 };

/ 3. rate of cost change wrt to the previous activation node
/ dCdn: dCdnL * dnLdz * dzdn
/ dzdn: w
dCdn1:{[b;w;n;a;da;y]
 nL: a zL:z[b;w;n]; / nL: feedfwd[b;w;n;a];
 2*(nL-y)*da[zL]*w
 };


/generalise to matrix (more than one nodes per layer L)
/ *verify below:WIP* need to link nk (node of previous layer for k index) fwd to nLj (node of layer L for j index)
dCdw:{[b;w;n;a;da;y]
 nL: a zL:z[b;w;n]; / nL: feedfwd[b;w;n;a]; nL: new node at L
 sum[2*(nL-y)] * da[zL]$n
 };

/ here y is a vector all js
/ w is also a vector of all wk's (ie all nodes in previous layer) that will eventually produce *one* node (nL) in this layer
dCdw:{[b;w;n;a;da;y]
 sum dCdw1[b;w;n;a;da;y] };


/ because calculating the above neg gradient for all weights is computationally expensive we seggregate our training set
/ into mini-batches


\d .

sigmoid:{reciprocal 1+exp neg x};
sigmoid_d:{x[y]*1-x[y]}sigmoid; / sigmoid derivative : f'(x)=f(x)(1-f(x))
relu: 0f|;

m:64;    / length of chess board vector
k:3;
n:1f*m?0b;  / random starting position for a piece
w:{-1+m?2f}each til k; / for next hidden layer, map n vector into a 3 element vector
b:k#0f;

.neuralnet.feedfwd[w;n;b;sigmoid]