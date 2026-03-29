\d .neuralnet

// ------ Core logic: FF and Backprop ------
/ feed forward: each layer is derived as a linear comb of the prev layer , with an activation func applied on top of that
/ to transform/squeeze the output into the desired one
/ eg sigmoid to convert a vector input with w weights - sigmoid(w*n) - to 0-1 probability
/ w: weights matrix, length k x n, k: nodes in next layer, n: nodes in previous layer
/ a: input activation nodes vector, length n
/ b: bias vector, length n (positive to have bias to be active, negative to have bias to be inactive, ie <0)
/ f: activation function,
/    eg sigmoid (for probability 0-1),
/    ReLU (Rectified Linear Unit), used in modern Deep NNs, good for learning one class at a time , ReLU=max(0,a)
/ z: linear combination of the prev layer
zf:{[b;w;a] b+w$a};

/ a=f z-> activation node at layer l. f can be eg the sigmoid function.
/ When forward feeding we should calculate and store both z and a=f[z]
/ .neuralnet.feedfwd[FP`f]\[I;W;B]
feedfwd:{[f;l;w;b] ([a:f z;z:z:zf[b;w;l`a];w;b;d:()]) };

nabla:{[dcda;dadz;l;pl]
 z:l`z;d:l`d;i:l`l;w:l`w; / nw -> next w (careful on the indexing)
 delta:$[not i;dcda;flip[w] mmu d] * dadz z;
 nabla_cw: flip[enlist delta] mmu enlist pl`a;
 pl,([nabla_cw;d:delta]) / nabla_cb~delta
 };

backpropagate:{[fp;y;a]
 a:update w:prev w,b:prev b,nabla_cw:count[i]#() from reverse a;
 a:update l:i from a;
 dcda:fp[`dc][a[0;`a];y];  / cost func derivative at output layer based on expected y and output activation (prediction) a
 .neuralnet.nabla[dcda;fp`df] scan a
 };

// ------ Train and Predict ------
train1NN:{[pm;wb;xy]
 x:xy 0;y:xy 1; s:.z.n;
 I:([a:x; z:`float$(); w:(); b:(); d:()]);
 / here W at layer l W[l] and B[l] are the weights applied to compute z[l] and a[l]
 / we append initial a[0] (input data X) since we will need a[l-1] in backprop
 P:I,.neuralnet.feedfwd[pm[`FP; `f]]\[I; wb`W; wb`B];
 fftime:.z.n-s;
 / last record of Prediction P will be the prediction we need to compare with output y;
 P:update a:pm[`FP; `ff] each z from P where i=max i; / final layer should apply final layer activation function eg softmax
 s:.z.n;
 G:.neuralnet.backpropagate[pm`FP; y] P;
 bptime:.z.n-s;
 `G`Y`C`fftime`bptime`step!(reverse G;y;pm[`FP; `c][last[P] `a;y];fftime;bptime;1)
 };

updwf.static:{[pm] pm[`w]-pm[`eta]*pm`g};
updwf.adam:{[pm]
 m1:pm`m1;m2:pm`m2;e:pm`e;a:pm`a;t:pm`t;pp:pm`pp;pq:pm`pq;g:pm`g;w:pm`w;
 p:(m1*pp)+(1-m1)*g;
 q:(m2*pq)+(1-m2)*g*g;
 phat:p%1-m1 xexp t;
 qhat:q%1-m2 xexp t;
 (w-a*phat%e+sqrt qhat;p;q)
 };

applyUpdWf:{value (updwf x 0;x[1],y)};

train1B:{[pm;wb]
 / select the (x;y) for the mini-batch's sampled indices
 Y:pm[`Y] batchids:first wb`batchids; X:pm[`X] batchids;
 / neural nets backpropagation for all (x;y) in the mini-batch
 nns:.Q.fc[train1NN[pm;wb]each;flip (X;Y)]; / parallelise
 / average the gradients across the training batch
 l:pm[`l]+1;
 Cw:l#avg nns[; `G; `nabla_cw];
 Cb:l#avg nns[; `G; `d];
 / use an optimiser like gradient decent to update the parameters
 w:l#nns[0;`G;`w];
 b:l#nns[0;`G;`b];
 avgC:avg nns`C;
 wpm:([w:w;g:Cw]);
 bpm:([w:b;g:Cb]);
 wpp:wpq:bpp:bpq:0n;
 if[`static~first pm`updwf;nw:applyUpdWf[pm`updwf;wpm];nb:applyUpdWf[pm`updwf;bpm]];
 if[`adam~first pm`updwf;
   (nw;wpp;wpq):applyUpdWf[pm`updwf;wpm,([t:wb`batchid;pp:wb`wpp;pq:wb`wpq])];
   (nb;bpp;bpq):applyUpdWf[pm`updwf;bpm,([t:wb`batchid;pp:wb`bpp;pq:wb`bpq])]];
 `W`B`avgC`devC`startC`endC`fftime`bptime`step`batchid`wpp`wpq`bpp`bpq`costreduction`batchids!(
  nw;
  nb;
  avgC;
  dev nns`C;
  first nns`C;
  last nns`C;
  wb[`fftime]+sum nns`fftime;
  wb[`bptime]+sum nns`bptime;
  wb[`step]+sum nns`step;
  wb[`batchid]+1;
  wpp;wpq;bpp;bpq;
  (wb[`costreduction]*1-pm[`cremal])+pm[`cremal]*avgC;
  1 _ wb`batchids)
 };

/ train: MiniBatch SGD
/ X:        input data: s-legnth list of n sized vectors (sxn), one for each sampled input
/           e.g. for classifying numbers 0-9, we need s=sample size, say 100 images, by n=900*700=630k pixels
/            for a chess board s=sample size of different positions, by n=64
/            predicting a price movement, s=1000 samples by n=23=20 past returns+1 volatility+1 volume feature+1 orderbook imbalance
/ Y:        output data: Y-length list of m sized vectors (Yxm), one for each output.
/           e.g for classifying numbers 0-9, we need 10 sets of m=10-length 0/1 outputs, (1 0 0 0 0 0 0 0 0 0;0 1 0 0 0 0 0 0 0 0;..)
/           for up/down/same probabilities we need Y sets of m=3 (1 0 0;0 1 0;0 0 1),etc Y being the number of available sampled data
/ avgx/devx: avg and deviation of input X set. use these to normalise test X.
/ k:        hidden layers length
/ l:        number of hidden layers
/ e:        zero or small constant to initialise bias vector
/ updwf:    upwdate-weights function
/ hactivf:  hidden activation function: eg ReLU, used in modern Deep NNs, good for learning one class at a time by removing the negative components
/ activf:   activation function:
/            sigmoid:for binary output 0-1/is or isnt/true or false. Can also be used with multi-labelling where output vector can have multiple ones eg (1 0 1)
/            softmax:for multiclass , assigns a probability
/ activf_d: activation function derivative
/ cost:     cost function: MSE (mean square error) for regression or cross-entropy/AUROC for classification
/ cost_d:   cost function derivative
/ batchsize:batch size. we will sample s%b batches Bi, where s is training sample size and b is batch size
/           {i0,..iB},{iB+1,...i2B},...,{iNB+1,...S}, where S is the training data sample size and i are randomly picked indices
/           mini-batch: apply SGD (avg over N gradients) for a batch eg 32/64/128 examples (for datasize S in thousands) instead of whole training data
/           rough rule: S/batch_size ~= 50-200
/ numepochs:number of epochs: each epoch is the training set, and for each epoch we sample B size bathces
/ n:input vector length k:hidden layer length; l: num of hidden layers; m:output vector length
trainMBSGD:{[pm]
 batchids:raze {[bs;s]bs cut neg[s]?s}[pm`batchsize]each pm[`numepochs]#count pm`X;
 initstate:(`W`B#pm),([avgC:0;devC:0n;startC:0n;endC:0n;fftime:0D;bptime:0D;step:0;batchid:1;wpp:0;wpq:0;bpp:0;bpq:0;costreduction:1;batchids:batchids]);
 stopcond:{all (x>z`step;y<z`costreduction;count z`batchids)}[pm`maxsteps;pm`crthresh];
 / iterate over batches using each steps estimated weights as an input to the next iteration
 $[pm`history;train1B[pm]\[stopcond;initstate];train1B[pm]/[stopcond;initstate]]
 };

argmax:{where x=max x};

predict:{[hactivf;activf;nn;x]
 P:.neuralnet.feedfwd[hactivf]\[([a:x; z:`float$(); w:(); b:(); d:()]);nn`W;nn`B];
 argmax activf last[P`z]
 };

validate1:{[hactivf;activf;nn;x;y] all where[y]=predict[hactivf;activf;nn;x]};

validate:{[([x;y;hactivf;activf;nn;history])]
 if[history;nn:last nn];
 validate1[hactivf;activf;nn]'[x;y]
 };

// ------ Init Stuff ------

setSeed:{system"S ",string x;-1"Seed S:",string system"S";};

/ initiate weights: this is an array of weight matrices, one W matrix for each layer
/ n: input length
/ k: hidden layers length
/ l: number of hidden layers
/ m: output length
initw:{[n;k;l;m]
 r:sqrt 6%n; / Uniform He: recommended for randomisation of weights
 wi:{[r;n;k]r*-1+n?2f}[r;n]each til k;
 wk:{[r;k;l]{[r;k;l]r*-1+k?2f}[r;k]each til k}[r;k]each til l-1;
 wo:{[r;k;m]r*-1+k?2f}[r;k]each til m;
 enlist[wi],wk,enlist wo
 };

/ initiate bias vectors: array of l+1 bias vectors
/ e: initialize to zeros or small constants.
initb:{[k;l;m;e]
 b:l#enlist k#e;
 b,enlist m#e
 };

initParam:{[pm]
 if[`Seed in key pm;.neuralnet.setSeed pm`Seed];
 n:count first pm`X;m:count first pm`Y;
 pm,
 (!) . flip (
  (`n;n);
  (`m;m);
  (`FP;`f`df`ff`c`dc!(pm`hactivf`hactivf_d`activf`cost`cost_d)); / f : hidden layer activation function df: derivative of f ff: final layer activation function
  (`W; .neuralnet.initw . (n;pm`k;pm`l;m));
  (`B; .neuralnet.initb . (pm`k;pm`l;m;pm`e)))
 };

/ pm:`x`avgx`devx!(..)
/ avgx devx are optional:should be calculated for train data, and passed *from* train data values when testing (predicting)
normalise:{[pm]
 rx:raze x:x%max over x:pm`x;
 ax:$[`avgx in key pm;pm`avgx;avg rx];
 dx:$[`devx in key pm;pm`devx;dev rx];
 `x`normx`avgx`devx!(pm`x;(x-ax)%dx;ax;dx)
 };

\d .


std:{(x-avg x)%dev x};
sigmoid:{reciprocal 1+exp neg x};
sigmoid_d:{x[y]*1-x[y]}sigmoid; / sigmoid derivative : f'(x)=f(x)(1-f(x))
relu: 0f|;
mse:{.5*avg xexp[x-y;2]};
mse_d:{2*x-y};
softmax:{[t;x]ex%sum ex:exp x%t}; / t:temperature x:data
xentropy:{neg y wsum log x} / y: target (1,0,0) etc x: softmax(final layer output)
xentropy_d:{x-y};  / https://levelup.gitconnected.com/killer-combo-softmax-and-cross-entropy-5907442f60ba
