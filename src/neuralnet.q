\d .neuralnet

/
Input layer length
Count how many input features you use per example (per time step or per window).
Example: if for each sample you feed 20 past returns, 3 technical indicators, and 1 volume feature, your input vector length is
 20+3+1=24.

Hidden layer length
The hidden layer size is a hyperparameter. it just needs to be compatible with matrix multiplication (input_size × hidden_size, hidden_size × 2, etc.).
Typically 16, 32, 64, etc., and tuned via validation:
Too small: underfits, can’t capture patterns.
Too large: overfits, trains slower.

Simple model example:
Input: length = number of features (e.g., 24)
Hidden: 32 neurons with ReLU
Output: 2 neurons with softmax for [down, up].
\

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

/ dCdw(l): dCda * dadz * dzdw (chain rule)
/ -> del of Cost function wrt activation a at layer l * del of activation function wrt to linear output z at layer l * d of z wrt w
/ Let delta be the approximate error wrt to activation at layer l
/ delta(L): dCdz: dCda * dadz    / L is output layer
/ dCda: 2(a-y) for c=(a-y)xexp 2 / y is output
/ delta(l): dCdz: (T(W(l+1)) * delta(l+1)) * dadz / T(W) is weights transpose of the next layer. ie apply the transposte of weights to the error d(l) to get the error at d(l-1)
/ dadz: derivative of activation func, da
/ dzdw: a[l-1], a of previous layer
/ therefore: dCdw = a[l-1] mmu delta[l]
/ l: layer results: eg l`z l`a. pl: previous layer
/ ref: http://neuralnetworksanddeeplearning.com/chap2.html?utm_source=perplexity
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

/ train: pm
/ x:        input data: s-legnth list of n sized vectors (sxn), one for each sampled input
/           e.g. for classifying numbers 0-9, we need s=sample size, say 100 images, by n=900*700=630k pixels
/            for a chess board s=sample size of different positions, by n=64
/            predicting a price movement, s=1000 samples by n=23=20 past returns+1 volatility+1 volume feature+1 orderbook imbalance
/ y:        output data: Y-length list of m sized vectors (Yxm), one for each output.
/           e.g for classifying numbers 0-9, we need 10 sets of m=10-length 0/1 outputs, (1 0 0 0 0 0 0 0 0 0;0 1 0 0 0 0 0 0 0 0;..)
/           for up/down/same probabilities we need Y sets of m=3 (1 0 0;0 1 0;0 0 1),etc Y being the number of available sampled data
/ k:        hidden layers length
/ l:        number of hidden layers
/ e:        zero or small constant to initialise bias vector
/ eta:      learning rate
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
init:{[([x;y;k;l;e;eta;hactivf;hactivf_d;activf;cost;cost_d;batchsize;numepochs;history])]
 n:count first x;m:count first y;
 (!) . flip (
  (`X;x);
  (`Y;y);
  (`n;n);
  (`k;k);
  (`l;l);
  (`m;m);
  (`e;e);
  (`eta;eta);
  (`batchsize;batchsize);
  (`numepochs;numepochs);
  (`history;history);
  (`FP;`f`df`ff`c`dc!(hactivf;hactivf_d;activf;cost;cost_d)); / f : hidden layer activation function df: derivative of f ff: final layer activation function
  (`W; .neuralnet.initw . (n;k;l;m));
  (`B; .neuralnet.initb . (k;l;m;e)))
 };

train1NN:{[pm;wb;xy]
 x:xy 0;y:xy 1;
 I:([a:x; z:`float$(); w:(); b:(); d:()]);
 / last record of Prediction P will be the prediction we need to compare with output y;
 s:.z.n;
 P:I,.neuralnet.feedfwd[pm[`FP; `f]]\[I; wb`W; wb`B];
 fftime:.z.n-s;
 P:update a:pm[`FP; `ff] each z from P where i=max i; / final layer should apply final layer activation function eg softmax
 s:.z.n;
 G:.neuralnet.backpropagate[pm`FP; y] P;
 bptime:.z.n-s;
 `G`Y`C`fftime`bptime`step!(reverse G;y;pm[`FP; `c][y; first[G] `a];fftime;bptime;1)
 };

train1b:{[pm;wb;batchids]
 / select the (x;y) for the mini-batch's sampled indices
 Y:pm[`Y] batchids; X:pm[`X] batchids;
 / neural nets backpropagation for all (x;y) in the mini-batch
 nns:.Q.fc[train1NN[pm;wb]each;flip (X;Y)]; / parallelise
 / average the gradients across the training batch
 l:pm[`l]+1;
 Cw:l#avg nns[; `G; `nabla_cw];
 Cb:l#avg nns[; `G; `d];
 / use an optimiser like gradient decent to update the parameters
 w:l#nns[0;`G;`w];
 b:l#nns[0;`G;`b];
 `W`B`fftime`bptime`step!(w-pm[`eta]*Cw;b-pm[`eta]*Cb;wb[`fftime]+sum nns`fftime;wb[`bptime]+sum nns`bptime;nns[`step]+sum nns`step)
 };

train:{[pm]
 batchids:raze {[bs;s]bs cut neg[s]?s}[pm`batchsize]each pm[`numepochs]#count pm`X;
 initstate:(`W`B#pm),([step:0;fftime:0D;bptime:0D]);
 $[pm`history;train1b[pm]\[initstate;batchids];train1b[pm]/[initstate;batchids]] / iterate over batches using the weights estimator as an input to the next iteration
 };

argmax:{where x=max x};

validate:{[([x;y;hactivf;activf;nn;history])]
 if[history;nn:last nn];
 validate1[hactivf;activf;nn]'[x;y]
 };

validate1:{[hactivf;activf;nn;x;y]
 P:.neuralnet.feedfwd[hactivf]\[([a:x; z:`float$(); w:(); b:(); d:()]);nn`W;nn`B];
 all where[y]=argmax activf last[P`z]
 };

\d .

std:{(x-avg x)%dev x};
sigmoid:{reciprocal 1+exp neg x};
sigmoid_d:{x[y]*1-x[y]}sigmoid; / sigmoid derivative : f'(x)=f(x)(1-f(x))
relu: 0f|;
mse:{.5*avg xexp[x-y;2]};
mse_d:{2*x-y};
softmax:{ex%sum ex:exp x};
xentropy:{neg y wsum log x} / y: target (1,0,0) etc x: softmax(final layer output)
xentropy_d:{x-y};  / https://levelup.gitconnected.com/killer-combo-softmax-and-cross-entropy-5907442f60ba

readMNIST:{[typ;n]
 / MNIST dataset: https://drive.google.com/file/d/1eEKzfmEu6WKdRlohBQiqi3PhW_uIVJVP/view
 r:$[n;n#;::]read0 hsym `$getenv[`HOME],"/Downloads/MNIST_CSV/mnist_",string[typ],".csv";
 flip "I"$csv vs/:r
 };
