/q neuralnet_example.q -s 8  -c 30 204 -p 5001

.utl.require"qsl/src/neuralnet.q";

readMNIST:{[typ;n]
 -1 "Reading ",string[typ]," MNIST data";
 / MNIST dataset: https://drive.google.com/file/d/1eEKzfmEu6WKdRlohBQiqi3PhW_uIVJVP/view
 r:$[n;n#;::]read0 hsym `$getenv[`HOME],"/Downloads/MNIST_CSV/mnist_",string[typ],".csv";
 S:flip "I"$csv vs/:r;
 Y:@[10#0;;:;1]each Y:first S;
 X:flip 1_ S;   / each of the X is 28x28=784 pixels
 `X`Y!(X;Y)};

prepData:{[xy;typ;pm]
 -1 string[typ]," on ",string[count xy`X]," data";
 `X`Y!(.neuralnet.normalise[([x:xy`X]),pm];xy`Y)
 };

initTrainPredict:{[ixyraw;pxyraw;pmi]
 xy:prepData[ixyraw;`train;()!()];
 x:xy[`X];
 / pm defaults
 pmd:([X:x`normx; Y:xy`Y; avgx:x`avgx; devx:x`devx;
       k:32; l:1; e:x`avgx; eta:0.1;
       hactivf:relu; hactivf_d: >[;0]; activf:softmax 0.99;
       cost:xentropy; cost_d:xentropy_d;
       batchsize:64; numepochs:1;
       history:0b]);
 pm0:pmd,pmi;
 pm:.neuralnet.initParam[pm0];
 -1 "Model Params:";
 show `X`Y _ pm;
 -1 ".neuralnet.train: MiniBatch Stochastic Gradient Decent";s:.z.n;
 nn:.neuralnet.trainMBSGD[pm];
 -1 ".neuralnet.train time:",string traintime:.z.n-s;
 / test data - predict
 pxy:prepData[pxyraw;`test;`avgx`devx#pm];
 -1 ".neuralnet.validate prediction on ",string[count pxyraw`X]," test data";s:.z.n;
 predict:.neuralnet.validate[([x:pxy[`X]`normx; y:pxy`Y; hactivf:pm[`FP;`f]; activf:pm[`FP;`ff]; nn: nn; history:pm`history])];
 predicttime:.z.n-s;
 -1"prediction accuracy:",string[{100*sum[x]%count x}predict],"\n";
 `nn`predict`pm`traintime`predicttime!(nn;predict;pm;traintime;predicttime)};

/ test with MNIST
ixyraw:readMNIST[`train;0];
pxyraw:readMNIST[`test;0];

/res0:initTrainPredict[ixyraw;pxyraw;()!()];
res1:initTrainPredict[ixyraw;pxyraw;([Seed:-314159i;history:1b;k:32;l:1;eta:0.1;batchsize:64;numepochs:1])];
res2:initTrainPredict[ixyraw;pxyraw;([Seed:-314159i;history:1b;k:32;l:1;eta:0.1;batchsize:64;numepochs:5])];
res3:initTrainPredict[ixyraw;pxyraw;([Seed:-314159i;history:1b;k:32;l:1;eta:0.1;batchsize:128;numepochs:10])];

/res0[`nn;`B]~last res1[`nn;`B]

/ summary stats
summary:{([accuracy:{100*sum[x]%count x} x`predict]),
  (exec last endC,sum fftime,sum bptime,last step from x`nn),
  (`traintime`predicttime#x),
  `k`l`e`eta`batchsize`numepochs# x`pm}each (res1;res2;res3);

show select k,l,eta,batchsize,numepochs,step,accuracy,endC,traintime from summary;
/ look at Cost function over training
select step,avgC,devC,startC,endC from res1[`nn]
