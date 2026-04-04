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

/ test with MNIST
ixyraw:readMNIST[`train;0];
pxyraw:readMNIST[`test;0];

/res0:initTrainPredict[ixyraw;pxyraw;()!()];
res1:.neuralnet.initTrainPredict[ixyraw;pxyraw;([Seed:-314159i;history:1b;k:32;l:1;updwf:(`static;([eta:0.1]));batchsize:64;numepochs:1])];
res2:.neuralnet.initTrainPredict[ixyraw;pxyraw;([Seed:-314159i;history:1b;k:32;l:1;updwf:(`static;([eta:0.1]));batchsize:64;numepochs:5])];
res3:.neuralnet.initTrainPredict[ixyraw;pxyraw;([Seed:-314159i;history:1b;k:32;l:1;updwf:(`static;([eta:0.1]));batchsize:128;numepochs:10])];
res4:.neuralnet.initTrainPredict[ixyraw;pxyraw;([Seed:-314159i;history:1b;k:64;l:2;updwf:(`static;([eta:0.15]));batchsize:128;numepochs:20])];
res5:.neuralnet.initTrainPredict[ixyraw;pxyraw;([Seed:-314159i;history:1b;k:64;l:2;updwf:(`static;([eta:0.15]));batchsize:128;numepochs:20;maxsteps:1000000])];
res6:.neuralnet.initTrainPredict[ixyraw;pxyraw;([Seed:-314159i;history:1b;k:64;l:2;updwf:(`static;([eta:0.15]));batchsize:128;numepochs:20;maxsteps:1000000;crthresh:0.05])];
res7:.neuralnet.initTrainPredict[ixyraw;pxyraw;([Seed:-314159i;history:1b;k:64;l:2;updwf:(`adam;([m1:.9;m2:.999;e:1e-8;a:.001]));batchsize:128;numepochs:20;maxsteps:1000000;crthresh:0.05])];

/ summary stats
summary:{last[x`validationstats],
  (exec finalAvgCost:last avgC,sum fftime,sum bptime,last step,last costreduction from x`nn),
   ((::;"n"$avg@)@'`traintime`predicttime#x),
  `k`l`e`updwf`batchsize`numepochs`maxsteps`crthresh`gradclipc# x`pm}each (res1;res2;res3;res4;res5;res6;res7);

show select k,l,eta:{x[1]`eta} each updwf,batchsize,numepochs,step,finalAvgCost,finalAvgValCost:valcost,traintime,accuracy from summary where i<3
show select k,l,eta:{x[1]} each updwf,batchsize,numepochs,step,maxsteps,crthresh,costreduction,finalAvgCost,finalAvgValCost:valcost,traintime,accuracy,predicttime from summary;
/ look at Cost function over training
fills (select step,avgC,devC,startC,endC,"f"$costreduction from res1[`nn]) lj `step xcol res1[`validationstats]
fills (select step,avgC,devC,startC,endC,"f"$costreduction from res7[`nn]) lj `step xcol res7[`validationstats]

