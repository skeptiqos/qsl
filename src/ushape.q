/ algorithms to detect time series of unusual shapes

\l dm.q

/ x: point
/ y: vector
/ eg 2 dimensions/features: .lof.l1[(x;x);(v;v)]
/    1 dimention/feature  : .lof.l1[enlist x;enlist v]
.ushape.l1:{sum abs x-y};

/ x: point
/ y: vector
/ eg 2 dimensions/features: .lof.l2[(x;x);(v;v)]
/    1 dimention/feature  : .lof.l2[enlist x;enlist v]
.ushape.l2:{sqrt r wsum r:x-y};

.ushape.zscore:{(x-avg x)%dev x};

/ .ushape.pointDist: Point outliers: Distance of a point from all others in a time series
/ for subseries (ie collective outliers) we will need the concepts of discord/motif further below
/ Find the most distant point:
/  first idesc .ushape.pointDist[x]
/ Find point with closest points (largest density)
/  last idesc .ushape.pointDist[x]
/ Find points that have distance greater than D
/  x where .ushape.pointDist[x]>D
.ushape.pointDist:{{.ushape.l2[x z]each x y except z}[x;tx]each tx:til count x};

.ushape.subseqDist: {{.ushape.l2[x z]each x y except z}[x;tx]each tx:til count x};

/ Given a timeseries x, return continuous sub-sequences (sub time series) of length m
.ushape.subseqs:{[x;m] {y#z _ x }[x;m]each til count[x]-m-1};

/ .ushape.applyNonSelfMatch - apply a function , test, or a distance measure to non-overlapping sub-timeseries
/ @param f    : the function , test, or a distance measure to apply
/ @param rf   : the reduce-filter, the filter applied to the result of the function being applied to subseries[i] vs all its non-self matches
/ @param s    : the parent time series
/ @param m    : the length of the subseries
/ @param z    : boolean whether to de-mean and zscore or not
.ushape.applyNonSelfMatch:{[f;rf;s;m;z]
 if[z;s:.ushape.zscore s];
 S:.ushape.subseqs[s;m]; / create all subsequences
 EI:{r where 0<=r:(y+til x)-(x:(2*x)-1) div 2}[m]each ts:til count S; / exclude indices of overlapping timeseries for each element of subseries
 {[f;rf;s;ei;ts;i] rf f[s i]each s ts except ei i}[f;rf;S;EI;ts]peach ts
 };

/ .ushape.discordMotif: Discord Motif :https://arxiv.org/pdf/2002.04236.pdf.
/ @param s: the time series
/ @param m: the window length of the subseries we want to examine for discords
/ @param z: boolean whether to de-mean and zscore or not
/ @return dictionary of subseries starting indices mapped to the distances d(D,D') in descending order
/ @ example x:10+15?10f;m:4;x,:m?5f; D: .ushape.discordMotif[x;m;0b];
/ discord:
/ x[first[key D]+til m]
/ motif
/ x[last[key D]+til m]
.ushape.discordMotif:{[s;m;z]
 MD:.ushape.applyNonSelfMatch[.ushape.l2;min;s;m;z];
 MI!MD MI:idesc MD
 };

/ .ushape.discord - select top-k discord
/ @param s: the time series
/ @param m: the window length of the subseries we want to examine for discord
/ @param z: boolean whether to deman and zscore or not
/ @param k: the outlier threshold based on the k max distances
/ @example .ushape.discord[x;m;0b;4]
.ushape.discord:{[s;m;z;k] k#.ushape.discordMotif[s;m;z]};
/ .ushape.motif - select top-k motif
/ @param s: the time series
/ @param m: the window length of the subseries we want to examine for motif
/ @param z: boolean whether to deman and zscore or not
/ @param k: the outlier threshold based on the k min distances
.ushape.motif:{[s;m;z;k] neg[k]#.ushape.discordMotif[s;m;z]};









