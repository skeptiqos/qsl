/ algorithms to detect time series of unusual shapes

\l dm.q

/ x: point
/ y: vector
/ eg 2 dimensions/features: .lof.l1[(x;x);(v;v)]
/    1 dimention/feature  : .lof.l1[enlist x;enlist v]
.shape.l1:{sum abs x-y};

/ x: point
/ y: vector
/ eg 2 dimensions/features: .lof.l2[(x;x);(v;v)]
/    1 dimention/feature  : .lof.l2[enlist x;enlist v]
.shape.l2:{sqrt r wsum r:x-y};

.shape.zscore:{(x-avg x)%dev x};

/ .shape.pointDist: Point outliers: Distance of a point from all others in a time series
/ for subseries (ie collective outliers) we will need the concepts of discord/motif further below
/ Find the most distant point:
/  first idesc .shape.pointDist[x]
/ Find point with closest points (largest density)
/  last idesc .shape.pointDist[x]
/ Find points that have distance greater than D
/  x where .shape.pointDist[x]>D
.shape.pointDist:{{.shape.l2[x z]each x y except z}[x;tx]each tx:til count x};

.shape.subseqDist: {{.shape.l2[x z]each x y except z}[x;tx]each tx:til count x};

/ Given a timeseries x, return continuous sub-sequences (sub time series) of length m
.shape.subseqs:{[x;m] {y#z _ x }[x;m]each til count[x]-m-1};

/ .shape.discordMotif: Discord Motif :https://arxiv.org/pdf/2002.04236.pdf.
/ @param s: the time series
/ @param m: the window length of the subseries we want to examine for discords
/ @param z: boolean whether to de-mean and zscore or not
/ @return dictionary of subseries starting indices mapped to the distances d(D,D') in descending order
/ @ example x:10+15?10f;m:4;x,:m?5f; D: .shape.discordMotif[x;m;0b];
/ discord:
/ x[first[key D]+til m]
/ motif
/ x[last[key D]+til m]
.shape.discordMotif:{[s;m;z]
 S:.shape.subseqs[s;m]; / create all subsequences
 EI:{r where 0<=r:(y+til x)-(x:(2*x)-1) div 2}[m]each ts:til count S; / exclude indices of overlapping timeseries for each element of subseries
 D:{[s;ei;ts;i] .shape.l2[s i]each s ts except ei i}[$[z;.shape.zscore each S;S];EI;ts]peach ts;
 MI!M MI:idesc M:min each D
 };

/ .shape.discord - select top-k discord
/ @param s: the time series
/ @param m: the window length of the subseries we want to examine for discord
/ @param z: boolean whether to deman and zscore or not
/ @param k: the outlier threshold based on the k max distances
/ @example .shape.discord[x;m;0b;4]
.shape.discord:{[s;m;z;k] k#.shape.discordMotif[s;m;z]};
/ .shape.motif - select top-k motif
/ @param s: the time series
/ @param m: the window length of the subseries we want to examine for motif
/ @param z: boolean whether to deman and zscore or not
/ @param k: the outlier threshold based on the k min distances
.shape.motif:{[s;m;z;k] neg[k]#.shape.discordMotif[s;m;z]};

/ 2-sample Kolmogorov-Smirnov
/ @param s1: the first sample
/ @param s2: the second sample
/ @param a:  the significance level (probability of obtaining the results due to chance)
.shape.ks:{[s1;s2;a]
 cumf:{iasc[x]!(1+til cx)%cx:count x};
 cfi1:cumf s1;
 cfi2:cumf s2;
 cf1:`s#s1[key cfi1]!value cfi1;
 cf2:`s#s2[key cfi2]!value cfi2;
 / combine
 cf1:{asc[key x]#x} (0^key[cf2]#cf1),cf1;
 cf2:{asc[key x]#x} (0^key[cf1]#cf2),cf2;
 D:max abs cf1-cf2;
 ca:sqrt .5*neg log .5*a;    / c(a)
 c1:count s1;c2:count s2;
 `KSD`KSThresh!(D;ca*sqrt (c1+c2)%c1*c2)
 };