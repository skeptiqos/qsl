/hypothesis testing

\l ushape.q

PHI:`s#(!). ("FF"; csv)0:`:phi.csv;
normd:{0^PHI"f"$x};

/ 1 or 2-sample Kolmogorov-Smirnov (KS Test)
/ @param  s1: the first sample
/ @param  s2: the second sample - when empty list, performs a 1 sample KS Test against normal CDF (normality test)
/ @param alt: `twoside`less`greater. indicates alternative hypothesis
/ @param   a:  the significance level (probability of obtaining the results due to chance)
.ht.KSTest:{[s1;s2;alt;a]
 cumf:{iasc[x]!(1+til cx)%cx:count x};
 cfi1:cumf s1;
 cf1:`s#s1[key cfi1]!value cfi1;
 if[count s2;
    cfi2:cumf s2;
    cf2:`s#s2[key cfi2]!value cfi2;
    / combine
    cf1:0^(k:asc[key[cf1],key cf2])#cf1;
    cf2:0^k#cf2;
   ];
 if[not count s2;cf2:key[cf1]!normd[key cf1];s2:s1]; / if s~(), perform 1 sample normality test
 op:$[alt=`less;neg;alt=`greater;::;abs];
 if[not alt in `less`greater;a:.5*a];
 D:max op cf1-cf2;
 ca:sqrt .5*neg log a;    / c(a)
 c1:count s1;c2:count s2;
 `KSD`KSThresh!(D;ca*sqrt (c1+c2)%c1*c2)
 };

/ .ht.ks - apply KS Test to all subseries of a timeseries and return the pairs that reject the hypothesis uat a given significance level
/ @param s: the time series
/ @param m: the window length of the subseries
/ @param alt: `twoside`less`greater
/ @param z: boolean whether to deman and zscore or not
/ @param a: the significance level (probability of obtaining the results due to chance)
/ @return a table with the starting index of each subseries for which the KSTest statistic has breached the threshold as well as the statistic and threshold values
/ Note: too noisy for many subseries of a periodic timeseries
/  one with most flagged entries in the ks test may not give meaningful results - discord could be a better measure
.ht.ks:{[s;m;alt;z;a]
 rf:?[;enlist (>;`KSD;`KSThresh);0b;()];
 ksr:.ushape.applyNonSelfMatch[.ht.KSTest[;;alt;a];rf;s;m;z];
 update idx:{raze (c idx)#'idx:where 0<c:count each x}[ksr] from raze ksr
 };

