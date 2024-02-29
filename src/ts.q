/ time series funcs

PHI:`s#(!). ("FF"; csv)0:`:phi.csv;
IPHI:`s#value[PHI]!key PHI;
normd:{0^PHI"f"$x};
cumf:{iasc[x]!(1+til cx)%cx:count x};
zscore:{(x-avg x)%dev x};

/ .ts.qqn - normal qq plot
/ @param x: the time series for which to plot the quantiles against the std normal quantiles
.ts.qqn:{
 Y:x cumf[x] bin b:.01*1+til 99;
 X:PHI bin b;
 ([]X;Y)
 };

/ auto covariance
/ @param x: the time series vector
/ @param k: the lag
/ @example: autocovariance of first n lags: .ts.acov[x]peach til count n
.ts.acov:{[x;k] ((k _x)$#[c-k;x])%c:count x-:avg x};

/ auto correlation
/ @param x: the time series vector
/ @param k: the lag
.ts.acor:{[x;k] .ts.acov[x;k]%var x};




