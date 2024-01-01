/ discrete maths

/ factorial
.dm.fact:{prd 1+til x};
/ combinations (n;k)
.dm.comb:{[dm;n;k] prd[(n-k)+1+til k]%dm[`fact] k}.dm;
/ create binary sequence up to power n
/ @param x: power
/ @param y: boolean whether to drop 0b prefixes
/ @return binary equivalent of til 2 xexp n
.dm.tilbin:{s:0b vs/:til `long$2 xexp x;$[y;neg[x]#'s;s]};
/ grey code iterator
.dm.greycodeiter:{raze (0b,/:x;1b,/:reverse x)};
/ generage n-bit grey codes
.dm.greycode:{x[`greycodeiter]/[y-1;01b]}.dm;

/ find indices of pair-subsets of a set (C(n;2))
.dm.subsetpairis:{raze{{y,/:y _ x}[1_x]each(-1_x)}til[count x]};
/ return the pair-subsets of a set (C(n;2))
.dm.subsetpairs:{y@/:x[`subsetpairis] y}.dm;

/ return the C(n;k) subsets of a set
/ WARN: this can be a huge number and easily result in an overflow, eg:
/.dm.comb[100;10]
/ 2.060025e+12
/ NOTE can we create the subset upon generation of each greycode , thus avoiding the preallocation of memory for all grey codes before iterating?
.dm.subsetcombs:{[dm;s;k] s where each i where k=sum each i:dm[`greycode][count s]}.dm;
/ WARN: same as above but in addition slower than using greycodes
.dm.subsetcombs1:{[dm;s;k] s where each i where k=sum each i:dm[`tilbin][count s;1b]}.dm;





