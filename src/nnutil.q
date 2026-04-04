\d .nnu
std:{(x-avg x)%dev x};
rmsnorm:{x*(1e-5+x$x) xexp -.5};
sigmoid:{reciprocal 1+exp neg x};
sigmoid_d:{x[y]*1-x[y]}sigmoid; / sigmoid derivative : f'(x)=f(x)(1-f(x))
relu: 0f|;
mse:{.5*avg xexp[x-y;2]};
mse_d:{2*x-y};
softmax:{[t;x]ex%sum ex:exp x%t}; / t:temperature x:data
xentropy:{neg y wsum log x} / y: target (1,0,0) etc x: softmax(final layer output)
xentropy_d:{x-y};  / https://levelup.gitconnected.com/killer-combo-softmax-and-cross-entropy-5907442f60ba
\d .