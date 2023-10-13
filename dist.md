## Multivariate weighted average

Imagine a multivariate function $\mathbf{x}:\mathbb{R} \to \mathbb{R}^n$, with variables individually approximated with m basis functions:

$$ \mathbf{x}(t)=\begin{bmatrix}
    x_1(t)\\
    \vdots\\
     x_n(t)\\
\end{bmatrix}=\begin{bmatrix}
    \beta_{11}f_1(t)+ \dots + \beta_{1m}f_m(t)\\
    \vdots\\
     \beta_{n1}f_1(t)+ \dots + \beta_{nm}f_m(t)\\
\end{bmatrix}=$$

We already know how to approximate a real time series with these m basis functions and a weight density function. Now we are interested in, given a weighted window provided by a weight which is a function of the elapsed time $-t$:

$$w(-t)=e^{\frac{-t}{\sigma}}$$

And a value $\mathbf{x'} \in \mathbb{R}^n$, define a similarity between $\mathbf{x'}$ and $x(t)$ for any $t$:

$$l_{\mathbf{x'}}(t)=\mathbf{x'}\cdot\mathbf{x}(t) \ w(t)$$

And its normalized form to crate a distribution over all $t$:

$$p_{\mathbf{x'}}(t)=\frac{l_{\mathbf{x'}}(t)}{\int_{-\infty}^0 l_{\mathbf{x'}}(t)dt}$$

And compute the average value of the time series according to that distribution:

$$ \hat{\mathbf{x}}=\int_{-\infty}^0\mathbf{x}(t) p_{\mathbf{x'}}(t)dt$$

Which can be rewritten as:

$$\hat{\mathbf{x}}=\frac{\int_{-\infty}^0\mathbf{x}(t)l_{\mathbf{x'}}(t)dt}{\int_{-\infty}^0 l_{\mathbf{x'}}(t)dt}$$

### Denominator

To compute the denominator:

$$\int_{-\infty}^0l_{\mathbf{x'}}(t)dt=\int_{-\infty}^0\mathbf{x'}\cdot\mathbf{x}(t) \ e^{\frac{t}{\sigma}}dt$$

We can first decompose the dot product into the sum of the product of each dimensions

$$\int_{-\infty}^0l_{\mathbf{x'}}(t)dt=\int_{-\infty}^0 \sum_{i=1}^nx_i'x_i(t) \ e^{\frac{t}{\sigma}}dt$$

With a small rearrangement:

$$\int_{-\infty}^0l_{\mathbf{x'}}(t)dt=\sum_{i=1}^nx_i'\int_{-\infty}^0 x_i(t) \ e^{\frac{t}{\sigma}}dt$$

Using $x_i(t)=\beta_{i1}f_1(t)+ \dots + \beta_{im}f_m(t)$:

$$\int_{-\infty}^0l_{\mathbf{x'}}(t)dt=\sum_{i=1}^nx_i'\int_{-\infty}^0 \sum_{j=1}^m \beta_{ij}f_j(t) \ e^{\frac{t}{\sigma}}dt$$

$$\int_{-\infty}^0l_{\mathbf{x'}}(t)dt=\sum_{i=1}^n\sum_{j=1}^mx_i'\beta_{ij}\int_{-\infty}^0  f_j(t) \ e^{\frac{t}{\sigma}}dt$$

Usin $I_j=\int_{-\infty}^0  f_j(t) \ e^{\frac{t}{\sigma}}dt$, which can be precomputed:

$$\int_{-\infty}^0l_{\mathbf{x'}}(t)dt=\sum_{i=1}^n\sum_{j=1}^mx_i'\beta_{ij}I_j$$

Or simply:

$$\int_{-\infty}^0l_{\mathbf{x'}}(t)dt=\mathbf{x}'^T\mathbf{\beta}\mathbf{I_*}$$

## Numerator

In the same fashion, the numerator:

$$\int_{-\infty}^0\mathbf{x}(t)\mathbf{x'}\cdot\mathbf{x}(t) \ e^{\frac{t}{\sigma}}dt$$

Can be split into dot product elements:

$$\int_{-\infty}^0 \mathbf{x}(t)\sum_{j=1}^nx_j'x_j(t) \ e^{\frac{t}{\sigma}}dt$$

And each component $i$ of the final vector can be computed independently:

$$\int_{-\infty}^0 x_i(t)\sum_{j=1}^nx_j'x_j(t) \ e^{\frac{t}{\sigma}}dt$$

With a small rearrangement:

$$\sum_{j=1}^nx_j'\int_{-\infty}^0 x_i(t)x_j(t) \ e^{\frac{t}{\sigma}}dt$$

Using $x_i(t)=\beta_{i1}f_1(t)+ \dots + \beta_{im}f_m(t)$:

$$\sum_{j=1}^nx_j'\int_{-\infty}^0 (\sum_{k=1}^m\beta_{ik}f_k(t))(\sum_{k=1}^m\beta_{jk}f_k(t)) \ e^{\frac{t}{\sigma}}dt$$

$$\sum_{j=1}^nx_j'\int_{-\infty}^0 \sum_{k=1}^m\sum_{l=1}^m\beta_{ik}f_k(t)\beta_{jl}f_l(t) \ e^{\frac{t}{\sigma}}dt$$

$$\sum_{j=1}^n \sum_{k=1}^m\sum_{l=1}^mx_j'\beta_{ik}\beta_{jl}\int_{-\infty}^0f_k(t)f_l(t) \ e^{\frac{t}{\sigma}}dt$$

Using $I_{ij}=\int_{-\infty}^0  f_i(t)f_j(t) \ e^{\frac{t}{\sigma}}dt$, which can be precomputed:

$$\sum_{j=1}^n \sum_{k=1}^m\sum_{l=1}^mx_j'\beta_{ik}\beta_{jl}I_{kl}$$

Or simply:

$$\sum_{j=1}^n x_j'\beta_{i*}^T\mathbf{I_{**}}\beta_{j*}$$

$$\beta_{i*}^T\mathbf{I_{**}}\beta\mathbf{x}'$$

And the entire vector result is:

$$\beta^T\mathbf{I_{**}}\beta\mathbf{x}'$$

### Result

Combining numerator and denominator, the result is:

$$\hat{\mathbf{x}}=\frac{\beta^T\mathbf{I_{**}}\beta\mathbf{x}'}{\mathbf{x}'^T\mathbf{\beta}\mathbf{I_*}}$$

Where:

$$\mathbf{I_{**}}=\begin{bmatrix}
    \int_{-\infty}^0  f_0(t)f_0(t) \ e^{\frac{t}{\sigma}}dt&\dots&\int_{-\infty}^0  f_0(t)f_m(t) \ e^{\frac{t}{\sigma}}dt\\
    \vdots&\ddots&\vdots\\
    \int_{-\infty}^0  f_m(t)f_0(t) \ e^{\frac{t}{\sigma}}dt&\dots&\int_{-\infty}^0  f_m(t)f_m(t) \ e^{\frac{t}{\sigma}}dt\\
\end{bmatrix}$$

$$\mathbf{I_{*}}=\begin{bmatrix}
    \int_{-\infty}^0  f_0(t) \ e^{\frac{t}{\sigma}}dt\\
    \vdots\\
    \int_{-\infty}^0  f_m(t) \ e^{\frac{t}{\sigma}}dt\\
\end{bmatrix}$$

And $\beta$ is the coefficient matrix that satisfies:

$$\begin{bmatrix}
    x_1(t)\\
    \vdots \\
    x_n(t)
\end{bmatrix}=\begin{bmatrix}
    \beta_{11}&\dots&\beta_{1m}\\
    \vdots&\ddots&\vdots\\
    \beta_{n1}&\dots&\beta_{nm}\\
\end{bmatrix}\begin{bmatrix}
    f_1(t)\\
    \vdots \\
    f_m(t)
\end{bmatrix}$$