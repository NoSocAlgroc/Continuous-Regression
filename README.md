# Continuous Regression

## Description

It is a well known fact that a linear regression finds the optimal $\boldsymbol{\beta}$ on 
the following equation:

$$ y_i=\beta_1x_{i1}+\beta_2x_{i2}+ \dots +\beta_px_{ip} $$ 

That given n by p predictor matrix $\boldsymbol{X}$ and n-dimensional target column vector 
$\boldsymbol{Y}$, minimize the mean squared difference between predicted and target values:

$$ S(\boldsymbol\beta)=\| \boldsymbol{Y-X\beta}\|^2  $$ 

This minimum is reached when:

$$ \boldsymbol\beta= (\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{Y}  $$ 


If instead we want to give each individual a weight in the MSQ, the error is:

$$ S(\boldsymbol\beta)=\| \boldsymbol{Y-X\beta}\|^T\boldsymbol{W}\| \boldsymbol{Y-X\beta}\|  $$ 

Where $\boldsymbol{W}$ is diagonal with trace $1$. Then the optimal coefficients are given by:

$$ \boldsymbol\beta= (\boldsymbol{X}^T\boldsymbol{W}\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{W}\boldsymbol{Y}  $$ 


But what if we had an infinite amount of individuals? Turns out that ordinary linear regression
with infinite samples can be used for a variety of applications.

## Local polynomial approximation

Let's look at an initial example of a practical use of the idea. Imagine we have a function
for which we want to obtain a polynomial approximation between a given range. There are already 
solution to this problem, like the Legendre approximations used in Legendre Memory Units, among
other applications.

There is, however, a much simpler approach to obtain polynomial approximations, formulating
the problem as a weighted linear regression. For now let us imagine we have an arbitrary
function $y(x)$ for which we want to obtain the polynomial approximation:

$$ y(x)=\beta_0+\beta_1x+\beta_2x^2+ \dots +\beta_px^p $$

That minimizes the weighted squared error according to a weight function $w(x)$:

$$ S(\boldsymbol\beta)=\int_{-\infty}^\infty w(x) \cdot (\beta_0+\beta_1x+\beta_2x^2+ \dots +\beta_px^p-y(x))^2 \ dx$$ 

If we want to minimize the squared error in an interval, one must simply define w as:

$$
w(x)=
\begin{cases}
    \frac{1}{r-l} &\quad\text{if x}\ge l, \text{x}\le r\\
    0 &\quad  \text{otherwise}\\

  \end{cases}
$$

But I liked using the Gaussian distribution pdf as weight as it allows for a softer transition between
sections of the function to approximate and not:

$$w(x)=\frac{1}{\sigma \sqrt{2 \pi }}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$$

At the end of the day any $w(x)$ works, but it's best it fulfills:

$$\int_{-\infty}^\infty w(x) \ dx=1$$

For now, instead of trying to optimize this continuous version let's attempt to generate $n$ samples and use
the aforementioned weighted regression to obtain the coefficient estimates. Doing this, the target $\boldsymbol{Y}$ 
ends up being:

$$\boldsymbol{Y}=
\begin{bmatrix}
    y(x_1)\\
    y(x_2)\\
    \vdots\\
    y(x_n)\\
\end{bmatrix}
$$

The data point matrix $\boldsymbol{X}$, for a rank $p$ approximation, is:$

$$\boldsymbol{X}=
\begin{bmatrix}
    1&x_1&x_1^2&\dots&x_1^p\\
    1&x_2&x_2^2&\dots&x_2^p\\
    1&x_3&x_3^2&\dots&x_3^p\\
    \vdots&\vdots&\vdots&\ddots&\vdots\\
    1&x_n&x_n^2&\dots&x_n^p\\
\end{bmatrix}
$$

And the weight matrix $\boldsymbol{M}$ is:

$$\boldsymbol{W}=
\begin{bmatrix}
    w(x_1)&0&\dots&0\\
    0&w(x_2)&\dots&0\\
    \vdots&\vdots&\ddots&0\\
    0&0&0&w(x_n)\\
\end{bmatrix}
$$

Of course, the size of all these matrices goes to infinity as does n, but we can actually obtain expressions for
$\boldsymbol{X}^T\boldsymbol{W}\boldsymbol{X}$ and $\boldsymbol{X}^T\boldsymbol{W}\boldsymbol{Y}$:

$$\boldsymbol{X}^T\boldsymbol{W}\boldsymbol{X}=
\begin{bmatrix}
    \sum_{i=1}^nw(x_i)x_i^0&\sum_{i=1}^nw(x_i)x_i^1&\sum_{i=1}^nw(x_i)x_i^2&\dots&\sum_{i=1}^nw(x_i)x_i^p\\
    \sum_{i=1}^nw(x_i)x_i^1&\sum_{i=1}^nw(x_i)x_i^2&\sum_{i=1}^nw(x_i)x_i^3&\dots&\sum_{i=1}^nw(x_i)x_i^{p+1}\\
    \sum_{i=1}^nw(x_i)x_i^2&\sum_{i=1}^nw(x_i)x_i^3&\sum_{i=1}^nw(x_i)x_i^4&\dots&\sum_{i=1}^nw(x_i)x_i^{p+2}\\
    \vdots&\vdots&\vdots&\ddots&\vdots\\
    \sum_{i=1}^nw(x_i)x_i^p&\sum_{i=1}^nw(x_i)x_i^{p+1}&\sum_{i=1}^nw(x_i)x_i^{p+2}&\dots&\sum_{i=1}^nw(x_i)x_i^{2p}\\
\end{bmatrix}
$$


$$\boldsymbol{X}^T\boldsymbol{W}\boldsymbol{Y}=
\begin{bmatrix}
    \sum_{i=1}^nw(x_i)x_i^0y(x_i)\\
    \sum_{i=1}^nw(x_i)x_i^1y(x_i)\\
    \sum_{i=1}^nw(x_i)x_i^2y(x_i)\\
    \vdots\\
    \sum_{i=1}^nw(x_i)x_i^py(x_i)\\
\end{bmatrix}
$$

Now, let $[x_1,x_2,\dots,x_n]$ be evenly distributed across the range $[l,r]$ and let $\Delta x=x_{i+1}-x_i$. 
If we multiply and divide both matrices by $\Delta x$ and introduce the numerator, all fields contain expressions 
akin to the following:

$$\sum_{i=1}^nf(x_i) \Delta x$$

And for large $n$, or equivalently for small $\Delta x$, the rectangle rule applies:

$$f(x_i) \Delta x \sim  \int_{x_i-\frac{1}{2}\Delta x}^{x_i+\frac{1}{2}\Delta x} f(x_i) \ dx$$

Therefore, the terms on the matrix become:

$$\sum_{i=1}^nw(x_i)x_i^k \Delta x \sim \sum_{i=1}^n \int_{x_i-\frac{1}{2}\Delta x}^{x_i+\frac{1}{2}\Delta x} w(x_i)x_i^k \ dx=\int_{l}^{r} w(x_i)x_i^k \ dx$$

Thus, as $n$, $-l$ and $r$ approach infinity:


$$\boldsymbol{X}^T\boldsymbol{W}\boldsymbol{X}=
\frac{1}{\Delta x}
\begin{bmatrix}
    \int_{-\infty}^{\infty} w(x)x^0 \ dx&\int_{-\infty}^{\infty} w(x)x^1 \ dx&\int_{-\infty}^{\infty} w(x)x^2 \ dx&\dots&\int_{-\infty}^{\infty} w(x)x^p \ dx\\
    \int_{-\infty}^{\infty} w(x)x^1 \ dx&\int_{-\infty}^{\infty} w(x)x^2 \ dx&\int_{-\infty}^{\infty} w(x)x^3 \ dx&\dots&\int_{-\infty}^{\infty} w(x)x^{p+1} \ dx\\
    \int_{-\infty}^{\infty} w(x)x^2 \ dx&\int_{-\infty}^{\infty} w(x)x^3 \ dx&\int_{-\infty}^{\infty} w(x)x^4 \ dx&\dots&\int_{-\infty}^{\infty} w(x)x^{p+2} \ dx\\
    \vdots&\vdots&\vdots&\ddots&\vdots\\
    \int_{-\infty}^{\infty} w(x)x^p \ dx&\int_{-\infty}^{\infty} w(x)x^{p+1} \ dx&\int_{-\infty}^{\infty} w(x)x^{p+2} \ dx&\dots&\int_{-\infty}^{\infty} w(x)x^{2p} \ dx\\
\end{bmatrix}
$$

$$\boldsymbol{X}^T\boldsymbol{W}\boldsymbol{Y}=
\frac{1}{\Delta x}
\begin{bmatrix}
    \int_{-\infty}^{\infty} w(x)x^0y(x)\ dx\\
    \int_{-\infty}^{\infty} w(x)x^1y(x) \ dx\\
    \int_{-\infty}^{\infty} w(x)x^2y(x) \ dx\\
    \vdots\\
    \int_{-\infty}^{\infty} w(x)x^py(x) \ dx\\
\end{bmatrix}
$$

And if we substitute them in the original coefficient estimation equation, we find out that the $\frac{1}{\Delta x}$ terms disappear:

$$\boldsymbol\beta= 
\begin{bmatrix}
    \int_{-\infty}^{\infty} w(x)x^0 \ dx&\int_{-\infty}^{\infty} w(x)x^1 \ dx&\int_{-\infty}^{\infty} w(x)x^2 \ dx&\dots&\int_{-\infty}^{\infty} w(x)x^p \ dx\\
    \int_{-\infty}^{\infty} w(x)x^1 \ dx&\int_{-\infty}^{\infty} w(x)x^2 \ dx&\int_{-\infty}^{\infty} w(x)x^3 \ dx&\dots&\int_{-\infty}^{\infty} w(x)x^{p+1} \ dx\\
    \int_{-\infty}^{\infty} w(x)x^2 \ dx&\int_{-\infty}^{\infty} w(x)x^3 \ dx&\int_{-\infty}^{\infty} w(x)x^4 \ dx&\dots&\int_{-\infty}^{\infty} w(x)x^{p+2} \ dx\\
    \vdots&\vdots&\vdots&\ddots&\vdots\\
    \int_{-\infty}^{\infty} w(x)x^p \ dx&\int_{-\infty}^{\infty} w(x)x^{p+1} \ dx&\int_{-\infty}^{\infty} w(x)x^{p+2} \ dx&\dots&\int_{-\infty}^{\infty} w(x)x^{2p} \ dx\\
\end{bmatrix}^{-1}
\cdot
\begin{bmatrix}
    \int_{-\infty}^{\infty} w(x)x^0y(x)\ dx\\
    \int_{-\infty}^{\infty} w(x)x^1y(x) \ dx\\
    \int_{-\infty}^{\infty} w(x)x^2y(x) \ dx\\
    \vdots\\
    \int_{-\infty}^{\infty} w(x)x^py(x) \ dx\\
\end{bmatrix}
$$
