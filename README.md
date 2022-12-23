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
    \frac{1}{r-l} &\quad\text{if}\ x \in [l,r]\\
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


Interestingly, this matrix can be precomputed, meaning the estimates can be calculated as a linear mapping of the right vector.
As it only needs to be computed once, each element can be numerically integrated depending on the weight function, but for some
of them there are solutions:

**Uniform weights in range**

If the weight function has the form:

$$
w(x)=
\begin{cases}
    \frac{1}{r-l} &\quad\text{if}\ x \in [l,r]\\
    0 &\quad  \text{otherwise}\\
  \end{cases}
$$

Each term can be computed directly:

$$\int_{-\infty}^{\infty} w(x)x^k \ dx=\int_{l}^{r} x^k \ dx=\left[\frac{1}{k+1}x^{k+1}\right]_l^r=\frac{r^{k+1}-l^{k+1}}{k+1}$$

Not that it matters much, but since all of them have to be computed up to a certain $k$, a faster recursive formula can be found:

$$\frac{r^{k+1}-l^{k+1}}{k+1}=\frac{(r^k-l^k)\cdot(r+l)}{k+1}=\frac{r+l}{k+1}k\frac{r^l-l^k}{k} $$

**Gaussian weights**

If the weight has the form:

$$w(x)=\frac{1}{\sigma \sqrt{2 \pi }}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$$

Without loss of generality, if $\mu =0$ then it follows that:

$$\int xw(x)=-\sigma^2 w(x)$$


With this information, a recurrence can be established between terms:

$$\int_{-\infty}^{\infty} w(x)x^k \ dx$$
$$\int_{-\infty}^{\infty} (xw(x)) \cdot(x^{k-1}) \ dx$$
$$\left[-\sigma^2w(x)x^{k-1} \right]_{-\infty}^{\infty} +(k-1)\cdot \sigma^2 \int_{-\infty}^{\infty} w(x)x^{k-2} \ dx$$

Therefore, we can establish the recurrence equation as:

$$\int_{-\infty}^{\infty} w(x)x^k \ dx=(k-1)\cdot \sigma^2 \int_{-\infty}^{\infty} w(x)x^{k-2} \ dx$$

With base cases:


$$\int_{-\infty}^{\infty} w(x)x^0 \ dx=1$$
$$\int_{-\infty}^{\infty} w(x)x^1 \ dx=0$$

Which leads to the simpler formula:

$$
\int_{-\infty}^{\infty} w(x)x^k \ dx=
\begin{cases}
    (k-1)!!\sigma^k &\quad\text{if $k$ even}\\
    0 &\quad  \text{otherwise}\\
  \end{cases}
$$

Allowing us to precompute the term $\Delta x\boldsymbol{X}^T\boldsymbol{W}\boldsymbol{X}$.

Thus, we can obtain the polynomial coefficients as a linear combination of $\Delta x\boldsymbol{X}^T\boldsymbol{W}\boldsymbol{Y}$.
This can be used in different ways:

### Direct appriximation

If we have enough information about $y(x)$ to obtain analytical expressions or approximations for:

$$\boldsymbol{X}^T\boldsymbol{W}\boldsymbol{Y}\Delta x=
\begin{bmatrix}
    \int_{-\infty}^{\infty} w(x)x^0y(x)\ dx\\
    \int_{-\infty}^{\infty} w(x)x^1y(x) \ dx\\
    \int_{-\infty}^{\infty} w(x)x^2y(x) \ dx\\
    \vdots\\
    \int_{-\infty}^{\infty} w(x)x^py(x) \ dx\\
\end{bmatrix}
$$

Then the optimal coefficients can be found.

### Online polynomial regression

Let's imagine we do not know what our target function $y(x)$ looks like, but instead we obtain a series of $(x,y)$ tuples of 
points belonging to the function. Of course we could store them all and perform the usual regression, but thanks to the previous
results we can actually maintain only our $p+1$ coefficients in memory and update them with every $(x,y)$, or a set of them, in 
order to improve the approximation.

Let's assume we have a current approximation $\boldsymbol{\beta}$ of our function, which does define a function on its own:

$$ y(x)'=\beta_0+\beta_1x+\beta_2x^2+ \dots +\beta_px^p $$

So, after recieving a new $(x_i,y_i)$, we can attempt to find coefficients that approximate this very same function, but changing the value at $x_i$:

$$
y(x)=
\begin{cases}
    s(y_i,y(x)',x,x_i,\lambda) &\quad  \text{if $x \in [x_i-\lambda]$}\\
    y(x)' &\quad\text{if $k$ even}\\
  \end{cases}
$$

Where $s(y_i,y(x)',x,\lambda)$ performs a transition from the continuous function $y(x)'$ to the constant $y_i$ in the range $[x_i-\lambda,x_i+\lambda]$. The simplest
is to just use the constant:

$$s(y_i,y(x)',x,x_i,\lambda)=y_i$$

Or some linear interpolation:

$$s(y_i,y(x)',x,x_i,\lambda)=y(x)'+(y_i-y(x)')\cdot\frac{\lvert x-x_i\rvert}{\lambda}$$

Or any you can think of, what matters is to obtain a new function that is equal to the already approximated with a local change 
at the newly recieved $(x_i,y_i)$.

If we want to find the difference between the previous $\boldsymbol{\beta}$ and the ones that optimally approximate the new function,
turns out that we only need the difference between the previous $\boldsymbol{X}^T\boldsymbol{W}\boldsymbol{Y}\Delta x$ and the new one:

$$\Delta (\boldsymbol{X}^T\boldsymbol{W}\boldsymbol{Y}\Delta x)=
\begin{bmatrix}
    \int_{x_i-\lambda}^{x_i+\lambda} w(x)x^0(s(y_i,y(x)',x,x_i,\lambda)-y(x)')\ dx\\
    \int_{x_i-\lambda}^{x_i+\lambda} w(x)x^1(s(y_i,y(x)',x,x_i,\lambda)-y(x)')\ dx\\
    \int_{x_i-\lambda}^{x_i+\lambda} w(x)x^2(s(y_i,y(x)',x,x_i,\lambda)-y(x)')\ dx\\
    \vdots\\
    \int_{x_i-\lambda}^{x_i+\lambda} w(x)x^p(s(y_i,y(x)',x,x_i,\lambda)-y(x)')\ dx\\
\end{bmatrix}
$$

Which can be approximated in many ways. For example, the simplest way is by the rectangle rule for small $\lambda$:

$$\Delta (\boldsymbol{X}^T\boldsymbol{W}\boldsymbol{Y}\Delta x)=
\begin{bmatrix}
    2\lambda w(x_i)x_i^0(y_i-y(x)')\ dx\\
    2\lambda w(x_i)x_i^1(y_i-y(x)')\ dx\\
    2\lambda w(x_i)x_i^2(y_i-y(x)')\ dx\\
    \vdots\\
    2\lambda w(x_i)x_i^p(y_i-y(x)')\ dx\\
\end{bmatrix}
$$

Which allows us to find $\Delta \boldsymbol{\beta}$:

$$\Delta \boldsymbol{\beta} =
(\boldsymbol{X}^T\boldsymbol{W}\boldsymbol{X}\Delta x)^{-1}
\cdot
\Delta (\boldsymbol{X}^T\boldsymbol{W}\boldsymbol{Y}\Delta x)
$$

And thus update our coefficients to approximate the previous function while being closer to the given point as well.


## Generalized online regression

Let us now consider the case where instead of using a polynomial basis we use arbitrary functions to model the response function:

$$ y(x)=\beta_1f_1(x)+\beta_2f_2(x)+ \dots +\beta_pf_p(x) $$

Which makes the matrix be:

$$\boldsymbol{X}^T\boldsymbol{W}\boldsymbol{X}\Delta x=
\begin{bmatrix}
    \int_{-\infty}^{\infty} w(x)f_1(x)f_1(x) \ dx&\int_{-\infty}^{\infty} w(x)f_1(x)f_2(x) \ dx&\int_{-\infty}^{\infty} w(x)f_1(x)f_3(x) \ dx&\dots&\int_{-\infty}^{\infty} w(x)f_1(x)f_p(x) \ dx\\
    \int_{-\infty}^{\infty} w(x)f_2(x)f_1(x) \ dx&\int_{-\infty}^{\infty} w(x)f_2(x)f_2(x) \ dx&\int_{-\infty}^{\infty} w(x)f_2(x)f_3(x) \ dx&\dots&\int_{-\infty}^{\infty} w(x)f_2(x)f_p(x) \ dx\\
    \int_{-\infty}^{\infty} w(x)f_3(x)f_1(x) \ dx&\int_{-\infty}^{\infty} w(x)f_3(x)f_2(x) \ dx&\int_{-\infty}^{\infty} w(x)f_3(x)f_3(x) \ dx&\dots&\int_{-\infty}^{\infty} w(x)f_3(x)f_p(x) \ dx\\
    \vdots&\vdots&\vdots&\ddots&\vdots\\
    \int_{-\infty}^{\infty} w(x)f_p(x)f_1(x) \ dx&\int_{-\infty}^{\infty} w(x)f_p(x)f_2(x) \ dx&\int_{-\infty}^{\infty} w(x)f_p(x)f_3(x) \ dx&\dots&\int_{-\infty}^{\infty} w(x)f_p(x)f_p(x) \ dx\\
\end{bmatrix}
$$

These terms can be numerically integrated, but for some combinations of $w(x)$, $f_j(x)$, $f_i(x)$ analytical expressions may be found.
For now, I will use the following function classes:

**Polynomial**

These are simply powers of $x$:

$$f(x)=x^k$$

**Trig**

Sine and cosine with some frequency:

$$f(x)=cos(kx)$$
$$f(x)=sin(kx)$$


### Matrix elements with Gaussian weight

Following is described how to obtain the analytical expressions for the integral of the product of any pair of function
classes with the Gaussian weight:

#### Poly-Poly

This one has already been described before:

$$
\int_{-\infty}^{\infty} w(x)x^{k_1}x^{k_2} \ dx=
\begin{cases}
    (k_1+k_2-1)!!\sigma^{k_1+k_2} &\quad\text{if $k_1+k_2$ even}\\
    0 &\quad  \text{otherwise}\\
  \end{cases}
$$

#### Trig-Poly

**Cosine:**

Recursion:

$$ \int_{-\infty}^{\infty} w(x)x^{k_1}cos(k_2x) \ dx$$

$$ \int_{-\infty}^{\infty} (w(x)x)\ (x^{k_1-1}cos(k_2x)) \ dx$$

$$(k_1-1)\cdot \sigma^2 \int_{-\infty}^{\infty} w(x)x^{k_1-2}cos(k_2x) \ dx-k_2\cdot \sigma^2 \int_{-\infty}^{\infty} w(x)x^{k_1-1}sin(k_2x) \ dx$$


Base case:

$$ \int_{-\infty}^{\infty} w(x)x^{0}cos(k_2x) \ dx$$

Let:

$$ I(k_2)=\int_{-\infty}^{\infty} w(x)cos(k_2x) \ dx$$

Then:

$$ I'(k_2)=-\int_{-\infty}^{\infty} xw(x)sin(k_2x) \ dx$$

$$ I'(k_2)=+\left[ \sigma^2w(x)sin(k_2x)\right]_{-\infty}^{\infty} -k_2\sigma^2\int_{-\infty}^{\infty} w(x)cos(k_2x) \ dx$$

$$ I'(k_2)=-k_2\sigma^2I(k_2)$$

Which has solution for:

$$I(0)=1$$

$$I(k_2)=e^{-k_2^2\frac{\sigma^2}{2}}$$

**Sine:**

Recursion:

$$ \int_{-\infty}^{\infty} w(x)x^{k_1}sin(k_2x) \ dx$$

$$ \int_{-\infty}^{\infty} (w(x)x)\ (x^{k_1-1}sin(k_2x)) \ dx$$

$$(k_1-1)\cdot \sigma^2 \int_{-\infty}^{\infty} w(x)x^{k_1-2}sin(k_2x) \ dx+k_2\cdot \sigma^2 \int_{-\infty}^{\infty} w(x)x^{k_1-1}cos(k_2x) \ dx$$

Base case:

$$ \int_{-\infty}^{\infty} w(x)x^{0}sin(k_2x) \ dx=0$$

Overall, using the short forms for any $k_2$:

$$C_{k_1}= \int_{-\infty}^{\infty} w(x)x^{k}cos(k_2x) \ dx$$

$$S_{k_1}= \int_{-\infty}^{\infty} w(x)x^{k}sin(k_2x) \ dx$$

The recursive rules can be rewritten:

$$C_{k_1}= +k_2\sigma^2S_{k_1-1}+(k_1-1)\sigma^2C_{k_1-2}$$

$$S_{k_1}= -k_2\sigma^2C_{k_1-1}+(k_1-1)\sigma^2S_{k_1-2}$$

Which allows it to be rewritten as a linear combination:

$$\begin{bmatrix}
    C_{i-1}\\
    S_{i-1}\\
    C_{i}\\
    S_{i}\\
\end{bmatrix}=
\begin{bmatrix}
    0&0&1&0\\
    0&0&0&1\\
    (i-1)\sigma^2&0&1&k_2\sigma^2\\
    0&(i-1)\sigma^2&-k_2\sigma^2&0\\
\end{bmatrix}\cdot
\begin{bmatrix}
    C_{i-2}\\
    S_{i-2}\\
    C_{i-1}\\
    S_{i-1}\\
\end{bmatrix}$$

With general expansion to the Bessel polynomial coefficients:

$$\begin{bmatrix}
    C_{i}\\
    S_{i}\\
\end{bmatrix}=
\begin{cases}
    \begin{bmatrix}
        C_1\sigma^i \sum_{j=0}^{n/2}\sigma^{2j} k_2^{2j}\Theta_{i-j,j}\\
        0\\
    \end{bmatrix}
    &
    \text{If $i$ even}\\
    \begin{bmatrix}
    0\\
    C_1\sigma^{i+1}k_2 \sum_{j=0}^{(n-1)/2}\sigma^{2j} k_2^{2j}\Theta_{i-j,j}\\
    \end{bmatrix}
&
\text{If $i$ odd}
\end{cases}$$

Where $\Theta_{i,j}$ is the $j^{th}$ coefficient of the $i^{th}$ reverse Bessel polynomial:

$$\Theta_{i,j}=\frac{(i+j)!}{(i-j)!j!2^j}$$


#### Trig-Trig

