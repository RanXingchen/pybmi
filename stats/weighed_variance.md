# Unbiased weighed variance

设 $X=[x_1, x_2, x_3, ..., x_n]$为代求数据，$W=[w_1, w_2, w_3, ..., w_n]$为权值。其中，$\bar{X}$为$X$的估计均值，$\mu$为$X$的真实均值，$S^2$为样本估计方差，$\sigma^2$为真实方差，$N$为$X$中样本的个数，$V=\sum_{i=1}^NW_i$，则：
$$
\begin{aligned}
    E[S^2]&=E[\frac{\sum_{i=1}^N(X_i-\bar{X})^2W_i}{\sum_{i=1}^NW_i}] \\
    &=E[\frac{1}{V}\sum_{i=1}^N(X_i-\bar{X})^2W_i] \\
    &=E[\frac{1}{V}\sum_{i=1}^N\{(X_i-\mu)-(\bar{X}-\mu)\}^2W_i] \\
    &=E[\frac{1}{V}\sum_{i=1}^N\{(X_i-\mu)^2-2(X_i-\mu)(\bar{X}-\mu)+(\bar{X}-\mu)^2\}W_i] \\
    &=E[\frac{1}{V}\sum_{i=1}^N(X_i-\mu)^2W_i-\frac{2}{V}(\bar{X}-\mu)\sum_{i=1}^N(X_i-\mu)W_i+\frac{1}{V}(\bar{X}-\mu)^2\sum_{i=1}^NW_i] \\
    &=E[\frac{1}{V}\sum_{i=1}^N(X_i-\mu)^2W_i-\frac{2}{V}(\bar{X}-\mu)\sum_{i=1}^N(X_i-\mu)W_i+(\bar{X}-\mu)^2] \\
    &=E[\frac{1}{V}\sum_{i=1}^N(X_i-\mu)^2W_i-\frac{2}{V}(\bar{X}-\mu)(\sum_{i=1}^NX_iW_i-\sum_{i=1}^N\mu{W_i})+(\bar{X}-\mu)^2] \\
    &=E[\frac{1}{V}\sum_{i=1}^N(X_i-\mu)^2W_i-\frac{2}{V}(\bar{X}-\mu)(V\bar{X}-\mu{V})+(\bar{X}-\mu)^2] \\
    &=E[\frac{1}{V}\sum_{i=1}^N(X_i-\mu)^2W_i-(\bar{X}-\mu)^2] \\
    &=E[\frac{1}{V}\sum_{i=1}^N(X_i-\mu)^2W_i]-E[(\bar{X}-\mu)^2] \\
    &=\sigma^2-E[(\bar{X}-\mu)^2]
\end{aligned}
$$

因为存在等式$E[\bar{X}]=E[\frac{\sum_{i=1}^NX_iW_i}{\sum_{i=1}^NW_i}]=\frac{1}{V}\sum_{i=1}^NE[X_iW_i]=\frac{1}{V}\sum_{i=1}^N\mu{W_i}=\mu$

所以，对于$E[(\bar{X}-\mu)^2]$：
$$
\begin{aligned}
    E[(\bar{X}-\mu)^2]&=E[(\bar{X}-E[\bar{X}])^2] \\
    &=var(\bar{X}) \\
    &=var(\frac{\sum_{i=1}^NX_iW_i}{\sum_{i=1}^NW_i}) \\
    &=\frac{1}{V^2}var(\sum_{i=1}^NX_iW_i) \\
    &=\frac{1}{V^2}\sum_{i=1}^Nvar(X_iW_i) \\
    &=\frac{1}{V^2}\sum_{i=1}^Nvar(X_i)W_i^2 \\
    &=\frac{1}{V^2}\sum_{i=1}^N\sigma^2W_i^2 \\
    &=\frac{\sigma^2}{V^2}\sum_{i=1}^NW_i^2
\end{aligned}
$$

令$V_2=\sum_{i=1}^NW_i^2$，则：
$$
\begin{aligned}
    E[(\bar{X}-\mu)^2]&=\frac{\sigma^2}{V^2}\sum_{i=1}^NW_i^2 \\
    &=\frac{V_2}{V^2}\sigma^2
\end{aligned}
$$

由此可得，
$$
\begin{aligned}
    E[S^2]&=E[\frac{1}{V}\sum_{i=1}^N(X_i-\bar{X})^2W_i] \\
    &=\sigma^2-E[(\bar{X}-\mu)^2] \\
    &=\sigma^2-\frac{V_2}{V^2}\sigma^2
\end{aligned}
$$

所以，
$$
\begin{aligned}
    \sigma^2&=E[\frac{\sum_{i=1}^N(X_i-\bar{X})^2W_i}{V(1-\frac{V_2}{V^2})}] \\
    &=E[\frac{\sum_{i=1}^N(X_i-\bar{X})^2W_i}{V-\frac{V_2}{V}}]
\end{aligned}
$$

证毕。
