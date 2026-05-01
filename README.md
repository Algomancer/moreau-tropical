# moreau-tropical

a smooth, sparse, learnable max. one knob slides from max to mean. fused. idk think i read a paper similar to it a while back, needed it, couldn't find but defs exists probs

```python
y = MoreauTropicalKernel(in_features=512, out_features=1024, lam=1.0)(x)
```

## what is

each output is

$$y_n = \max_{p \in \Delta_D}\ \langle p,\ W_n + x\rangle - \tfrac{\lambda}{2}\|p\|^2$$

closed form: $p^* = (W_n + x - \tau)_+ / \lambda$. so $y_n$ is a weighted combo of the inputs, but only the ones above threshold $\tau$ count. everything else is exactly zero. real zeros, not $10^{-6}$.

## the knob

$\lambda$ controls:

- $\lambda \to 0$: hard tropical max, one feature wins per output
- $\lambda \to \infty$: smooths out toward a mean
- in between: sparse top-k-ish, where $k$ emerges from the data

$\lambda$ is differentiable. let the optimizer pick. envelope theorem one-shots the gradient, no kernel needed, freeeee.

## when maybe good

- you want attention but with actual zeros
- you want feature selection baked into the layer instead of bolt ons
- you're in tropical / max-plus / morphological territory and want a trainable primitive that respects the algebra.
- you're approximating something piecewise-linear and would rather express that directly

## when it's not

- you want a linear layer. use a linear layer.
- $D > 2048$. v0 holds a row in registers. bigger D needs tiling, i'll add maybe later

## how it runs

triton-fused forward, two triton backward kernels, envelope-theorem trick for $\partial L / \partial \lambda$ that needs no kernel because the math hands it to you.

girlie saves $(B, N)$ just $\tau$ and recomputes $p^*$ on the fly. same trade flashattention makes for softmax stats.

## use it

```python
layer = MoreauTropicalKernel(in_features=512, out_features=1024, lam=1.0).cuda()
y = layer(x)  # x: (B, 512) → y: (B, 1024)
```

make `lam` a parameter if you want it to train.
