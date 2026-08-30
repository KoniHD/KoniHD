---
title: 'Distributed Neural Network Training from First Principles with MPI and CUDA'
date: 2026-06-25
modified: 2026-08-26
permalink: /posts/2026/distributed-ml-mpi-cuda/
license: 'CC BY-NC 4.0'
license_url: https://creativecommons.org/licenses/by-nc/4.0/
tags:
  - Machine Learning
  - Artificial Intelligence
  - CUDA
  - MPI
published: true
sitemap: true
toc: true
toc_sidebar: true
human_authored: true
translated: false # TODO Needs to be translated before publishing!
read_time_stop_at: '<!-- read-time-stop -->'
---

{% include toc levels="1" sticky=true sidebar=true %}

<div class="article__body" markdown="1">

<div class="exercise-disclaimer">
  <i class="fas fa-graduation-cap" aria-hidden="true"></i>
  <span>The full technical implementation of this blog is part of an exercise I came up through my tutoring in <a href="https://www.ce.cit.tum.de/caps/lehre/sommersemester-26/vorlesungen/parallel-programming/" target="_blank" rel="noopener noreferrer" referrerpolicy="no-referrer">IN2147 - Parallel Programming</a>. If you'd rather read about the techincal aspects in an exercise form visit this <a href="https://home.cit.tum.de/~zeko/" target="_blank" rel="noopener noreferrer" referrerpolicy="no-referrer">link</a> or attend the course 😃.</span>
</div>

# Introduction

Modern machine learning models such as LLMs or VLAs often require huge effort to be trained. Partially due to the model's own size but also partially due to the shear amount of training data that has to be iterated over. For a while now it has been established that the best way to train AI is to use accelerators such as GPUs, mostly from Nvidia, TPUs or occasionaly other ASICS. While there has been huge progress and these accelerators have become more powerful, it has become clear that a single accelerator is not able to match the demand needed by these new models. This has led to training no longer being viable on a single accelerator. Instead work has to be put into parallelizing and distributing training and inference accross multiple accelerators.

As one of the most prominent machine learning frame works, PyTorch currently offers four built in distribution strategies trough its [`torch.distributed`](https://docs.pytorch.org/tutorials/beginner/dist_overview.html) package. Similarly Tensorflow offers an option to [train on multiple GPUs](https://www.tensorflow.org/guide/keras/distributed_training).
This goes to show that the multi-accelerator training is very common.

In this blog post I want to demonstrate, using a rudimentary example, on how to train a multi-layer perceptron (MLP) on the MNIST dataset. For this I will code the training and inference loop using CUDA and distribute the training according to the [Distributed Data Parallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) strategy using MPI. Using this strategy allows me to easily demonstrate one important aspect of distributing training accross multplie accelerators: overlapping communication with computation.

*Note:* A two layer MLP is by no means close to exhasterbating the resources of any common GPU in use today, especially if the model weights and cached tensors combined are approximately 1 MByte.

# Basic Mathematics

In order to later write the kernels in CUDA it helps me to start by first defining the desired model architecture
and then writing down the mathematical operations.
Especially the second step is helpful since some operations can be fused together,
reducing the actual number operations needed. Only after that do I start thinking about the actual software implementation.

## The model -- Architecture

MNIST is a dataset that has been worked on countless times with models outperforming humans for a while now.
Hence the goal of this implementation shall not be to improve any scores but to provide a simple architecture that has a verifiable training progress.

That said I opted for a simple two-layer MLP with: `input_dim = 784`, `hidden_dim = 256`, `output_dim = 10` (number of digits 0 - 9).
The activation function chosen is ReLU.
The final layer is softmax, the loss is  mean cross-entropy loss (for simplicity only called cross-entropy loss) which can be fused into the softmax as we'll see once we write down the mathematical operations.
And finally I chose SGD as an optimizer which requires less tensors to be cached than a 2nd momentum optimizer such as the widely used Adam.

<figure>
  {% include mnist-architecture.svg %}
  <figcaption>Model architecture (at inference): Two-layer perctron network with <code>input_dim = 784</code>, <code>hidden_dim = 256</code> after ReLU activation and <code>output_dim = 10</code> to prediction one of 10 digit classes.</figcaption>
</figure>

## The model -- Maths

To describe the mathematical operations we use indices: $$i \in 1, \ldots, 784$$ with $$\mathbf{784}$$ being the number of pixels per image – 
each pixel has a value $$v \in [0, 1]$$; $$j \in 1, \ldots, 256$$ with $$256$$  being the number of hidden dimensions; and finally
$$k \in 1, ..., C$$ with $$\mathbf{C = 10}$$ being the number of classes.

### Forward pass

With the indices defined we can write down the per sample ($$s$$) forward pass of the model starting with two linear layers.

**1st Hidden Layer**

$$
H_{j}^{(s)} = X_{i}^{(s)} W_{ij}^{(1)} + b_{j}^{(1)}, \: \text{X: flattened 1D input image}
$$

**ReLU Activation**

$$
\hat{H}_{j}^{(s)} = \max \left( 0, H_{j}^{(s)} \right)
$$

After this the 2nd linear layer is similarly with the exception that no activation is needed since it would stand in the way of the softmax.

**Logits**

$$
Z_{k}^{(s)} = \hat{H}_{j}^{(s)} W_{jk}^{(2)} + b_{k}^{(2)}
$$

Now after having computed the logits we already know the model's prediction since Softmax is a strictly monotoneaus function.
This means the final layer, softmax, does not change for which index $$k$$, $$Z_{k}^{(s)}$$ is the highest. Therefore at inference we get our prediction simply through:

$$
\hat{y}^{(s)} = \arg \max_{k} Z_{k}^{(s)}
$$

On the other hand during training this prediction is of no direct interest but rather important for e.g. logs and hyper-parameter training.
Instead we are interested in applying softmax to normalize the predictions, simultanously getting a sort of certainty in the prediction, and finally obtaining the loss of our prediction.

**Softmax**

$$
p_{k}^{(s)} = \frac{e^{Z_{k}^{(s)} - a}}{\sum_{l=1}^{C} e^{Z_{l}^{(s)} - a}}
$$

Here the factor $$a^{(s)} = \max_{k} Z_{k}^{(s)}$$ is for numerical stability which ensures the denominator doesn't explode by capping the exponent in the numerator and denomiator $$\leq 1$$.

**Cross Entropy Loss**

$$
L^{(s)} = -\sum_{k=1}^{C} y_{k}^{(s)} \log p_{k}^{(s)}
$$

Again it helps to pay attention to the details since $$y_{k}$$ is one-hot encoded we can simplify this formula even further. (One-hot encoding: $$y_{k} = 1$$ for the correct class $$k^{*}$$ and $$y_{k \neq k^{*}} = 0$$).

$$
L^{(s)} = -y_{k^{*}}^{(s)} \log p_{k^{*}}^{(s)} = -\log p_{k^{*}}^{(s)} = -\left( Z_{k^{*}}^{(s)} - a^{(s)} - \log \left( \sum_{l=1}^{C} e^{Z_{l}^{(s)} - a^{(s)}} \right) \right)
$$

Where we inserted the definition of Softmax in the last part. Again, the actuall value of loss $$L^{(s)}$$ is not directly relevant to the training but quite useful for hyperparameter training. For training we are only interested in its gradient w.r.t. $$Z_{k}^{(s)}$$, 

### Backward Pass

As stated before for the actual training we need the gradient of the cross entropy loss $$\frac{\partial L^{(s)}}{\partial p_{k}^{(s)}}$$ or when fusing the loss with softmax $$\frac{\partial L^{(s)}}{\partial Z_{k}^{(s)}}$$. Perhaps somewhat counter intuitive this fused gradient actually has a simpler form as derived neatly in [Paras Dahal's blog](https://parasdahal.com/softmax-crossentropy):

$$
\frac{\partial L^{(s)}}{\partial Z_{k}^{(s)}} = p_{k}^{(s)} - y_{k}^{(s)}
$$

Finishing up the backward pass using this simple gradient however it is important to now take into account the data parallelism. So far every operation has only been performed on a single sample, indicated by the superscript $$(s)$$. This implies multiple things. First, it means we can split different samples accross $$R$$ independent devices since the consequent steps only rely on results of the same sample. Secondly, it also means we can perform all of the previous operations on multiple samples at the same time. To do so we change the input vector $$X_{i}$$ into an input matrix $$X_{i} \in \mathbb{R}^{N_{\text{local}} \times i}$$ where $$N_{\text{local}}$$ is the number of samples to process in parallel (on one device).

This becomes important since we initially specified our loss to be a mean cross-entropy loss which requires:

$$
L_{\text{total}} = \frac{1}{N_{\text{total}}} \sum_{n = 1}^{N_{\text{total}}} L^{(s)} = \frac{1}{R} \sum_{r = 1}^{R} \frac{1}{N_{\text{local}}} \sum_{n = 1}^{N_{\text{local}}}
$$

Here $$N_{\text{total}} = R \cdot N_{\text{local}}$$ constant and independent of model parameters. In practise again we want to use the local normalization factor $$\frac{1}{N_{\text{local}}}$$ to prevent the gradients from growing too large. Which is why we incorporate the normalization directly in the per-sample loss:

$$
G_{k}^{(s)} = \frac{1}{N_{\text{local}}} p_{k}^{(s)} - y_{k}^{(s)}
$$

Continuing the backward pass from here on we however need to take care of this double parallelism (local and accross devices). Say we distribute the work accross $$R$$ devices, which in this example are `MPI Processes`, we need to perform mean operation: $$\frac{1}{R} \sum_{r = 1}^{R}$$ after having successfully communicated every device's loss to all others. On an implementation layer this means henceforth every $$\frac{1}{R} \sum_{r = 1}^{R}$$ is equivalent to an `MPI_Allreduce` with `MPI_SUM`.

**2nd Layer -- Gradient**

In total we are interested in three gradients from this layer. Directly important for the weight updates are: $$\frac{\partial L}{\partial W_{jk}^{(2)}}$$ and $$\frac{\partial L}{\partial b_{k}^{(2)}}$$. Additionally, for the gradient flow to the 1st layer we also need: $$\frac{\partial L}{\partial \hat{H}_{j}^{(2)}} = D_{j}^{H, (s)}$$. Using the chain rule we can compute all of these gradients:

$$
\frac{\partial L}{\partial W_{jk}^{(2)}}
    = \frac{\partial L}{\partial Z_{k}}\,\frac{\partial Z_{k}}{\partial W_{jk}^{(2)}}
    = \frac{1}{R} \sum_{r=1}^{R} \sum_{s=1}^{N_{\text{local}}} \hat{H}_{j}^{(s)}\, G_{k}^{(s)} 
  \qquad
\frac{\partial L}{\partial b_{k}^{(2)}}
    = \frac{\partial L}{\partial Z_{k}}\,\frac{\partial Z_{k}}{\partial b_{k}^{(2)}}
    = \frac{1}{R} \sum_{r=1}^{R} \sum_{s=1}^{N_{\text{local}}} G_{k}^{(s)}
$$

And similarly we can compute $$D_{j}^{H, (s)}$$ where we have to keep in mind that this operation is again *per sample*.

$$
D_{j}^{H,(s)}
    = \frac{\partial L}{\partial \hat{H}_{j}^{(s)}}
    = \frac{\partial L}{\partial Z_{k}}\,\frac{\partial Z_{k}}{\partial \hat{H}_{j}^{(s)}}
    = W_{jk}^{(2)}\, G_{k}^{(s)}
$$

**1st Layer -- Gradient**

Since there is a ReLU activation between the 1st and 2nd linear layer we first need to consider its derivative:

$$
D_{j}^{(s)}
    = \frac{\partial L}{\partial H_{j}^{(s)}}
    = \frac{\partial L}{\partial \hat{H}_{j}^{(s)}}
      \frac{\partial \hat{H}_{j}^{(s)}}{\partial H_{j}^{(s)}}
    = D_{j}^{H,(s)}\, \frac{\partial \hat{H}_{j}^{(s)}}{\partial H_{j}^{(s)}}
    = D_{j}^{H,(s)} \cdot \mathbb{1}\!\left[H_{j}^{(s)} > 0\right]
$$

And can now finally compute the gradients w.r.t. the weights:

$$
\frac{\partial L}{\partial W_{ij}^{(1)}}
    = \frac{\partial L}{\partial H_{j}^{(s)}}\,\frac{\partial H_{j}^{(s)}}{\partial W_{ij}^{(1)}}
    = \frac{1}{R} \sum_{r=1}^{R} \sum_{s=1}^{N_{\text{local}}} X_{i}^{(s)}\, D_{j}^{(s)}
  \qquad
\frac{\partial L}{\partial b_{j}^{(1)}}
    = \frac{\partial L}{\partial H_{j}^{(s)}}\,\frac{\partial H_{j}^{(s)}}{\partial b_{j}^{(1)}}
    = \frac{1}{R} \sum_{r=1}^{R} \sum_{s=1}^{N_{\text{local}}} D_{j}^{(s)}
$$

### Optimizer Step

And finally we can update each weight according to SGD.

$$
\theta = \theta - \eta \nabla_{\theta}L, \quad \theta \in \left\{ W^{(1)}, W^{(2)}, b^{(1)}, b^{(2)} \right\}
$$

# Kernel implementation -- CUDA

After having written down the math so rigorously we observe that we can implement the whole neural network using only three CUDA kernel, provided we use C++ templates. One (templated) kernel can be used for the forward linear layer, one kernel for the fused softmax + cross-entropy operations and one (templated) kernel for the backward pass through the linear layer.

## Forward pass Linear Layer -- CUDA

**Input:** One or more samples i.e. vector or matrix respectively ($$X_{i} \in \mathbb{R}^{i \times N_{\text{local}}}$$ or $$\hat{H}_{j}$$).  
***Optional*** **ReLU:** Based on a template parameter `template <bool FuseReLU>` ReLU is applied before the output in order to not write to device memory and reloading unnecessarily.  
**Output:** A output vector ($$\hat{H}_{j}$$ hidden layer or logits $$Z_{k}$$).

This kernel is launched twice. Once for the first linear layer with `FuseReLU = true` and once for the second linear layer with `FuseReLU = false`.

## Fused Softmax + Cross-Entropy -- CUDA

**Input:** Logits (vector: $$Z_{k}$$).  
**Output:** Derivative $$\frac{\partial L}{\partial Z_{k}} \: (= G_{k})$$.  
**Additional:** The loss and correctness of the prediction as a byproduct for logging and hyperparameter tuning. ($$L^{(s)}$$: loss per sample, and $$C^{(s)}$$: correctness per sample).

**Inference Kernel:** At inference only the following is executed:

$$
y^{(s)} = \arg \max_{k} Z_{k}^{(s)}
$$

## Backward pass Linear Layer -- CUDA

**Input:** The upstream gradient ($$G_{k}$$ loss of Softmax or $$D_{j}^{H}$$).  
**Output:** Gradients w.r.t. Weight matrix, and bias ($$\frac{\partial L}{\partial W}$$, and $$\frac{\partial L}{\partial b}$$).  
***Optional*** **Gradient w.r.t. input:** This only happens in the second layer since the gradient
w.r.t. the model input is irrelevant ($$\frac{\partial L}{\partial \hat{H}_{j}}$$). This is indicated by another template parameter `<bool InputGrad>`.  
***Optional*** **ReLU gradient:** In case of the first layer the ReLU gradient is also fused into the kernel through a template parameter `<bool FuseReLU>` such that $$\frac{\partial L}{\partial H_{j}}$$ is computed before com puting the other gradients($$\frac{\partial L}{\partial H_{j}}$$ instead of $$\frac{\partial L}{\partial \hat{H}_{j}}$$).

Like the forward pass, his kernel is launched *twice*. Once for the second layer with `InputGrad = true` and `FuseReLU = false` and once for the first layer with `InputGrad = false` and `FuseReLU = true`.

# Efficient Interleaving of Communications

<figure>
  {% include mnist-pipeline.svg %}
  <figcaption>Every operations is displayed as a seperate block. One can observe that the blue forward boxes are operations that can be execuded without communication with other devices whereas the green optimizer step operation needs the gradients from the orange backward passes of the linear layers. <br>
  <b>Note:</b> The arrows labelled with gradients are communications that can overlap with subsequent operation blocks.</figcaption>
</figure>

After having written down the mathematical operations of the model, including some optimizations, and fixing the kernel, there is only one more aspect I would like to draw attention to. As stated in the beginning this example is not only useful to demonstrate the DDP strategy using MPI but also how to efficiently overlap communications with computation.

Looking at how the [optimizer step](#optimizer-step) is performed we can see it only needs the parameters $$\theta$$ and the respective loss $$\nabla_{\theta} L$$. Meaning we don't have to wait for all gradients $$\nabla_{\theta} L$$ to become available to perform one optimizer step but instead can do so as soon as the gradient becomes available. Therefore once $$\frac{\partial L}{\partial W_{jk}^{(2)}} = \sum_{s = 1}^{N_{\text{local}}} \hat{H}_{j}^{(s)} G_{k}^{(s)}$$ and $$\frac{\partial L}{\partial b_{k}^{(2)}} = \sum_{s = 1}^{N_{\text{local}}} G_{k}^{(s)}$$ are computed per device locally we can imediately start a **non-blocking** `MPI_Iallreduce` (i.e. $$\frac{1}{R} \sum_{r = 1}^{R}$$) operation which can overlap with the computation of the gradients of the 1st layer.

# Conclusions

This article showcased a simple hands-on example to distribute the training of a machine learning model accross multiple devices. Most importantly it demonstrates efficiency optimisations on a mathematical level which then translate to simpler coding implementations.

Further this article uses the Distributed Data Parallel strategy to showcase simple communication and computation overlapping. To do so this article uses the [MPI API](https://www.open-mpi.org/) which in theory can be exchanged with for instance the [NCLL API](https://developer.nvidia.com/nccl).

# References

- [<i class="fa-solid fa-blog"></i> Paras Dahal -- Softmax and Cross Entropy Loss](https://parasdahal.com/softmax-crossentropy/)
- [<i class="fa-solid fa-file-lines"></i> PyTorch DistributedDataParallel](https://doi.org/10.48550/arXiv.2006.15704)
- [<i class="fa-solid fa-file-lines"></i> PyTorch CrossEntropyLoss -- Docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [<i class="fa-brands fa-github"></i> Full reference code](https://github.com/KoniHD/MNIST-MLP)


<!-- read-time-stop -->

# Implementation Code -- CUDA

## Kernel

```cpp
/// @brief Bias-add tail after the cuBLAS GEMM, with optional fused ReLU.
/// @tparam FuseReLU true for layer 1 (stores pre-activation in `pre`, ReLU in `post`);
///                  false for layer 2 (pre == post == biased logits).
template<bool FuseReLU>
__global__ void kernel_linear_forward(float *pre, float *post, const float *bias, int B, int out_dim)
{
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    int o  = blockIdx.y * blockDim.y + threadIdx.y;
    if (bi >= B || o >= out_dim)
        return;

    int idx  = bi * out_dim + o;
    float v  = pre[idx] + bias[o];
    pre[idx] = v; // pre-activation gets cached for the backward mask
    if constexpr (FuseReLU)
        post[idx] = relu(v);
    else
        post[idx] = v;
}

/// @brief Fused softmax + cross-entropy: one block per sample, one thread per class.
///
/// For sample s, class k:
///   reduction 1: a = max_k z[s][k]  (numerical-stability shift; argmax kept for accuracy)
///   reduction 2: denom = Σ_k exp(z[s][k] - a)
///   gradient:    g[s][k] = (1/N)(exp(z[s][k]-a)/denom - target[s][k])
///
/// Thread 0 folds per-sample loss and correct-count into metric[] via atomicAdd.
__global__ void kernel_softmax_ce(const float *z, // [N][OUTPUT_DIM]
                                  const float *target, // [N][OUTPUT_DIM]  one-hot floats
                                  float *g, // [N][OUTPUT_DIM]  gradient output
                                  float *metric, // [2] {loss_sum, correct_count}
                                  int N)
{
    const int s = blockIdx.x; // sample index
    const int k = threadIdx.x; // class index (0 .. SOFTMAX_BLOCK-1)
    if (s >= N)
        return;

    const float *zrow       = z + s * OUTPUT_DIM;
    const float *target_row = target + s * OUTPUT_DIM;
    float *grow             = g + s * OUTPUT_DIM;

    __shared__ float value_scratch[SOFTMAX_BLOCK]; // value scratch (max, then sum)
    __shared__ int index_scratch[SOFTMAX_BLOCK]; // index scratch (argmax)
    __shared__ float target_logit; // logit of the true class
    __shared__ int target_label; // index of the true class

    const int active = k < static_cast<int>(OUTPUT_DIM);
    const float zk   = active ? zrow[k] : -INFINITY;

    // Exactly one lane is hot in the one-hot target row: it records the label.
    if (active && target_row[k] != 0.f) {
        target_logit = zk;
        target_label = k;
    }

    // === reduction 1: z_max = max_k z ===
    value_scratch[k] = zk;
    index_scratch[k] = k;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (k < stride) {
            const float other value_scratch[k + stride];
            if (other > value_scratch[k]
                || (other == value_scratch[k] && index_scratch[k + stride] < index_scratch[k])) {
                value_scratch[k] = other;
                index_scratch[k] = index_scratch[k + stride];
            }
        }
        __syncthreads();
    }
    const float z_max    = value_scratch[0];
    const int prediction = index_scratch[0];
    __syncthreads(); // before reusing value_scratch for the sum

    // ---- reduction 2: denom = Σ_k exp(z_k - a) ----
    const float e = active ? std::exp(zk - z_max) : 0.f;
    value_scratch[k] = e;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (k < stride)
            value_scratch[k] += value_scratch[k + stride];
        __syncthreads();
    }
    const float denom = value_scratch[0];

    // ---- fused gradient g = (1/N)(softmax - target); reuse e, never log p ----
    if (active)
        grow[k] = ((e / denom) - target_row[k]) / static_cast<float>(N);

    // ---- byproducts: thread 0 folds this sample into the metric accumulator ----
    if (k == 0) {
        const float loss_s  = std::log(denom) - (target_logit - z_max); // L = -(Z_k* - z_max -(log(denom)))
        const float correct = (prediction == target_label) ? 1.f : 0.f;
        atomicAdd(&metric[0], loss_s);
        atomicAdd(&metric[1], correct);
    }
}

/// @brief ReLU-grad mask (optional) and bias-gradient column-sum.
/// @tparam FuseReLU  true for layer 1: masks dY in-place with 1[h_pre > 0] before summing.
template<bool FuseReLU>
__global__ void kernel_linear_backward(float *dY, // [B][out_dim] in: upstream grad; out (FuseReLU): masked D
                                       const float *h_pre, // [B][out_dim] pre-activation (read only if FuseReLU)
                                       float *db, // [out_dim] bias gradient
                                       int B, int out_dim)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_dim)
        return;

    float acc = 0.f;
    for (std::size_t bi = 0; bi < B; ++bi) {
        int idx = bi * out_dim + o;
        float d   = dY[idx];
        if constexpr (FuseReLU) {
            d       = d * static_cast<float>(h_pre[idx] > 0.f); // ReLU-gradient mask
            dY[idx] = d; // write back so the weight-grad GEMM reads D
        }
        acc += d;
    }
    db[o] = acc;
}
```

## Communication Overlapping

```cpp
void Module::backward(const std::vector<float> &true_labels)
{
    // DDP overlap scheme: Start non-blocking communication: training metric + four gradient all-reduces
    // CUDA-aware MPI: device pointers are handed straight to MPI.

    // softmax_ce fills _dg and the loss/accuracy byproducts in _dmetric.
    _softmax_ce(true_labels);
    cudaDeviceSynchronize();
    MPI_Iallreduce(MPI_IN_PLACE, _dmetric, 2, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &_metric_req);

    _fc2_backward();
    cudaDeviceSynchronize();
    // Start non-blocking reduction of W2 and b2
    MPI_Iallreduce(MPI_IN_PLACE, _dgW2, HIDDEN_DIM * OUTPUT_DIM, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &_grad_reqs[0]);
    MPI_Iallreduce(MPI_IN_PLACE, _dgb2, OUTPUT_DIM, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &_grad_reqs[1]);

    _fc1_backward();
    cudaDeviceSynchronize();
    // Start non-blocking reduction of W1 and b1
    MPI_Iallreduce(MPI_IN_PLACE, _dgW1, INPUT_DIM * HIDDEN_DIM, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &_grad_reqs[2]);
    MPI_Iallreduce(MPI_IN_PLACE, _dgb1, HIDDEN_DIM, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &_grad_reqs[3]);
}
```
