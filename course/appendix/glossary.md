# Glossary

A reference of every term used in the course, in alphabetical order.

---

**Activation Function** — A non-linear function applied between linear layers. Enables networks to learn complex patterns. In microgpt.py: ReLU² (`max(0, x)²`).

**Adam** — Adaptive Moment Estimation. An optimizer that combines momentum (running average of gradients) and adaptive learning rates (per-parameter step sizes). Lines 174-182.

**Attention** — A mechanism that lets tokens "look at" other tokens to gather context. Computes relevance scores via dot products between queries and keys, then takes a weighted average of values.

**Autograd** — Automatic differentiation. A system that computes derivatives automatically by recording operations and replaying them in reverse. The `Value` class implements this.

**Autoregressive** — A generation strategy where each output token becomes the input for producing the next token.

**Backward Pass** — Walking the computation graph in reverse to compute gradients using the chain rule. Triggered by `loss.backward()`.

**Bias Correction** — A fix for the zero-initialization of Adam's moment estimates, important in early training steps.

**Block Size** — The maximum sequence length the model can process. Set to 8 in microgpt.py.

**BOS (Beginning of Sequence)** — A special token (ID 26) used to mark the start and end of a sequence.

**Causal Masking** — Preventing the model from attending to future tokens. In microgpt.py, this happens naturally because tokens are processed one at a time.

**Chain Rule** — The derivative of a composition of functions equals the product of the individual derivatives: `d(f∘g)/dx = df/dg × dg/dx`.

**Computation Graph** — A tree of `Value` nodes recording all operations performed during the forward pass.

**Cosine Decay** — A learning rate schedule that smoothly decreases from the initial value to zero using a cosine curve.

**Cross-Entropy Loss** — The loss function `-log(P(correct token))`. Heavily penalizes confident wrong predictions.

**Dataset** — The collection of training examples. Here: ~32,000 human names from `input.txt`.

**Derivative** — The rate of change of a function's output with respect to its input. Tells us "which direction to nudge."

**Document** — A single training example. In this case, one name (e.g., "emma").

**Dot Product** — Multiply corresponding elements and sum: `dot(a, b) = a₀b₀ + a₁b₁ + ...`. Measures similarity between vectors.

**Embedding** — A learnable vector representation of a token. Converts token IDs into rich numerical representations.

**Embedding Table** — A matrix where row `i` is the embedding for token `i`. Token/position embeddings are both tables.

**Epoch** — One complete pass through the entire dataset.

**Epsilon (ε)** — A tiny number (e.g., 1e-5 or 1e-8) added to prevent division by zero.

**Exponential Moving Average** — `new = β × old + (1-β) × current`. Smooths a sequence of values over time.

**Forward Pass** — Computing the output from the input, step by step, creating the computation graph.

**Gradient** — The derivative of the loss with respect to a parameter. Tells us how to adjust the parameter to reduce the loss.

**Gradient Accumulation** — Summing gradient contributions from multiple paths through the graph (`+=` in backward).

**Gradient Descent** — The simplest optimizer: `parameter -= learning_rate × gradient`.

**Head (Attention Head)** — One independent attention mechanism operating on a subset of dimensions.

**Hyperparameter** — A setting chosen by the programmer (n_embd, n_head, learning_rate, etc.), not learned during training.

**Inference** — Using the trained model to generate new predictions, without updating parameters.

**KV Cache** — Storing keys and values from previous tokens to avoid recomputation during generation.

**Layer** — One complete attention+MLP block in the Transformer (microgpt.py has 1 layer).

**Learning Rate** — Step size for parameter updates. Controls how much parameters change each step.

**Linear Layer** — Matrix multiplication: `y = Wx`. Mixes and recombines information.

**Local Gradient** — The derivative of a single operation with respect to its immediate input.

**Logits** — Raw, unnormalized scores output by the model before softmax.

**Loss** — A single number measuring how wrong the model's predictions were. Lower is better.

**MLP (Multi-Layer Perceptron)** — A two-layer feedforward network with a non-linear activation in between. Expand → Activate → Compress.

**Momentum** — A running average of past gradients, used in Adam to smooth out noisy updates.

**Multi-Head Attention** — Running multiple attention heads in parallel, each on a subset of dimensions, then concatenating results.

**Normalization** — Scaling values to have consistent magnitude. Prevents numerical instability.

**Output Projection** — A linear layer applied after multi-head attention to mix the concatenated head outputs.

**Parameters** — The learnable numbers in the model. Start random, get tuned during training.

**Position Embedding** — A vector encoding a token's position in the sequence (1st, 2nd, 3rd, etc.).

**Pre-normalization** — Applying normalization before (not after) each block. Used in microgpt.py.

**Probability Distribution** — A list of non-negative numbers that sum to 1.

**Query (Q)** — In attention: "What am I looking for?" The current token's search vector.

**Key (K)** — In attention: "What do I offer?" Each token's advertisement vector.

**Value (V)** — In attention: "Here's my content." The actual information a token provides.

**ReLU** — Rectified Linear Unit: `max(0, x)`. A simple activation function.

**Residual Connection** — Adding the input back to the output: `y = x + f(x)`. Preserves information and helps gradient flow.

**RMSNorm** — Root Mean Square Normalization: `x / √(mean(x²))`. A simplified alternative to LayerNorm.

**Sampling** — Randomly choosing the next token based on the probability distribution.

**Scaled Attention** — Dividing attention scores by `√(head_dim)` to prevent softmax saturation.

**Sequence** — An ordered list of tokens.

**Softmax** — Function that converts logits to probabilities: `exp(xᵢ) / Σexp(xⱼ)`.

**State Dict** — A dictionary mapping parameter names to weight matrices.

**Temperature** — A scalar that controls randomness during generation. Low = deterministic, high = creative.

**Token** — The smallest unit the model processes. In microgpt.py: individual characters.

**Tokenizer** — The system that converts between text and token IDs.

**Topological Sort** — Ordering graph nodes so children always come before parents. Needed for the backward pass.

**Training** — The process of iteratively adjusting parameters to minimize loss.

**Training Step** — One complete forward → loss → backward → update cycle.

**Transformer** — The architecture consisting of attention + MLP + residual connections + normalization.

**Vocabulary** — The complete set of all possible tokens (27 in microgpt.py).

**Vocab Size** — The number of unique tokens (27 = 26 letters + BOS).

**Weight Matrix** — A 2D grid of learnable parameters used in linear transformations.
