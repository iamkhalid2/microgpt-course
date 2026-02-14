# Generating Text

## The Payoff

The model is trained. Its 4,064 parameters have been carefully adjusted over 500 steps. Now we get to **use it** — generating names that the model has never seen but that "sound like" real names.

## The Generation Loop (Lines 186–200)

```python
# Lines 186-200 of microgpt.py
temperature = 0.5
print("\n--- inference ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS                         # start with BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)  # predict next
        probs = softmax([l / temperature for l in logits])  # temp-scaled probs
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:                # BOS means "stop"
            break
        sample.append(uchars[token_id])    # decode token to character
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

## How It Works

The generation process is simple:

```
1. Start with the BOS token
2. Feed it to the model → get probabilities for all 27 possible next tokens
3. Randomly pick one token (weighted by probabilities)
4. If the picked token is BOS → stop (name is complete)
5. Otherwise → add the character to the output
6. Use the picked token as input for step 2
7. Repeat
```

### Traced example

```
Step 1: Input BOS
        Model predicts: 'e'(15%), 'a'(12%), 's'(10%), 'm'(9%), ...
        Random pick: 'k' (token 10)
        sample = ['k']

Step 2: Input 'k'
        Model predicts: 'a'(22%), 'e'(18%), 'i'(15%), ...
        Random pick: 'a' (token 0)
        sample = ['k', 'a']

Step 3: Input 'a'
        Model predicts: 'y'(14%), 'r'(12%), 'l'(11%), 'n'(10%), ...
        Random pick: 'y' (token 24)
        sample = ['k', 'a', 'y']

Step 4: Input 'y'
        Model predicts: BOS(35%), 'a'(15%), 'l'(12%), ...
        Random pick: 'a' (token 0)
        sample = ['k', 'a', 'y', 'a']

Step 5: Input 'a'
        Model predicts: BOS(40%), 'n'(10%), ...
        Random pick: BOS (token 26)
        STOP!

Output: "kaya"
```

The model invented the name "kaya" — it wasn't explicitly in the training data, but it follows the patterns the model learned.

## Key Details

### Fresh KV cache (Line 190)

```python
keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
```

Each generation starts with an empty KV cache. The model has no context — only the BOS token.

### Starting token (Line 191)

```python
token_id = BOS
```

We always start with BOS because that's what the model was trained to expect at the start of a name.

### Sampling (Line 196)

```python
token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
```

`random.choices` picks a random token, weighted by the probabilities. Higher probability → more likely to be chosen, but not guaranteed. This **randomness** is why each run can produce different names.

### Stopping condition (Lines 197-198)

```python
if token_id == BOS:
    break
```

During training, the model learned to predict BOS after a name ends. During generation, when it predicts BOS, we take that as the signal to stop.

### Decoding (Line 199)

```python
sample.append(uchars[token_id])
```

Convert the token ID back to a character using our vocabulary. This is the reverse of encoding.

## Training vs. Inference

| | Training | Inference |
|---|---|---|
| **Goal** | Adjust parameters | Generate text |
| **Loss computed?** | Yes | No |
| **Backward pass?** | Yes | No |
| **Parameters change?** | Yes | No (frozen) |
| **Input** | Real data | Model's own output |
| **Uses** | Building knowledge | Applying knowledge |

Notice that during inference, we **don't** compute gradients or update parameters. The model is frozen. We're just running the forward pass repeatedly.

## Max Length

```python
for pos_id in range(block_size):  # block_size = 8
```

Generation is capped at `block_size` (8) characters. Even if the model doesn't produce BOS, it stops at 8 characters. This is because the position embedding table only has 8 entries.

## Terminology

| Term | Meaning |
|------|---------|
| **Inference** | Using the trained model to generate predictions |
| **Generation** | Producing new text by repeatedly sampling from the model |
| **Sampling** | Randomly choosing the next token based on probabilities |
| **Autoregressive** | Each generated token becomes the input for the next step |

## Next

The randomness of generation can be controlled with [temperature and sampling](./01-temperature-and-sampling.md).
