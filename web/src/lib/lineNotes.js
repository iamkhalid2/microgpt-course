// Plain-English notes for every part of microgpt.py. Ranges are inclusive, 1-indexed, and the NARROWEST range containing
// a line wins. `ch` is the chapter where the idea was built.
export const NOTES = [
  { from: 1, to: 7, ch: 0, title: 'The file introduces itself', text: 'A short description: "The most atomic way to train and inference a GPT in pure, dependency-free Python." Everything below is the complete algorithm. Everything else, the author says, is just efficiency.' },
  { from: 9, to: 11, ch: 1, title: 'Borrow three toolkits', text: 'os lets the program check whether a file exists. math supplies log and exp. random supplies the dice. These are all part of Python itself, which is what "dependency-free" means.' },
  { from: 12, to: 12, ch: 1, title: 'Fix the dice', text: 'Seed 42 makes the "random" numbers the same on every run, so results are repeatable. "Let there be order among chaos."' },
  { from: 14, to: 21, ch: 1, title: 'Get the names', text: 'Download the list of names if it is not already here (15–18), read it into a list called docs with one name per entry (19), shuffle it so the order is random (20), and say how many there are (21): 32,033.' },
  { from: 23, to: 27, ch: 1, title: 'Letters become numbers', text: 'Find every distinct character in the names, sorted, and number them (24). Give the start/end symbol the next free number (25). Count the total, 27 (26).' },
  { from: 29, to: 29, ch: 6, title: 'The one-line summary of autograd', text: 'Apply the chain rule, recursively, across a computation graph. The next 44 lines do exactly that.' },
  { from: 30, to: 37, ch: 7, title: 'A number that remembers', text: 'A Value is a card with four boxes: its value (data), its slope (grad, starting at 0), the cards it was made from (children), and how sensitive it is to each of them (local_grads).' },
  { from: 39, to: 41, ch: 7, title: 'Adding two cards', text: 'Wrap a plain number in a card if needed (40), then make a new card whose value is the sum, remembering both ingredients, each with a local slope of 1 (41).' },
  { from: 43, to: 45, ch: 7, title: 'Multiplying two cards', text: 'Same shape as adding, but the local slopes are each other\'s values: nudging one factor changes the product by the other factor (45).' },
  { from: 47, to: 50, ch: 7, title: 'Power, log, exp, relu', text: 'Four more steps, each with its own local slope: x to the power k has slope k·x^(k−1); log has 1/x; exp is its own slope; relu passes slope through if the value was positive, and blocks it otherwise.' },
  { from: 51, to: 57, ch: 7, title: 'Conveniences', text: 'Negation, subtraction and division are rewritten in terms of the steps above (a − b is a + (−b); a ÷ b is a × b⁻¹). The "r" versions make 3 + v work as well as v + 3.' },
  { from: 59, to: 72, ch: 7, title: 'The backward sweep', text: 'Line the cards up so each comes after its ingredients (60–68, a recursive recipe), set the final card\'s slope to 1 (69), then walk the list backwards (70) and add (local slope × this card\'s slope) to each ingredient (72).' },
  { from: 72, to: 72, ch: 6, title: 'The chain rule, in one line', text: 'child.grad += local_grad * v.grad. The ingredient\'s slope grows by (how sensitive v is to it) × (how sensitive the loss is to v). The += adds up the contributions when a card is used in several places.' },
  { from: 74, to: 74, ch: 4, title: 'Start of the dials', text: 'Everything the model will ever "know" lives in numbers called parameters. This section creates them.' },
  { from: 75, to: 79, ch: 14, title: 'The shape of the model', text: 'Numbers per letter: 16 (75). Attention heads: 4 (76). Layers: 1 (77). Longest name: 8 positions (78). Numbers per head: 16 ÷ 4 = 4 (79). Change these and you change the model\'s size.' },
  { from: 80, to: 80, ch: 9, title: 'A table of small random numbers', text: 'matrix(rows, columns) makes a table of random dials from a bell curve centred on 0 with spread 0.02, each wrapped in a Value so slopes can be tracked.' },
  { from: 81, to: 81, ch: 9, title: 'The three big tables', text: 'wte: coordinates for each of the 27 symbols. wpe: coordinates for each of the 8 positions. lm_head: the final table that turns 16 numbers into 27 scores.' },
  { from: 82, to: 88, ch: 14, title: 'The tables inside each layer', text: 'Query, key, value and output tables for attention (83–86), then the expand and shrink tables of the MLP (87–88). Two start at zero (std=0, lines 86 and 88) so each block begins by adding nothing to the lane.' },
  { from: 89, to: 89, ch: 14, title: 'One flat list of every dial', text: 'Gather every Value from every table into one long list, params. The optimiser will walk through it.' },
  { from: 90, to: 90, ch: 4, title: 'Count the dials', text: 'Prints "num params: 4064".' },
  { from: 92, to: 93, ch: 14, title: 'The plan', text: 'The model is a function that maps symbols and dials to scores. It follows GPT-2 with three small changes: RMSNorm instead of LayerNorm, no bias terms, and ReLU-squared instead of GeLU.' },
  { from: 94, to: 95, ch: 9, title: 'A linear layer', text: 'For each row of a table, take the dot product of the row with the input. A table times a list is just one dot product per row.' },
  { from: 97, to: 101, ch: 8, title: 'Softmax', text: 'Subtract the biggest score (98, to avoid overflow), exp every score (99), add them up (100), and give each its share (101).' },
  { from: 103, to: 106, ch: 13, title: 'RMSNorm', text: 'Find how loud the list is (the mean of the squares, 104), compute 1 ÷ √(that + a tiny safety amount) (105), and multiply every number by it (106).' },
  { from: 108, to: 108, ch: 14, title: 'The model function', text: 'Read one symbol at one position and return 27 scores for what comes next. keys and values are the filing cabinet of earlier positions.' },
  { from: 109, to: 112, ch: 9, title: 'Letter plus position', text: 'Look up the letter\'s row (109) and the position\'s row (110), add them (111), and normalise (112).' },
  { from: 114, to: 114, ch: 14, title: 'For each layer', text: 'Repeat the two blocks below once per layer. Here there is one.' },
  { from: 115, to: 134, ch: 10, title: 'The attention block', text: 'Remember the lane (116), normalise a copy (117), make query, key and value (118–120), file the key and value (121–122), let each head blend (124–132), mix the heads (133), and add the answer to the lane (134).' },
  { from: 116, to: 117, ch: 13, title: 'Save the lane, normalise a copy', text: 'x_residual keeps the lane as it was. The block works on a normalised copy.' },
  { from: 118, to: 122, ch: 10, title: 'Query, key, value, and the filing cabinet', text: 'Three linear layers give each position its question, its badge and its note. The key and value are filed so later positions can look back at them.' },
  { from: 123, to: 128, ch: 11, title: 'Split into heads', text: 'For each head, take its own 4-number slice of the query and of every filed key and value.' },
  { from: 129, to: 129, ch: 10, title: 'Match the question against every badge', text: 'The dot product of the query with each earlier key, divided by √(head size) so the scores stay in a friendly range.' },
  { from: 130, to: 131, ch: 10, title: 'Weights, then blend', text: 'Softmax the matches into attention weights (130), then add up each earlier position\'s value scaled by its weight (131).' },
  { from: 132, to: 133, ch: 11, title: 'Glue the heads, mix them', text: 'Put the four heads\' answers side by side (132) and mix them with one more linear layer (133).' },
  { from: 134, to: 134, ch: 13, title: 'Add the block\'s answer to the lane', text: 'The skip lane: the attention block\'s output is added to what was already there, so nothing is lost.' },
  { from: 135, to: 141, ch: 12, title: 'The MLP block', text: 'Save the lane (136), normalise (137), expand 16 → 64 (138), gate with ReLU and square (139), shrink 64 → 16 (140), and add back to the lane (141).' },
  { from: 143, to: 144, ch: 14, title: 'Scores for the next symbol', text: 'One last linear layer turns the 16 numbers into 27 scores, one per possible next symbol, and returns them.' },
  { from: 146, to: 149, ch: 15, title: 'Set up Adam', text: 'Learning rate 0.01, memory settings 0.9 and 0.95, a safety 1e-8 (147), and two lists with one running average per dial (148–149).' },
  { from: 151, to: 153, ch: 5, title: 'The training loop begins', text: 'Repeat for 500 steps. Each step teaches the model from one name.' },
  { from: 155, to: 158, ch: 16, title: 'Pick a name and wrap it', text: 'Take the next name (156), wrap it in start/end symbols as numbers (157), and keep at most 8 lessons (158).' },
  { from: 160, to: 169, ch: 16, title: 'Forward: score every lesson', text: 'For each position, run the model (165), softmax (166), take the surprise at the right next symbol (168), and average them into one loss (169).' },
  { from: 168, to: 168, ch: 3, title: 'Surprise', text: 'Minus the log of the probability the model gave to the symbol that really came next.' },
  { from: 171, to: 172, ch: 7, title: 'Backward', text: 'loss.backward() runs your Chapter 7 engine and fills in the slope of every one of the 4,064 dials.' },
  { from: 174, to: 182, ch: 15, title: 'Adam update', text: 'Work out today\'s learning rate on the cosine schedule (175). For each dial: update its two running averages (177–178), correct them (179–180), step the dial (181), and wipe its slope (182).' },
  { from: 184, to: 184, ch: 16, title: 'Report', text: 'Print the step number and this step\'s loss.' },
  { from: 186, to: 200, ch: 17, title: 'Write some names', text: 'The model writes 20 names. Temperature 0.5 (187). For each name: start at the start symbol (191), ask the model for scores (194), divide by the temperature and softmax (195), spin the weighted wheel (196), stop if it picks the end symbol (197–198), otherwise write the letter (199).' },
];

// The narrowest note that contains a line.
export function noteFor(n) {
  let best = null;
  for (const e of NOTES) if (n >= e.from && n <= e.to && (!best || e.to - e.from < best.to - best.from)) best = e;
  return best;
}
