// The whole course. `ready: false` chapters show on the map as "being built" so the path is visible from day one.
// Every chapter follows: a pain you feel -> a guess -> play -> build -> break -> bank.
export const acts = [
  { id: 'a0', name: 'Prologue', blurb: 'See the trick before learning it.' },
  { id: 'a1', name: 'Act I · Counting', blurb: 'No calculus yet. Just counting, and what "good" means.' },
  { id: 'a2', name: 'Act II · Learning', blurb: 'Give the model knobs, then invent how to turn them.' },
  { id: 'a3', name: 'Act III · Context', blurb: 'A model that can look back: the transformer appears.' },
  { id: 'a4', name: 'Act IV · Training', blurb: 'Make it learn for real, and make it speak.' },
  { id: 'a5', name: 'Finale', blurb: 'You already know every line.' },
];

export const chapters = [
  { num: 0, id: 'ch0', act: 'a0', ready: true, title: 'The Trick', pain: 'What can 200 lines of Python actually do?', invent: 'Nothing yet. Just watch, then make a promise.', load: () => import('./Ch0.svelte') },
  { num: 1, id: 'ch1', act: 'a1', ready: true, title: 'Babble', pain: 'Computers can’t read letters, and random letters aren’t names.', invent: 'Tokens, the dataset, a first (terrible) generator.', load: () => import('./Ch1.svelte') },
  { num: 2, id: 'ch2', act: 'a1', ready: true, title: 'Habits', pain: 'Random is hopeless. Real names have habits.', invent: 'Probability, sampling, the bigram table.', load: () => import('./Ch2.svelte') },
  { num: 3, id: 'ch3', act: 'a1', ready: true, title: 'Scoring', pain: 'Is model B better than model A? By how much?', invent: 'Surprise, logarithms, the loss.', load: () => import('./Ch3.svelte') },

  { num: 4, id: 'ch4', act: 'a2', ready: true, title: 'Knobs', pain: 'A counting table can’t grow. It has nothing to adjust.', invent: 'Parameters, and random hill-climbing (which fails).', load: () => import('./Ch4.svelte') },
  { num: 5, id: 'ch5', act: 'a2', ready: true, title: 'The Wiggle', pain: 'Which way should each knob turn?', invent: 'Derivatives as slopes, gradient descent.', load: () => import('./Ch5.svelte') },
  { num: 6, id: 'ch6', act: 'a2', ready: true, title: 'Chains', pain: 'Knobs are buried many steps from the loss.', invent: 'The chain rule, computation graphs.', load: () => import('./Ch6.svelte') },
  { num: 7, id: 'ch7', act: 'a2', ready: true, title: 'The Machine', pain: 'Doing the chain rule by hand is hopeless.', invent: 'Autograd: you build the Value class.', load: () => import('./Ch7.svelte') },
  { num: 8, id: 'ch8', act: 'a2', ready: true, title: 'Softmax', pain: 'Knobs make any numbers, not probabilities.', invent: 'Softmax; a trained bigram that matches your table.', load: () => import('./Ch8.svelte') },

  { num: 9, id: 'ch9', act: 'a3', ready: true, title: 'Letters as Points', pain: 'One letter of context is too little.', invent: 'Embeddings, position, the dot product.', load: () => import('./Ch9.svelte') },
  { num: 10, id: 'ch10', act: 'a3', ready: true, title: 'Looking Back', pain: 'Which earlier letters matter right now?', invent: 'Attention: queries, keys, values.', load: () => import('./Ch10.svelte') },
  { num: 11, id: 'ch11', act: 'a3', ready: true, title: 'Many Eyes', pain: 'One question per step isn’t enough.', invent: 'Multi-head attention.', load: () => import('./Ch11.svelte') },
  { num: 12, id: 'ch12', act: 'a3', ready: true, title: 'Thinking', pain: 'Stacked linear layers collapse into one.', invent: 'The MLP and nonlinearity.', load: () => import('./Ch12.svelte') },
  { num: 13, id: 'ch13', act: 'a3', ready: true, title: 'Staying Alive', pain: 'Deep stacks explode or forget.', invent: 'Residual connections and RMSNorm.', load: () => import('./Ch13.svelte') },
  { num: 14, id: 'ch14', act: 'a3', ready: true, title: 'Assembly', pain: 'Pieces on the table aren’t a model.', invent: 'The full gpt() function, all 4,064 parameters.', load: () => import('./Ch14.svelte') },

  { num: 15, id: 'ch15', act: 'a4', ready: true, title: 'Adam', pain: 'Plain gradient descent zigzags and crawls.', invent: 'Momentum and per-knob step sizes.', load: () => import('./Ch15.svelte') },
  { num: 16, id: 'ch16', act: 'a4', ready: true, title: 'The Loop', pain: 'One prediction isn’t a lesson.', invent: 'Next-token training on every position.', load: () => import('./Ch16.svelte') },
  { num: 17, id: 'ch17', act: 'a4', ready: true, title: 'Speaking', pain: 'A trained model is silent until you sample it.', invent: 'Temperature and generation.', load: () => import('./Ch17.svelte') },

  { num: 18, id: 'ch18', act: 'a5', ready: true, title: 'The Reveal', pain: 'Here is microgpt.py. Read it.', invent: 'Nothing new. You recognise every line.', load: () => import('./Ch18.svelte') },
  { num: 19, id: 'ch19', act: 'a5', ready: true, title: 'Blank File', pain: 'Could you write it from nothing?', invent: 'Capstone: rebuild it yourself, then make it yours.', load: () => import('./Ch19.svelte') },
];

export const byNum = (n) => chapters.find((c) => c.num === Number(n));
