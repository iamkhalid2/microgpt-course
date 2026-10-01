// Loads the same input.txt that microgpt.py reads, and derives everything the widgets need.
import * as S from './stats.js';

export const data = $state({ status: 'idle', error: null, docs: [], vocab: null, train: [], test: [], uni: [], bi: [], losses: null, vc: null });
let promise = null;

export function loadData() {
  if (promise) return promise;
  data.status = 'loading';
  promise = fetch(new URL('data/input.txt', document.baseURI))
    .then((r) => { if (!r.ok) throw new Error('could not load names'); return r.text(); })
    .then((text) => {
      const docs = S.parseDocs(text);
      const vocab = S.buildVocab(docs);
      const { train, test } = S.splitDocs(docs);
      const uni = S.unigramCounts(docs, vocab);
      const bi = S.bigramCounts(docs, vocab);
      data.docs = docs; data.vocab = vocab; data.train = train; data.test = test; data.uni = uni; data.bi = bi;
      data.vc = S.vowelTransitions(docs);
      data.losses = {
        chance: S.lossUniform(vocab.V),
        unigram: S.lossUnigram(docs, vocab),
        bigram: S.lossBigram(docs, vocab),
        bigramHeldOutRaw: S.lossBigram(test, vocab, train),
        bigramHeldOutSmooth: S.lossBigram(test, vocab, train, 1),
      };
      data.status = 'ready';
      return data;
    })
    .catch((e) => { data.status = 'error'; data.error = String(e); throw e; });
  return promise;
}
