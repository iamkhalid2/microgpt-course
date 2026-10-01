// microgpt.py is imported verbatim (single source of truth). This file says which chapter earns which lines.
import source from '../../../microgpt.py?raw';

export const LINES = source.replace(/\n$/, '').split('\n');

// A "code line" is any non-blank line inside or after the docstring that is not just a comment.
export const CODE_LINES = LINES.map((t, i) => [t, i + 1])
  .filter(([t]) => t.trim() !== '' && !t.trim().startsWith('#'))
  .map(([, n]) => n);

// Inclusive [start, end] line ranges, 1-indexed. Only lines whose every idea has been built by then.
export const UNLOCKS = {
  ch0: [[1, 7]],
  ch1: [[9, 12], [14, 21], [23, 27], [155, 155], [157, 157]],
  ch2: [[186, 186], [188, 189], [191, 192], [197, 200]],
  ch3: [[162, 162], [164, 164], [167, 169]],
  ch4: [[74, 74], [90, 90]],
  ch5: [[151, 153]],
  ch6: [[29, 29]],
  ch7: [[30, 72]],
  ch8: [[97, 101], [166, 166]],
  ch9: [[80, 81], [94, 95], [109, 111]],
  ch10: [[118, 122], [129, 131]],
  ch11: [[123, 128], [132, 133]],
  ch12: [[138, 140]],
  ch13: [[103, 106], [112, 112], [116, 117], [134, 134], [136, 137], [141, 141]],
  ch14: [[75, 79], [82, 89], [108, 108], [114, 114], [143, 144]],
  ch15: [[146, 149], [174, 182]],
  ch16: [[156, 156], [158, 161], [163, 163], [165, 165], [171, 172], [184, 184]],
  ch17: [[187, 187], [190, 190], [193, 196]],
  ch18: [],   // nothing new: every line was earned by now. This chapter is about reading the whole file.
  ch19: [],
};

// Lines a lesson is "about" right now (pulsed in the file panel while you read it).
export const FOCUS = {
  ch0: [[1, 7]],
  ch1: [[14, 27], [157, 157]],
  ch2: [[191, 200]],
  ch3: [[162, 169]],
  ch4: [[74, 90]],
  ch5: [[151, 153]],
  ch6: [[59, 72]],
  ch7: [[30, 72]],
  ch8: [[94, 101]],
  ch9: [[80, 81], [94, 95], [109, 111]],
  ch10: [[115, 134]],
  ch11: [[123, 133]],
  ch12: [[135, 141]],
  ch13: [[103, 106], [112, 112], [116, 117], [134, 134], [136, 137], [141, 141]],
  ch14: [[74, 144]],
  ch15: [[146, 149], [174, 182]],
  ch16: [[151, 184]],
  ch17: [[186, 200]],
};
