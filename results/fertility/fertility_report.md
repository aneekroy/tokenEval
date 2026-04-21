# Tokenizer Fertility Report


## Corpus: `owt`

| Tokenizer | \|V\| | tok/word (mean) | tok/word (p95) | bytes/tok | unk% |
|---|---:|---:|---:|---:|---:|
| byt5 | 256 | 4.103 | 9.0 | 1.000 | 0.0000 |
| llama2 | 32000 | 1.244 | 3.0 | 3.890 | 0.0000 |
| gpt2 | 50257 | 1.278 | 2.0 | 4.411 | 0.0000 |
| qwen25 | 151643 | 1.253 | 2.0 | 4.566 | 0.0000 |
| gemma3 | 262144 | 1.165 | 2.0 | 4.508 | 0.0000 |

## Corpus: `text8`

| Tokenizer | \|V\| | tok/word (mean) | tok/word (p95) | bytes/tok | unk% |
|---|---:|---:|---:|---:|---:|
| byt5 | 256 | 4.900 | 10.0 | 1.000 | 0.0000 |
| llama2 | 32000 | 1.291 | 3.0 | 4.566 | 0.0000 |
| gpt2 | 50257 | 1.388 | 3.0 | 5.012 | 0.0000 |
| qwen25 | 151643 | 1.338 | 3.0 | 5.142 | 0.0000 |
| gemma3 | 262144 | 1.213 | 2.0 | 5.365 | 0.0000 |

## Segmentation Examples

These show *how* each tokenizer splits morphologically non-trivial words.


### `byt5`

- `🙂` → `['ð', '\x9f', '\x99', '\x82']`
- `TensorFlow` → `['T', 'e', 'n', 's', 'o', 'r', 'F', 'l', 'o', 'w']`
- `北京` → `['å', '\x8c', '\x97', 'ä', 'º', '¬']`

### `llama2`

- `🙂` → `['▁', '<0xF0>', '<0x9F>', '<0x99>', '<0x82>']`
- `TensorFlow` → `['▁T', 'ensor', 'Flow']`
- `北京` → `['▁', '北', '京']`

### `gpt2`

- `🙂` → `['ðŁ', 'ĻĤ']`
- `TensorFlow` → `['T', 'ensor', 'Flow']`
- `北京` → `['åĮ', 'Ĺ', 'äº', '¬']`

### `qwen25`

- `🙂` → `['ðŁĻĤ']`
- `TensorFlow` → `['Tensor', 'Flow']`
- `北京` → `['åĮĹäº¬']`

### `gemma3`

- `🙂` → `['🙂']`
- `TensorFlow` → `['Tensor', 'Flow']`
- `北京` → `['北京']`