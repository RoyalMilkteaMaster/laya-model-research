# License scope and third-party notices

The root MIT license applies to original benchmark/analysis/publication code and
repository documentation. It does not relicense the paper, third-party
datasets, copied passages, pretrained models, or third-party paper-format files.
The paper and its translation remain with their authors; no additional
paper reuse license is granted here. Factual experimental measurements are
published with source attribution.

## Data

See [docs/DATA.md](docs/DATA.md) for dataset creators, original URLs, transformations,
and per-dataset terms. The same upstream terms apply to dataset-derived text in
`data/runs/main-20260930/records/`, including reference answers and retrieved evidence.
The original generated license summary is retained in `data/datasets/LICENSES.md`.

- HotpotQA: [CC BY-SA 4.0](LICENSES/HotpotQA-CC-BY-SA-4.0.txt).
- 2WikiMultihopQA: [Apache-2.0, including original notice](LICENSES/2Wiki-Apache-2.0.txt).
- MuSiQue: [CC BY 4.0](LICENSES/MuSiQue-CC-BY-4.0.txt).

## Laya and retrieval/model dependencies

This is an independent evaluation of the existing
[Laya model and library](https://github.com/NandhaKishorM/laya), developed by
Nandha Kishor / Convai Innovations. It is not the upstream Laya repository.
The controller uses [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual).
The training implementation follows Laya's RLCD/proper-scoring design; Laya remains
an external dependency. Its [Apache-2.0 license](LICENSES/Laya-Apache-2.0.txt) is retained
for attribution. This release does not bundle upstream or fine-tuned model weights.

Qwen3.5, BGE-M3, and bge-reranker-v2-m3 are also external downloads. Follow their model
cards and licenses when downloading or redistributing them. A code license does not
replace a checkpoint's license.

## Paper formatting

`paper/cvpr.sty` and `paper/ieeenat_fullname.bst` came from the
[CVPR author kit](https://github.com/cvpr-org/author-kit).
The bibliography style contains its original Patrick W. Daly copyright and LPPL
notice; those notices are preserved. These files are excluded from the root MIT grant.
