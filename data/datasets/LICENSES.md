# 資料集授權

由 `multihop-benchmark prepare-data` 產生；授權於 2026-09-30 依下列原始來源查證。

| 資料集 | Hugging Face 來源 | 授權 | 查證來源 |
|---|---|---|---|
| HotpotQA（distractor） | `hotpotqa/hotpot_qa` | CC BY-SA 4.0 | https://hotpotqa.github.io/ ；HF dataset card metadata `license: cc-by-sa-4.0` |
| 2WikiMultihopQA | `framolfese/2WikiMultihopQA` | Apache-2.0 | https://github.com/Alab-NII/2wikimultihop/blob/main/LICENSE （GitHub API `spdx_id: Apache-2.0`）；HF card 內文聲明沿用 Apache License 2.0（無 metadata 欄位） |
| MuSiQue（musique-ans） | `dgslibisey/MuSiQue` | CC BY 4.0 | https://github.com/StonyBrookNLP/musique/blob/main/LICENSE （GitHub API `spdx_id: CC-BY-4.0`）；HF 鏡像無 dataset card 與授權標示 |

本目錄的 `sample_200`／`train_800`／`calibration_100`／`test_100`／`corpus` 為上述資料集的抽樣與
格式轉換（衍生物），沿用各自原授權；HotpotQA 衍生物依 CC BY-SA 4.0 以相同授權分享。
