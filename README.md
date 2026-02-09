# Experiment 2: Text -> Image -> Text Reversibility (BAGEL)

> 中文 + English 双语说明。This README is bilingual (Chinese + English).

---

## 1) 概览 / Overview

本项目用于验证 T2I -> I2T 的“可逆性/可解释性”。流程：
1) 按分桶与模板生成 prompts
2) T2I 生成图片
3) I2T 反推文本
4) 合并结果并评分
5) 输出报告

This project validates the reversibility of a T2I -> I2T pipeline:
1) Generate prompts by buckets/templates
2) Generate images (T2I)
3) Caption images (I2T)
4) Merge and score
5) Report

---

## 2) 目录结构 / Directory Layout

```
/home/ma-user/work/code/bagel_experiment2/
  configs/
    exp2_json_slots_default.yaml   # 默认配置 / default config
    exp2_json_slots_smoke_20.yaml  # 20 样本冒烟 / 20-sample smoke
    exp2_json_slots_smoke_8.yaml   # 8 样本最短冒烟 / shortest smoke
  data/
    prompts/
      buckets.json                 # 分桶配额 / bucket quotas
      templates.json               # 模板家族 / template family
      vocab/                       # 词表 / vocab
        subjects/
        attributes/
        actions/
        scenes/
        styles/
        lighting/
    slot_vocab/                    # 评估词表 / eval vocab
      subject.txt
      attribute.txt
      scene.txt
      style.txt
      synonyms.json
  legacy/
    configs/                       # v1 配置 / v1 configs
    data/prompts/                  # v1 prompts
    scripts/                       # v1 scripts
  outputs/
    exp2_run_YYYYMMDD_HHMMSS/
  scripts/
    make_prompts.py                # v2 prompt 生成
    generate_images.py
    caption_images.py
    merge_results.py
    score_texts.py
    summarize.py
    run_experiment2.sh
  README.md
```

---

## 3) 关键路径 / Key Paths

BAGEL 模型：
```
/home/ma-user/work/models/bagel_base/BAGEL-7B-MoT
```

BAGEL 推理代码：
```
/home/ma-user/work/code/bagel
```

---

## 4) 配置说明 / Config Notes

默认配置：
- [configs/exp2_json_slots_default.yaml](configs/exp2_json_slots_default.yaml)

核心字段（含解释）：
- `prompt_mode`: v2 (使用分桶+模板)
- `vocab_dir`: 词表根目录
- `bucket_config`: 分桶配额
- `template_file`: 模板家族
- `caption_prompt`: I2T 反推提示词
- `disable_styles`: 是否屏蔽 style 词表
- `embedding_model_path`: 评分 embedding 模型
- `offline_embeddings`: 是否离线加载

---

## 5) v2 Prompt 生成 / Prompt Generation

生成逻辑：
- 按 `buckets.json` 分配各 bucket 的数量
- 按 `templates.json` 选择模板与复杂度
- 从 vocab 中抽样槽位
- Jaccard 去重（阈值可调）

提示：
- `disable_styles: true` 可关闭 style 词表，提升 I2T 准确性测试稳定性

---

## 6) 端到端流程 / End-to-End Pipeline

顺序如下：
1) `make_prompts.py` 生成 prompts 和 samples
2) `generate_images.py` 生成图片与 gen_results
3) `caption_images.py` 生成反推文本
4) `merge_results.py` 合并为 combined_results
5) `score_texts.py` 计算指标
6) `summarize.py` 生成报告

脚本入口（推荐）：
```
bash /home/ma-user/work/code/bagel_experiment2/scripts/run_experiment2.sh \
  --config /home/ma-user/work/code/bagel_experiment2/configs/exp2_json_slots_default.yaml \
  --nproc 4
```

---

## 7) 输出产物 / Outputs

每次运行生成：
- `prompts_expanded.jsonl`
- `samples.jsonl`
- `images/*.png`
- `gen_results_rank*.jsonl`
- `caption_results_rank*.jsonl`
- `combined_results.jsonl`
- `scored_results.jsonl`
- `scored_results.csv`
- `pairs_preview.csv`
- `report.md`
- `meta.json`

---

## 8) 评分说明 / Scoring

评分策略：
- token F1 / ROUGE-L
- slot F1（若 I2T 输出包含可解析 JSON）
- embedding cosine（SentenceTransformers 或 HF 模型）

说明：
- 若 I2T 没有输出 JSON，仍会用词表与 token 评分
- 更强的 embedding 只能提升评估稳定性，不能修复 I2T 语义漂移

---

## 9) 常见问题 / FAQ

Q: 为什么进程被中断？
- 通常是外部 SIGINT（例如 Ctrl+C 或平台自动终止）

Q: 如何验证改动后仍有效？
- 使用 `exp2_json_slots_smoke_8.yaml` 跑 8 图冒烟

---

## 10) GitHub 清理建议 / GitHub Hygiene

已添加 `.gitignore`，默认忽略：
- outputs
- 模型权重（.pt/.pth/.ckpt）
- __pycache__
- 数据集图片

如需保留某些输出，请手动移出 ignored 目录。

