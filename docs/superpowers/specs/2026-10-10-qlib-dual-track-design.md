# Qlib 双轨选股/预测（对比研究）设计

日期：2026-10-10  
状态：已实现（MVP；convert/train/predict/compare 已冒烟）

## 1. 目标

在**保留现有** vnpy Alpha158 + LightGBM 选股/预测/利弗莫尔/纸交易/实盘流程的前提下，增加一条 **Qlib 研究轨**：

- 同源股票池与日线数据
- Alpha158 + LGB 训练与预测
- 导出与现有格式兼容的 `signal`（仅分数，不含权重）
- 与现有信号**并排评估**（信号层 + 策略层）
- **只写报告、不下单**；执行层默认仍读现有 `lgb_pred_*`

## 2. 已确认决策

| # | 项 | 决定 |
|---|----|------|
| 1 | 股票池 | 与现有 `STOCK_LIST` 一致 |
| 2 | 标签定义 | MVP 与 vnpy Alpha158 默认对齐：`close[t+3]/close[t+1]-1`（Qlib: `Ref($close,-3)/Ref($close,-1)-1`）；1/5 日第二期 |
| 3 | 输出 | 只出 `signal` 分数 |
| 4 | 对比 | 两层都做：信号质量 + 同一规则下策略净值 |
| 5 | 工程边界 | Qlib 独立仓 `/Users/chendi/project/qlib`；vnpy 仅 subprocess 调用其程序 |
| 6 | 上线策略 | 对比期只报告、不切换下单源 |
| 7 | MVP 范围 | 同源数据 → Qlib 训 Alpha158+LGB → 每日导出 → 并排评估 |
| MVP 持有期 | **B** | 先只训/导出 **PRIMARY_HORIZON=3**；1/5 日预留，第二期 |
| 信号落盘 | | `qlib_pred_*.parquet` **直接写入** vnpy 的 `alpha_data/signal/` |

## 3. 与现有项目的关系

| 复用（只读或调用） | 路径 |
|-------------------|------|
| 日线 | `alpha_data/daily/*.parquet` |
| 股票池 | `examples/alpha_a_share/stock_universe.py` |
| 持有期定义 | `examples/alpha_a_share/horizons.py` |
| 现有信号 | `alpha_data/signal/lgb_pred_YYYYMMDD.parquet` |
| 现有评估 | `evaluate_predictions.py` 等（扩展为双源） |

| 不动（MVP） | 说明 |
|-------------|------|
| `predict_daily.py` 默认信号源 | 仍用现有 LGB |
| Livermore / paper_trade / 账户 | 不因 Qlib 改行为 |
| 现有模型文件 | `alpha_data/model/lgb_alpha_158_*.pkl` 继续服务路径 A |

## 4. 架构

```
alpha_data/daily + STOCK_LIST
        │
        ├─ Path A（现有，不变）
        │     vnpy.alpha 训练/预测 → alpha_data/signal/lgb_pred_*.parquet
        │
        └─ Path B（新增，Qlib 仓）
              convert → train → predict_export
                    → alpha_data/signal/qlib_pred_*.parquet
                              │
                              ▼
              vnpy: compare_vnpy_qlib.py → 报告（不下单）
```

调用约定：vnpy 侧薄封装用绝对路径调用，例如：

```bash
/Users/chendi/project/qlib/.venv/bin/python \
  /Users/chendi/project/qlib/scripts/a_share_bridge/predict_export.py \
  --vnpy-root /Users/chendi/project/vnpy \
  --asof YYYY-MM-DD
```

禁止在 vnpy `.venv` 内 `import qlib`（环境隔离）。

## 5. 信号契约

文件：`alpha_data/signal/qlib_pred_YYYYMMDD.parquet`

| 列 | 类型 | 说明 |
|----|------|------|
| `datetime` | date/datetime | 信号日（与现有一致） |
| `vt_symbol` | str | 如 `600487.SSE`（与现有一致） |
| `signal` | float | 预测分数，截面越大越偏多 |
| `source` | str | 固定 `qlib`（可选但建议有） |
| `horizon` | int | MVP=`3` |

现有 `lgb_pred_*` 不改名、不覆盖。评估脚本按文件名前缀区分来源。

## 6. Qlib 仓内交付物（`/Users/chendi/project/qlib`）

建议目录：`scripts/a_share_bridge/`

| 脚本 | 作用 |
|------|------|
| `export_instruments.py` | 从 vnpy `STOCK_LIST` 生成 Qlib instruments |
| `convert_daily.py` | `alpha_data/daily/*.parquet` → Qlib bin（同源） |
| `train_lgb_alpha158.py` | Alpha158 + LGB，标签=3 日累计收益 |
| `predict_export.py` | 最新截面预测 → 写入 vnpy `alpha_data/signal/qlib_pred_*.parquet` |
| `README.md` | 独立 venv 安装与命令 |

数据目录（Qlib 侧，不进 vnpy git）：如 `~/.qlib/qlib_data/cn_a_share_vnpy` 或 qlib 仓内 `data/cn_a_share_vnpy`（实现时二选一，写入 README）。

独立 `.venv`：在 qlib 仓创建并安装 qlib 及依赖。

## 7. vnpy 侧交付物

| 项 | 说明 |
|----|------|
| `examples/alpha_a_share/run_qlib_bridge.py`（名可微调） | 封装 convert/train/predict 的 subprocess 调用；路径可配置 |
| `examples/alpha_a_share/compare_vnpy_qlib.py` | 双轨对比入口 |
| 扩展评估 | 复用/扩展 `evaluate_predictions` 逻辑，同时读 `lgb_pred_*` 与 `qlib_pred_*` |
| README 小节 | 双轨对比命令与「不下单」说明 |

### 7.1 对比报告两层

**层 1 — 信号质量（同一池、同一 3 日标签）**

- 日度 IC / ICIR  
- 分层（如 5 组）平均收益  
- 覆盖率、缺失标的数  

**层 2 — 策略代理（同一规则、仍不下单）**

- 规则与现有可比：按 `signal` 取 TopN（默认与 Livermore Top5 对齐，N 可配）  
- 可选：再套同一套 Livermore 管仓规则做净值曲线（实现成本高则可二期；MVP 至少做 **等权 TopN 持有 PRIMARY_HORIZON 日** 的截面回测）  
- 输出：净值、最大回撤、换手、与 Path A 并排表/图  
- 结果写入如 `alpha_data/evaluation/vnpy_vs_qlib_*.parquet` + 控制台/简易 markdown 摘要  

## 8. MVP 明确不做

- 切换 `predict_daily` / 纸交易默认信号源  
- 组合优化权重 `weight`  
- Qlib 1 日 / 5 日模型训练（接口预留即可）  
- 把 qlib 源码或 venv 塞进 vnpy 仓库  
- 实时行情、盘中信号  

## 9. 第二期（预留） / 已部分落地

- 对齐训练/导出 1d、5d  
- ~~策略层完整 Livermore 双轨回测~~ → MVP：`backtest_1y_livermore.py --signal-source qlib`（同规则读 `qlib_pred_*`）  
- 配置项 `signal_source=vnpy|qlib|blend`（仍默认 vnpy）— 回测已支持；执行层未切  
- 组合优化出权重  

## 10. 验收标准

1. 现有 `predict_daily` / 纸交易行为与改前一致。  
2. 运行 Qlib bridge 后，`alpha_data/signal/` 出现 `qlib_pred_YYYYMMDD.parquet`，列符合契约，`vt_symbol` 与现有池可 join。  
3. `compare_vnpy_qlib.py` 对同一评估区间产出 Path A vs B 的 IC/分层 + TopN 策略对比，且不触发任何下单。  
4. vnpy 与 qlib 环境隔离：vnpy `.venv` 无 qlib 硬依赖。  

## 11. 实现顺序建议

1. qlib 仓：venv + convert + instruments  
2. qlib 仓：train（3d）+ predict_export → 写入 vnpy signal 目录  
3. vnpy：`run_qlib_bridge.py` 调用封装  
4. vnpy：`compare_vnpy_qlib.py`（层 1，再层 2 TopN）  
5. README / 本设计状态改为「已实现」  

---

确认本设计后开始实现；实现过程中若 Qlib 数据目录具体路径有偏好，可在 README 中固定一处即可。
