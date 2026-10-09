# A 股 Alpha / 利弗莫尔 — 可执行命令手册

所有命令均在项目根目录执行：

```bash
cd /Users/chendi/project/vnpy
```

Python 解释器统一用：

```bash
.venv/bin/python examples/alpha_a_share/<脚本名>.py
```

---

## 一、每日实盘流程（推荐顺序）

收盘后按此顺序操作。

| 步骤 | 命令 | 作用 |
|------|------|------|
| 1 | `predict_daily.py` | 多持有期选股 + 利弗莫尔建仓/管仓建议 |
| 2 | 手动改 `alpha_data/livermore_positions.json` | 更新持仓（或贴截图让助手改） |
| 3 | `snapshot_positions.py --note "收盘更新"` | 把当前持仓记入历史流水（复盘用） |

```bash
# 1. 每日预测（会自动拉最新日线）
.venv/bin/python examples/alpha_a_share/predict_daily.py

# 2. 更新持仓后，记入历史
.venv/bin/python examples/alpha_a_share/snapshot_positions.py --note "收盘更新"

# 查看历史流水
.venv/bin/python examples/alpha_a_share/snapshot_positions.py --list

# 强制写入一条（内容与上条相同也记）
.venv/bin/python examples/alpha_a_share/snapshot_positions.py --note "说明" --force
```

### 单票趋势检查

输出：**建仓 / 加仓 / 持仓 / 减半 / 清仓 / 观望** 及说明（结合 ML Top5、20 日突破、持仓管仓规则）。

```bash
# 默认用本地最新日线收盘价（收盘后建议先 --update）
.venv/bin/python examples/alpha_a_share/check_trend.py 600487 --update

# 盘中临时价（可选，非实时行情接口）
.venv/bin/python examples/alpha_a_share/check_trend.py 600487 --price 60.29

# 港股（仅趋势，不参与 predict_daily）
.venv/bin/python examples/alpha_a_share/check_trend.py 01024 --update
```

| 建议 | 条件（摘要） |
|------|----------------|
| **建仓** | 无仓 + ML Top5 + 20 日突破（仅 A 股） |
| **关注** | 港股无仓 + 20 日突破（仅趋势参考） |
| **加仓** | 有仓 + 涨 10% 且仍在 Top5 |
| **持仓** | 有仓 + 未触发止损/早期盈利区/移动止盈 |
| **减半/清仓** | 止损 -7%；或早期区 +3%~+15% 从高点回撤 8%；或移动止盈回撤 |
| **观望** | 无仓且未满足建仓条件 |

建仓判断依赖最新 `predict_daily` 信号（`alpha_data/signal/lgb_pred_*.parquet`），请先跑预测。

支持 `600487` 或 `600487.SSE` 两种写法。

### 模拟操盘 Web

本地 Gradio 界面：用**新模拟账户**（名称前缀 `模拟_`）、按可选交易日的日线收盘价手动买卖，含涨跌停校验与持仓/历史复盘。不改动实盘账户。

```bash
# 依赖（首次）
.venv/bin/pip install -r examples/alpha_a_share/requirements.txt

# 启动 → 浏览器打开 http://127.0.0.1:7860
.venv/bin/python examples/alpha_a_share/app_paper_trade.py
```

也可从 `menu.py` → 每日实盘 →「模拟操盘 Web」启动。

| 规则 | 说明 |
|------|------|
| 账户隔离 | 下拉只列 `模拟_` 账户；创建时自动补前缀；练习账户不出现在预测/盈亏/菜单实盘列表 |
| 成交价 | 默认=所选交易日收盘，可改；须落在涨跌停板内 |
| 涨停/跌停 | 收盘涨停拒买；收盘跌停拒卖 |
| 账本 | `alpha_data/accounts/<模拟_xxx>/livermore_positions.json` |

---

## 二、模型训练与优化

| 命令 | 作用 | 主要输出 |
|------|------|----------|
| `run_ml.py` | 训练 1/3/5 日 LightGBM 并回测 | `alpha_data/model/lgb_alpha_158_{1,3,5}d.pkl` |
| `evaluate_predictions.py` | 评估历史预测 IC、命中率等 | `alpha_data/evaluation/*.parquet` |
| `optimize_prediction.py` | 根据评估结果优化选股参数 | `alpha_data/optimized_config.json` |

```bash
.venv/bin/python examples/alpha_a_share/run_ml.py
.venv/bin/python examples/alpha_a_share/evaluate_predictions.py
.venv/bin/python examples/alpha_a_share/optimize_prediction.py
```

**建议：** 首次使用或隔一段时间重训一次 `run_ml.py`；`predict_daily.py` 依赖其中的模型文件。

---

## 三、一年回测（研究 / 对比策略）

回测区间（各脚本基本一致）：

- 训练：`2023-01-01 ~ 2025-06-28`
- 回测：`2025-06-29 ~ 2026-06-29`

| 命令 | 策略说明 |
|------|----------|
| `backtest_1y.py` | ML 等权 Top 1/3，无个股止损止盈 |
| `backtest_1y_sltp.py` | ML + 固定止损/止盈网格搜索 |
| `backtest_1y_circuit.py` | ML + 移动止盈 + 组合回撤熔断 |
| `backtest_1y_livermore.py` | **小资金 4 万 / Top5 / 整手**；利弗莫尔 vs 基准 A/B 对比 |

```bash
.venv/bin/python examples/alpha_a_share/backtest_1y.py
.venv/bin/python examples/alpha_a_share/backtest_1y_sltp.py
.venv/bin/python examples/alpha_a_share/backtest_1y_circuit.py
.venv/bin/python examples/alpha_a_share/backtest_1y_livermore.py
```

利弗莫尔回测结果：`alpha_data/livermore_result_40k.json`

---

## 四、早期 / 基准脚本（可选）

| 命令 | 作用 |
|------|------|
| `run.py` | 20 日动量因子 + 等权 Top 1/3（非 ML） |
| `run_multifactor.py` | Alpha158 多因子等权合成 + 回测 |

```bash
.venv/bin/python examples/alpha_a_share/run.py
.venv/bin/python examples/alpha_a_share/run_multifactor.py
```

---

## 五、关键数据文件

| 路径 | 说明 | 是否提交 Git |
|------|------|--------------|
| `alpha_data/livermore_positions.json` | 当前持仓快照（预测读取） | 否（.gitignore） |
| `alpha_data/livermore_positions_history.jsonl` | 持仓历史流水（复盘） | 否 |
| `alpha_data/livermore_positions.example.json` | 持仓 JSON 格式示例 | 是 |
| `alpha_data/signal/lgb_pred_YYYYMMDD.parquet` | 每日预测信号 | 视仓库习惯 |
| `alpha_data/daily/*.parquet` | 个股日线 | 视仓库习惯 |
| `alpha_data/model/*.pkl` | LightGBM 模型 | 视仓库习惯 |
| `alpha_data/optimized_config.json` | 优化后的选股参数 | 视仓库习惯 |

### 持仓 JSON 字段

```json
{
  "updated": "2026-08-24",
  "cash": 23895.34,
  "positions": {
    "600487.SSE": {
      "shares": 200,
      "cost": 60.946,
      "high": 61.56,
      "last_buy": 60.946,
      "stage": 1,
      "halved": false
    }
  }
}
```

| 字段 | 含义 |
|------|------|
| `cash` | 可用资金 |
| `shares` | 持股数 |
| `cost` | 加权成本 |
| `high` | 持仓以来最高价（只升不降；`predict_daily` 或手动改高后，截图更新不会覆盖更低） |
| `last_buy` | 上次买入价（加仓判断） |
| `stage` | 金字塔档位：1=首仓 60%，2=加满 |
| `halved` | 是否已移动止盈减半 |

---

## 六、利弗莫尔参数（与回测 / 每日预测一致）

| 参数 | 值 |
|------|-----|
| 候选池 | ML Top **5** |
| 突破窗口 N | **20** 日 |
| 加仓间距 X | **10%**（回测最优） |
| 金字塔 | **60% / 40%**（两档，整手 100 股） |
| 止损 | **7%** |
| 早期盈利区 | 峰值浮盈 **+3%~+15%**，从高点回撤 **8%** → 清仓 |
| 移动止盈 | 先涨 **15%**，回撤 **8%** → 减半再清 |
| 掉出 Top5 | **不强制卖** |

---

## 七、下单时机说明

- 信号按**收盘价**计算；A 股收盘后当天无法连续竞价成交。
- 优先：**盘后定价委托**（约 15:05–15:30，按当日收盘价）。
- 未成或部分成交：**次日开盘**补单。
- **T+1**：当日买入次日才能卖。

---

## 八、非独立执行脚本（库模块）

以下文件被其他脚本 `import`，一般不直接运行：

| 文件 | 作用 |
|------|------|
| `stock_universe.py` | 自选股股票池（168 只 A 股） |
| `horizons.py` | 多持有期 1/3/5 日配置 |
| `datafeed.py` | 日线下载（akshare / baostock） |
| `livermore_positions_store.py` | 持仓读写与历史流水 |
| `a_share_limits.py` | A 股涨跌停幅度与判定 |
| `paper_trade.py` | 模拟成交引擎（供 Web UI） |

---

## 九、设计文档

- 利弗莫尔策略设计：`docs/superpowers/specs/2026-08-21-livermore-trend-design.md`
