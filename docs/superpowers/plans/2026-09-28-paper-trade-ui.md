# 模拟操盘 Web UI — 实现计划

依据：`docs/superpowers/specs/2026-09-28-paper-trade-ui-design.md`

## 依赖顺序

```
涨跌停公共函数
    → 模拟成交引擎（按交易日）
        → Gradio：账户 + 下单
            → 持仓页 / 历史页
                → 菜单入口 + requirements
```

## Task 1: 抽取涨跌停公共函数

**Description:** 从 `predict_daily.py` 抽出 A 股涨跌停幅度/涨停判定，并补跌停判定与涨跌停价计算，供预测与模拟下单共用。

**Acceptance criteria:**
- [ ] 存在模块（建议 `examples/alpha_a_share/a_share_limits.py`）提供：`a_share_daily_limit_pct`、`limit_up_price`、`limit_down_price`、`is_a_share_limit_up`、`is_a_share_limit_down`
- [ ] `predict_daily.py` 改为调用该模块，行为与现网一致
- [ ] 主板/创业板/ST 样例价格计算正确

**Verification:**
- [ ] `.venv/bin/python -c` 小脚本断言 10%/20%/5% 涨跌停价
- [ ] `py_compile predict_daily.py a_share_limits.py`

**Dependencies:** None  

**Files:** `examples/alpha_a_share/a_share_limits.py`, `examples/alpha_a_share/predict_daily.py`

---

## Task 2: 模拟成交引擎（无 UI）

**Description:** 实现按交易日买入/卖出、更新模拟账本与流水；含现金/持仓/整手/费用/涨跌停校验。

**Acceptance criteria:**
- [ ] 模块（建议 `examples/alpha_a_share/paper_trade.py`）提供：
  - `list_sim_accounts()` / `create_sim_account(name, cash)`（强制 `模拟_` 前缀）
  - `resolve_trade_price(vt, trade_date)` → 日线收盘
  - `execute_order(account, side, code, shares, trade_date, price=None, update_data=False)`
- [ ] 买入/卖出正确改 cash、positions、history
- [ ] 涨停拒买、跌停拒卖、价出板拒绝、现金不足/股数不足拒绝

**Verification:**
- [ ] 用临时模拟账户跑一笔买+卖（可选交易日），检查 JSON/jsonl
- [ ] 构造涨停日买入应失败

**Dependencies:** Task 1  

**Files:** `examples/alpha_a_share/paper_trade.py`（可小改 `livermore_positions_store` 若需）

---

## Task 3: Gradio 下单 + 账户页

**Description:** `app_paper_trade.py` 启动 Gradio；页签「下单」：创建/选择模拟账户、买卖表单、结果反馈。

**Acceptance criteria:**
- [ ] `gradio` 写入 requirements 并可安装
- [ ] 启动后浏览器可打开
- [ ] 仅列出 `模拟_` 账户
- [ ] 选日后自动填默认价；可改价提交
- [ ] 可选「下单前更新该股日线」

**Verification:**
- [ ] 手动：创建 `模拟_练习1`，买入一笔成功
- [ ] 实盘账户名不出现在下拉框

**Dependencies:** Task 2  

**Files:** `examples/alpha_a_share/app_paper_trade.py`, `examples/alpha_a_share/requirements.txt`

---

## Task 4: 持仓页 + 历史页

**Description:** 持仓按估值日展示市值/盈亏；历史展示流水并可按交易日筛选。

**Acceptance criteria:**
- [ ] 持仓表含现金、权益、当日盈亏、持仓盈亏
- [ ] 历史默认最近 50 条，可筛交易日
- [ ] 下单成功后可刷新两页数据

**Verification:**
- [ ] 与 `daily_pnl` 口径目测一致（同估值日）
- [ ] 历史能看到模拟买卖备注

**Dependencies:** Task 3  

**Files:** `examples/alpha_a_share/app_paper_trade.py`

---

## Task 5: 菜单入口与文档一行说明

**Description:** `menu.py` 增加「模拟操盘 Web」；README 增加启动命令。

**Acceptance criteria:**
- [ ] 菜单可启动（subprocess 或打印启动说明）
- [ ] README 有启动与模拟账户前缀说明

**Verification:**
- [ ] 从菜单或命令行均可启动

**Dependencies:** Task 4  

**Files:** `examples/alpha_a_share/menu.py`, `examples/alpha_a_share/README.md`

---

## 建议落地顺序

一次只做一个 Task；每 Task 验收后再进下一个。默认从 Task 1 开始。
