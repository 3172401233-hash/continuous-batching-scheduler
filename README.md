# continuous-batching-scheduler
Python 连续批处理调度库，处理请求优先级、token 预算、KV 页约束和公平性

## 概述

传统固定 batch 的 LLM 服务中，长请求会拖住同批的短请求，新请求只能等下一批。
本库实现 iteration-level 的连续批处理：每个 decode tick 重新选择下一批执行的请求，
新请求随到达随调度，完成的请求立即释放 KV 页，GPU 利用率保持平稳。

时间为离散 tick（由调度器内部推进，不依赖墙钟），相同输入序列必然产生相同调度，
便于测试与复现。

## 快速开始

```python
from continuous_batching import Request, Scheduler

sched = Scheduler(num_kv_pages=64, max_batch_size=8, max_tokens_per_tick=512)
sched.add_request(Request(
    request_id="r1",
    prompt_tokens=128,       # prompt token 数
    max_output_tokens=64,    # 预计生成 token 数
    deadline=100,            # 绝对 tick 截止时间
    priority=5,              # 数值越大优先级越高
    kv_pages=None,           # 缺省按 (prompt+output)/page_size 推导
))
batch = sched.step()         # 每个 decode tick 调用一次
print(batch.decoded, batch.finished, batch.preempted)
print(sched.stats.throughput, sched.stats.timeout_rate)
```

每个 tick 内：`step()` 先驱逐已过期的请求，再按策略准入/抢占，
最后在 token 预算内执行 chunked prefill（长 prompt 跨 tick 分片）与 decode
（prefill 完成的最后一个分片同时产出首 token）。

## 保序策略（排序规则）

所有待调度请求按如下全序键排序，保证确定性：

1. **紧急度**：`deadline - now - 剩余生成 token <= 0` 的请求标记为 urgent，
   即使立即每 tick 解码也无法按时完成，排最前；
2. **有效优先级**：`priority + 等待tick数 // aging_interval + 被抢占次数`，
   等待越久、被抢占越多，有效优先级越高（aging 防饥饿）；
3. **截止时间**：EDD（earliest deadline first）；
4. **到达序号**：同级请求严格 FIFO 保序。

## 替换策略（抢占规则）

- 当 KV 页不足或 batch 已满时，候选请求只会抢占**排序键严格更差**的在运行请求，
  且按最差者优先、最小化抢占数量；即使抢占所有更差者仍放不下则不抢占，避免无效颠簸。
- 抢占采用 **swap-out**：被抢占请求保留已生成 token 与 prefill 进度，仅释放 KV 页，
  重新准入时无需重算，因此抢占不会浪费已完成的计算。
- 被抢占会累积 `preemptions` 计数并抬高有效优先级，配合 aging 保证
  低优先级请求最终一定被重新调度，**不会饥饿**。
- 超过 deadline 的请求被驱逐并计入超时；`cancel()` 可在任意时刻取消
  等待中或运行中的请求并立即释放 KV 页。

## 资源约束

- `num_kv_pages`：KV 缓存总页数，运行中请求的页数总和永不超过该值；
- `max_batch_size`：每 tick 最多运行的请求数；
- `max_tokens_per_tick`：每 tick 的 token 预算，由 prefill 分片与 decode 共享，
  按排序键从高到低分配。

## 测试

```bash
python -m unittest
```

覆盖：持续到达（吞吐与完成率）、长尾解码（长请求不阻塞短请求）、
KV 页不足（抢占且无死锁）、并发取消（运行/等待混合取消、页释放）、
优先级反转（紧急请求抢占 + aging 防饥饿、超时率统计）、
以及同输入确定性（两次运行 trace 与统计完全一致）。
