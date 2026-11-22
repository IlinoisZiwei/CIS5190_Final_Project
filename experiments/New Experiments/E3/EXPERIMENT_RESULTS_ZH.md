# 获准实验：做了什么、为什么、结果

E0 是基础准备，用户确认的六个实验编号为 E1–E6。以下数字来自保存的实验结果。训练内部 OOF、复用开发集和排行榜成绩分别记录；本轮没有新排行榜成绩。E4 须通过 E1/E2 前置门槛；门槛失败停止不等于完成了训练。

| 阶段 | 状态 | 做了什么 | 为什么 | 结果 |
|---|---|---|---|---|
| E0 | complete | 审计获准输入，并复核原 V4 validated | 确认数据边界和保底模型的可复现性 | course_dev: 810/951; legacy_dev: 142/306; paired_dev: 115/200；审计训练来源记录 8096 条 |
| E1 | complete | 按冻结协议补缺失嵌套折 | 用三套完整划分评估选择流程 | Nested 9/9 folds; internal gate=True. course -0.223 pp [-0.810, +0.383]; new +4.000 pp [+2.774, +5.198]; old +0.409 pp [-0.492, +1.334] |
| E2 | complete | 锁定 A/C 各拟合与导出一次，在三个开发面板确认 | 检验内部收益是否迁移，且不按开发结果再调参 | course_dev: A 776/951, C 764/951; external_dev: A 122/306, C 125/306; paired_dev: A 112/200, C 120/200; development gate=False; fits=2 |
| E3 | complete | 汇总错误变化、长度切片、区间与受控对照 | 说明改进或退步发生在哪里，并量化条件不确定性 | course_dev: fixed 22, new errors 34, persistent 153; net -12; external_dev: fixed 11, new errors 8, persistent 173; net +3; paired_dev: fixed 11, new errors 3, persistent 77; net +8 |
| E4 | not_started_gate_failed | 有条件地把固定词法适配迁移到 V4 | 先通过内部与开发保护门槛，再使用条件预算 | E4 已有条件授权；前置门槛失败，未启动训练：G5_course_accuracy, G5_course_NBC_recall。Course 776→764/951; NBC 387→382/475; allowed extra errors/misses 4/4. 模型拟合次数为 0。 |
| E5 | complete | 冻结最终模型及其来源 | 分开开发评估模型和全数据提交模型 | 决策=retain_original_v4_submission；拟合行数=4800；新增训练=False；覆盖原模型=False。810/951 属于 validated 模型，不是拟合全部 4800 行的提交模型的独立开发成绩。 |
| E6 | complete | 验证部署接口与 CPU 耗时 | 在获准或合成输入上检查课程推理要求 | 1 threads, 3 repeats: 18.28–18.63s wall; 4 threads, 3 repeats: 13.48–13.57s wall；PyTorch 2.9.1；通过本地目标=True。仅使用获准的 8000 行计时样例，服务器硬件等价性未知。 |
