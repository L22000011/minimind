# 第 1 节验证记录

日期：2026-11-01。上游提交：`233d00fa068ec2b8c0d4f07b703934d6391a5a73`。

## 环境

- Windows，独立 `.venv`，Python 3.12.13。
- torch 2.6.0+cpu，transformers 4.57.6，numpy 1.26.4。
- 此教学环境 `torch.cuda.is_available()` 为 False；这是 CPU 安装的结果，不能据此判断机器是否有 NVIDIA GPU。
- 无权重、无训练数据下载；无 GPU 基准测试、无 vLLM 部署。

## 实际执行

```powershell
.\.venv\Scripts\python.exe learning\experiments\01_rmsnorm.py
```

退出码 0，stdout：

```text
manual output: [0.8485280871391296, 1.1313707828521729]
output mean (not zero): 0.9899494647979736
3D shape: (2, 3, 4)
4D shape: (2, 3, 2, 4)
raw fp16 square has inf: True
real RMSNorm fp16 output: [0.8486328125, 1.1318359375]
input and gamma gradients: finite and nonzero
ALL CHECKS PASSED
```

断言覆盖：真实模块与公式一致、正数缩放近似不变、非均匀 γ 广播、token 间不混合、4D 最后一维归约、FP16 输出类型及有限性、零向量、输入和 γ 梯度。

导入时环境已有的 `TRANSFORMERS_CACHE` 变量触发弃用提示，不影响断言；本次没有修改全局缓存配置。

## 交付边界

已完成：代码克隆、源码地图、7 天计划、第 1 节完整讲义、CPU 最小实验、分模块模拟面试题及答案、学习日志。

未执行：Day 2～7 的实验、全量预训练/SFT、真实模型评估、格式转换和在线 Serving。它们是后续逐节学习任务，不能把第 1 节验证当作整个项目验证。
