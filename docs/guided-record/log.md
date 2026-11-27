# 引导采集实施记录

## Viewer #5：在录制进程内完成摆放、确认和进度回写

依据 [viewer #5](https://github.com/AtticusZeller/xense-lerobot-viewer/issues/5)、[总览 #6](https://github.com/AtticusZeller/xense-lerobot-viewer/issues/6) 的讨论结论，以及 [lerobot-check #2](https://github.com/AtticusZeller/lerobot-check/issues/2) 的范围简化。[补采计划 v1](https://github.com/AtticusZeller/lerobot-check/blob/3873cdf/docs/collection-plan.md) 定义身份和进度；[alignment PR #3](https://github.com/AtticusZeller/lerobot-check/pull/3) 定义按原生画幅检查 IoU 的接口，不判断物体身份、正反、平移或 yaw。目标是同一本地数据集追加；不改台架配方、机器人控制策略、`xense-robot-flexiv` 或 Hub 数据集。

### Task: 逐轮确认后录制，并将真实保存编号写回

**Change**

- [x] 顶层启用参数 `--collection_plan=/path/plan.json`；`--collection_alignment_thresholds=/path/thresholds.json` 只在引导模式有效，省略即用 alignment 默认阈值。启用时要求 `--resume=true`、`--dataset.push_to_hub=false`，不用当前默认的 Hub 上传。禁止仅在退出时批量编码视频，保证写回时有视频。无计划时原录制流程、键位及依赖均不变。
- [x] 从已合并的 lerobot-check `3873cdf` 原样复用标准库 `collection_progress.py`，记录出处（上游文档明确允许逐文件复用，仓库未声明通用许可证）；`plan_session` 覆盖启动校验至会话退出，按 `next_pending` 顺序执行。用本地单一 `.cache/huggingface/trees/<sha>.json` 推断 revision，与 lerobot-check 共用规则；无唯一树为 `local`，不拿计划的 revision 当证据。已有本地集才能启用，查不到文件则报错、不下载。按 v1 核对 repo ID、基准 revision、已有 episode 数、数据集图像 feature、头图实际序列号及原生尺寸。计划中的 `observation.images.head` 对应观测中的 `head`；Flexiv 配方配置的是 `robot.cameras.head`，从已连接 RealSense profile 读取真实序列号和画幅，而非 YAML 声明值，并检查未旋转的实际 RGB 帧。通过校验前不显示叠加、不录制。demo 计划真正要求序列号 `346522071766`，不能用仓库其他台架配方的序列号替代。
- [x] `record()` 为每个待采轮在录制前调用现有 `record_loop(dataset=None)`，包含首轮与重录同轮。`record_loop` 已经每帧调用 `robot.get_observation()`，Flexiv `get_observation()` 已使用连接过的相机 `async_read()`；仅向引导呈现层交接同一观测，不再取第二路相机，不改变发送控制指令的方式。普通录制循环没有引导回调。引导 reset 不因 `reset_time_s` 超时开始录制。窗口聚焦时 Enter 确认，是否达到阈值不限制确认。右箭头在 reset 不确认，只在录制保留提早结束并保存；左箭头在录制丢弃当前缓冲、回到同一待采轮，reset 中不会倒回已保存轮；Escape/关窗停止引导，当前待采轮不写回；Escape 在录制阶段沿用原版语义，已有帧则保存并写回后停止，丢弃用左箭头。原有机器人活动或单次推理尚未结束时不进入录制。计划无待采轮或达到本次 `dataset.num_episodes` 上限退出，不再开 reset。
- [x] 读取已经生成的 `dataset.episode_buffer["episode_index"]`，只在 `save_episode()` 成功并关闭两类 Parquet writer、写出 footer 之后 `mark_collected`。协调员要求先实测才改分片。基线实测：正常 finalize 时原版追加共享分片；成功调用 `save_episode()` 后用独立进程硬退，`info` 已报 3 条，但两轮数据的共享分片缺 footer，且没有两轮的元数据分片。因此只在引导模式先关闭并逐轮切换两个 Parquet 分片；普通录制保留聚合。保存、写回失败终止而不跳到下一轮；保存已成功但写回失败时打印 `plan_id/round_id/episode_index` 供人工修复。两文件系统操作不是同一事务，不自动猜测、补记或覆盖。

**Verification**

1. [x] 用假机器人和 teleop 测首轮 reset、右箭头不会确认、Escape、左箭头重录、校验失败、保存失败不写、保存后写真实非零编号、锁和计划耗尽。回归测试无计划路径。
2. [x] 将本地 `Xense/pack_iphone_box_1119` 复制到私有验收目录，从副本用真实 LeRobotDataset 追加；假机器人的头图从本地视频读，其余观测和动作由假设备提供。不改原导出、源数据集或 Hub。记录同一导出副本的两轮流程及一次重录；退出后用 PyArrow 核对进度编号、数据和元数据，确认计划中其余字段不变。

**Done**

- [x] 录屏中至少两次从 reset 确认进入录制，期间窗口关闭；输出更新后的 `collection-plan.json`、日志和两轮对应的真实 Parquet 行，给出绝对路径。

### Task: reset 时显示目标、实时判定和可人工确认的降级状态

**Change**

- [x] 使用 OpenCV HighGUI 独立窗口（主线程非阻塞事件泵，仅在 reset 中，复用现成 OpenCV），不使用 Rerun 实时录制会话。检查图形后端、显示环境和原有键盘监听；不能开窗时报错，不盲录。原生画幅坐标画目标蒙版轮廓、编号、锚点、箭头和观察轮廓，显示每物体 IoU/状态、当前轮号、已采/跳过/待采进度及 Enter 提示。无可靠轮廓的 `box_base` 画锚点与箭头并标为人工，不触发全到位。首次整体到位给视觉提示和一次 `log_say` 提示（尊重 `play_sounds`）；复用现有语音基础设施。
- [x] 分割是可选安装：录制器只懒加载 `lerobot_check.alignment`；在现有录制环境中按文档另行安装核验过的 lerobot-check 版本与 `mask-template` extra 及本地固定模型缓存，不改变基础 deps 或下发权重，不替换本 fork。按 #3 合并版使用 `AlignmentEvaluator.from_files`、`prepare_round`、`evaluate`。缺包/无 CUDA/缺权重/推理错误显示原因并降级；只有无掩膜目标的轮次不加载模型。损坏的计划、错误的图像身份与显式阈值配置错误不是降级。
- [x] 一个后台线程独占可复用的 evaluator，只在 reset 期间处理最新帧副本（积压最多一帧），结果按本轮隔离并携带采样时间；切换轮次清空，过时结果不能触发到位。首次准备和实时推理均不在控制或窗口线程上。确认停止投递后，继续泵界面及原有 teleop 控制，直到在途推理完成，再关窗开始录制；异常/退出在 finally 停止并回收该线程。整个录制期不提交、不推理、不渲染指南。

**Verification**

1. [x] 用阻塞的假 evaluator 验证 UI/控制继续、确认时等待在途任务、无积压、换轮不会串提示；用假结果验证成功提示、人工轮、降级与过期逻辑。
2. [x] 用 Xvfb、OpenCV GUI、xdotool 和 FFmpeg 真实按键/录屏；本机已存在 pack_iphone 的头部视频，已有 RTX 4060 和 SAM 固定权重缓存。测实际视频上的有分割与强制确认、另测未安装或无 GPU 的人工路径。不把新摆放位置回放成真实对齐正例，必要时用可控样本测试到位。验收日志说明真假设备、自动/人工行为以及清理进程结果。
3. [x] 核对本分支 diff、静态检查和本地录制相关测试。为台架操作员提供使用现场 `pack_iphone-taccap.yaml`、本地计划副本与本地数据集、自动判定和强制确认/重录的手工步骤及通过标准，不连接远端。PR 检查和评审状态以 GitHub 为准；不在此工作中合并。

**Done**

- [x] 录屏包含目标/实况轮廓、IoU 和确认过程；本地测试区分实际分割回放与手工降级；PR 明确记录真机仍待验证。

本机验收：RTX 4060 上运行真实 SAM 与归档头部视频，假 Flexiv 和假手柄未连接设备。用 GUI 按键走完重录及两次保存，对应 episode 121（150 帧）与 122（47 帧），核对了两组可读取的 Parquet 行、视频路径与计划其余字段不变。禁用 CUDA 的另一次回放显示人工模式，reset 中 Escape 未修改计划。全到位及一次语音调用用可控测试结果验证；台架听到语音和实物对齐仍待现场验证。
