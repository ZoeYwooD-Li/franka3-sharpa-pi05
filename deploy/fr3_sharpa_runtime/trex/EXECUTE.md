# 同机双臂双手 T-Rex all 执行

工程：`/home/descfly/LTT/franka3-sharpa-trex-all`。
权重：`/home/descfly/桌面/fanghao_Trex_all_6250`。只跑 all。
现有相机、FR3 控制器与 Safety Gateway 继续使用原硬件 Docker；模型 server 和执行进程运行于本机已有 Python 环境。执行器不会启动另一套机器人控制器，不修改已有环境、权重或 π0.5 工程。

## 模型服务

```bash
cd /home/descfly/LTT/franka3-sharpa-trex-all
bash deploy/fr3_sharpa_runtime/trex/local.sh serve
```

一个 server 即可，监听 `127.0.0.1:5555`。看到 listening 后启动执行进程。已有新版服务无需重复启动。旧版需先 Ctrl-C 后重启，因为执行器要求新版 exclusive_sessions 握手。执行期间不要另开 probe/shadow。

每次准备执行前，可先在另一个终端做只读握手检查：

```bash
cd /home/descfly/LTT/franka3-sharpa-trex-all
bash deploy/fr3_sharpa_runtime/trex/local.sh health
```

`health` 只发送 `hello`，核对已预热、权重路径、16×62动作形状、图像尺寸、
URDF哈希和独占会话支持；不会导入ROS/手部SDK、占用模型会话、运行推理或发送
运动命令。成功输出 `status=PASS`，失败返回退出码1。也可使用
`local.sh execute --check-server`。执行检查通过仍不代表动作可达或任务成功。

`health`不检查ROS网关或实时传感器。可用同一执行环境检查这些依赖：

```bash
bash deploy/fr3_sharpa_runtime/trex/local.sh doctor --duration 30 --output /tmp/trex-doctor.json
```

`doctor`只订阅现有相机/双臂状态，并读取网关图和准入参数、检查模型握手；不会连接或使能双手、创建命令发布者、运行模型推理。报告明确标记`published=false`和`joint_targets_sent=0`。启动脚本在加载ROS消息环境前保存ROS发现配置，避免环境hook把未设置的`ROS_LOCALHOST_ONLY`补成0而与网关的1不一致。

T-Rex专用相机配置关闭未使用的cam0深度流，保留RGB尺寸、帧率、三路相机身份及原始采集时间戳。此配置通过同一`openpi-cameras`服务加载，不修改π0.5工程或原相机配置：

```bash
bash deploy/fr3_sharpa_runtime/trex/cameras.sh check
bash deploy/fr3_sharpa_runtime/trex/cameras.sh up
```

上述`up`只更新现有相机服务，不启动/重启机器人控制器。切回π0.5相机配置用`cameras.sh restore`；π0.5原`hardware.sh up`也会加载原相机配置。配置是否改善延迟以现场检查报告为准，执行中的200ms门限继续生效。

原生触觉缓冲区按16位串行帧号处理65535→0回绕，原始帧号与时间戳保留在记录中。重复/倒序帧仍丢弃；真实时间间隔超过75ms时重建历史，禁止补帧，新鲜度和十指同步门限不变。`frame_id_rollovers`记录接受的回绕次数。

## 右手两个侧摆关节的受限控制

小幅手部预测越界可以显式增加`--prediction-clip-rad 0.05`（默认0严格拒绝，最大0.05rad）。参考π0.5，仅将模型手部目标投影回既有软件范围，逐值修正超过此额度或机械臂越界仍停止；不修改Home或实测反馈，实际下发仍检查全部限位和跟随误差。被屏蔽的两个关节先使用合法实测基准进行预测检查，再由执行器替换为固定锚点，不会仅因它们不下发的模型目标越界而退出。

`raw`及`decoded_model`保留原始预测，`decoded`保留实际使用的解码；报告`prediction_clipping`记录修正次数及最近256次修正审计。越界失败会在退出清理后保存`rejected-prediction.npz`，包含raw/decoded/decoded_model/base，报告及runtime.log写出关节名、预测行号、值、上下限和超出幅度。该选项只处理有界的小幅手部预测越界，不代表模型任务已正确完成。

触觉主机接收抖动可显式用`--tactile-host-gap-ms 85`将主机历史连续性门限改为85ms（允许75..100ms，默认仍75ms）。只改变主机间隔；SDK源间隔仍75ms、最新数据年龄仍200ms、十指跨度仍100ms，仍需要16个真实帧，不补帧。采集、历史对齐和快照检查使用同一主机门限，报告保存`tactile_timing_limits`及实际间隔。更大的真实源间断仍停止。

对于运行中正向时间间断引起的历史重建，可以显式增加`--tactile-recovery-s 3`（默认0关闭，最大3秒）：丢弃中断的模型动作块，持续发送最后已发送的位置并逐帧检查真实关节反馈、限位、跟随误差及发送时序。只有十指均有16个中断后真实原始帧、完整观测重新通过检查，且旧推理任务已结束后，才从当前实测状态发起新slow/fast推理；不会续用旧chunk、旧fast结果或补帧。

该限时恢复也覆盖“所有最新帧仍在200ms新鲜度内，但十指最新SDK/接收时间跨度暂时超过100ms”的情形。跨度门限保持100ms；超限数据不用于模型，暂停后仍须重新积累十指中断后的真实历史并重新推理。源数据过期、未来接收时间戳等其他异常不会因此自动恢复。

超时、关节反馈异常、跟随超差、SDK采集线程故障均停止并走失能确认流程。60秒恢复次数默认最多3次，可显式使用`--tactile-max-recoveries 10`增加到10次（允许1..20）。无论次数配置如何，每60秒的累计恢复保持时间上限仍为10秒，超过也停止。被中断块不计为完成，原预测文件保留并使用独立安装序号避免覆盖。报告`tactile_recoveries`记录原因、被中断的chunk/row、原预测文件及恢复状态。此选项处理短暂采集间断，不修复持续丢帧或设备故障。

显式增加`--hold-right-ring-pinky-aa`，可在模型阶段屏蔽右手`ring_MCP_AA`和`pinky_MCP_AA`的模型目标。这是受限抓取模式，不代表关节修复或理想反馈。可选Home仍按完整原目标执行；Home后第一次完整观测就绪时记录这两个关节的真实角度，整次执行的slow、fast及APPROACH目标均保持这两个固定值。未指定Home时，以首次策略观测就绪时的实测角度为准。

真实关节反馈、图像、触觉和FK基准不修改，所有58关节的限位、新鲜度、跟随误差及速度检查继续生效；其他20个右手关节及全部左手/双臂目标保持原模型输出。该选项只允许用于`--execute`，不改变`--home-only`语义。

报告增加`degraded_execution=true`及`policy_joint_hold`，记录固定角度和最近256次预测中被覆盖的原始目标。NPZ的`raw`保留原始16×62输出，`decoded_model`保留覆盖前16×58解码，`decoded`记录实际使用的受限目标；逐帧记录中的`measured`仍是真实读数。成功下发的完成类型为`commands_sent_with_joint_hold`，不表示原模型姿态或物体任务完成。去掉该选项即可恢复完整模型控制，不写入持久SDK配置。

服务未启动、尚未完成预热或没有及时响应时，3秒超时将明确显示请求类型
（如`hello`）、send/receive阶段、端点及等待时间，不再仅显示ZMQ的
`Resource temporarily unavailable`。确认服务终端打印listening后再检查；
不要因该超时重启硬件容器。失效的REQ连接会被丢弃，不自动重发claim/slow/fast，
避免重复改变模型缓存；若claim已被服务接收但回复丢失，租约可能保留到原120秒
期限。运行中的模型请求超时仍停止并按既有流程退出，不放宽观测新鲜度或控制保护。

## 无下发检查

```bash
cd /home/descfly/LTT/franka3-sharpa-trex-all
bash deploy/fr3_sharpa_runtime/trex/local.sh execute --plan
bash deploy/fr3_sharpa_runtime/trex/local.sh execute --offline-replay \
  /home/descfly/文档/ChatGPT/部署/trex_live_validation/after-hand-home-002.npz
```

离线回放验证真实模型 slow_and_fast、fast、两次 IK 和关节范围，不初始化 ROS 或手部 SDK。重复使用记录触觉仅验证协议，不能代表在线闭环验收。

## 独立双臂双手 Home

仅回位时运行以下命令，它会使双臂双手运动：

```bash
cd /home/descfly/LTT/franka3-sharpa-trex-all
ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
CYCLONEDDS_URI=file:///home/descfly/ZH/gello_upper_body_teleop/config/cyclonedds.xml \
bash deploy/fr3_sharpa_runtime/trex/local.sh execute --home-only --enable-hardware
```

该模式沿用既有 powder_weighing 双臂 Home 和 hand_home.json，以0.03rad/s、30Hz
回位；全部实测在0.02rad容差内、命令到目标并稳定0.5s后退出。
不校验模型权重、不连接模型服务、不运行推理、不进入APPROACH或RUN。
仍检查SDK独占、真实关节反馈、Safety Gateway、速度、范围、停滞和接续保护，
并使用相同执行锁；没有启动额外机器人控制器。已有硬件容器需要保持运行。
正常完成报告包含 `mode=home_only`、`home_completed=true`、`status=PASS`；
policy_frames_sent和prepare_frames_sent均为0。退出确认双手失能并断开，
不会持续保持手部Home；机械臂目标发布停止后由既有控制器/watchdog处理。

## 当前现场首次运行

机械臂 FCI、现有硬件控制容器和三路相机需要正常运行；工作空间留空，现场人员可随时停止。以下命令会使双臂双手运动。

`--home` 使用与 π0.5 相同的 powder_weighing Home。默认任务对齐π0.5现场记录：右手拿番茄串、左手摘取小番茄并放入左篮；其他任务通过 `--prompt '实际英文任务描述'` 指定，现场物体必须与任务相符。报告保存实际prompt。

模型执行在连接双手、发送 Home 前先检查三路相机：原始源时间戳与接收年龄均须在原有200ms门限内，时间戳不能在未来，历史帧率须达标并持续通过2秒。最多等待60秒（仅启动准备时间，非传感器年龄门限），失败记录为`CAMERA_PREFLIGHT`并保持手部未连接、运动帧数为0。单独`--home-only`不依赖相机。该检查不修改时间戳；执行中的新鲜度及同步检查仍然生效。若相机时钟异常，可只重启`docker-openpi-cameras-1`，不必重启机器人控制器；重启后仍须重新检查，不能据此认定完整任务已通过。

如果独立 Home 保持进程仍运行，首次命令使用接管参数。2026-10-27 21:25 的执行已接管并停止了该进程，随后退出并确认双手失能；该现场状态的下一次启动必须省略接管参数：

```bash
cd /home/descfly/LTT/franka3-sharpa-trex-all
bash deploy/fr3_sharpa_runtime/trex/local.sh execute \
  --execute --enable-hardware \
  --home --approach-first-target \
  --num-chunks 1 --speed slow
```

接管验证 Home 哈希、报告新鲜度和进程身份，只停止该 Home 保持进程，确认双手失能并断开后再连接 SDK。过期报告、进程不符或断开未确认会拒绝启动。没有运行 Home 保持进程的后续启动，删除 `--takeover-home ...json` 两项。

Home 与显式首目标接续的目标速度为 0.03 rad/s。首目标相对实测超过臂 0.05 rad / 手 0.15 rad 时，`--approach-first-target` 允许缓慢接近预测目标。目标进入原接续门限并稳定0.5s后丢弃旧预测、重新观测推理，最多三次；新结果仍须通过全部安装/接续检查。该选项本身会导致运动；不加此选项时超差直接停止。

模型目标速度上限 0.2 rad/s；slow 将 16 行动作按 0.1 倍时间速度发送。每四个 30 Hz 周期尝试一次触觉 fast 更新，仅替换未发送行并保留原 FK 基准。观测断流、关节越界、接续超差、网关拒绝或发送超时会停止。

`--num-chunks 1` 一个动作块后退出，不代表完整任务完成。单块现场验收后，需要持续执行时改成 `--num-chunks 0`，并根据是否仍有 Home 保持进程决定接管参数。

## 停止与记录

执行端默认采用简洁终端输出：启动及日志位置、当前chunk/总chunk（0显示无限）、结束/失败摘要。完整Python输出、SDK原生stdout/stderr、每秒进度和traceback保存到本次目录的`runtime.log`；`progress.json`仍保留最新结构化状态，`report.json`和`frames.jsonl`继续保存。状态输出由后台进度线程处理，不在30Hz运动循环等待终端。修改代码不会改变已运行进程的输出方式，下次启动执行器生效。

终端 Ctrl-C 停止目标发布，机械臂网关按既有 watchdog 处理；执行器确认双手失能并断开。失能后手指可能回弹，不保持最后姿态。紧急情况使用现场急停。

默认记录目录 `trex-local-records/execute-时间-PID/`：`frames.jsonl` 保存命令与实测，`chunk-*.npz` 保存安装的原始模型输出、IK 和固定基准。被用于APPROACH的完整预测在退出清理后保存为`prepare-*.npz`（raw/decoded/base/原始时间戳），不在命令循环压缩写盘。`report.json` 保存完成块数及双手退出验证。策略块PASS的completion为commands_sent，不是所有关节精确到位或物体任务成功判定。

连接初始化若在get_speed_coeff/get_current_coeff/get_control_source/get_control_mode
读取阶段失败，且本次没有任何一只手连接成功、所有原生句柄均已关闭，最多重试3次，
间隔1s；其他错误或部分连接成功时立即停止，不重试使能或start错误。
每次连接尝试保存于shutdown.connection_attempts。

若右手初始化失败，保留已成功连接的左手句柄，用于既有失能/读回/断开流程。
SDK在失败初始化后已经销毁的句柄无法读回：直接标记UNCONFIRMED、attempts=0，
不做120次无效重试，也不把断连当作失能证明。有效句柄的120s/120次确认期限不变。
shutdown FAIL/UNCONFIRMED时不能依据本程序宣称设备已失能或安全释放。

已实际发生模型首目标的低速接近运动；尚未完成16行动作块的真机运动验收。

## 准备进度与反馈停滞诊断

执行器在阶段切换和每秒输出 `[trex-progress]`，同一内容原子更新到记录目录的
`progress.json`。终端和进度文件由后台线程输出，30Hz 命令循环只提交最新状态，
不会等待终端或进度文件写入。`report.json` 仍是退出后的完整结果。

HOME / APPROACH 进度包含耗时、期限、最大目标误差、未到位关节数及误差最大的
五个关节；逐关节列出名称、最终目标、当前命令、实测、命令偏差、目标偏差和
SDK 声明限位。`stable` 仅表示相邻反馈变化不超过0.005rad；`at_target` 表示
全部实测与最终目标相差不超过0.02rad。HOME仍须命令到目标、实测到位并稳定0.5s；
APPROACH的completion_criterion为policy_acquisition_bounds，只要求目标进入
原臂0.05rad/手0.15rad门限并稳定0.5s，然后重采。prepare_completion保存交接状态。

准备阶段新增提前停止：同一个关节在至少连续2s内，始终命令跟随偏差>0.03rad、
目标误差>0.02rad，且整个窗口实测范围<=0.005rad，记录为反馈停滞。HOME停滞仍
停止；APPROACH中仅当停滞关节的目标误差仍超出原臂0.05rad/手0.15rad接续门限时
作为blocking_stalled_joints停止。门限内偏差仍记录，不当作Home到位要求。
它不把停滞位置当作物理硬限位。逐帧接续保护、速度、SDK范围、HOME容差均不变。

`prepare_target` 和 `prepare_chunk_id` 保留最近一次首目标接近使用的预测。
停滞时 `prepare_progress.stalled_joints` 保存关节、命令/目标/实测、窗口时长和
实测变化范围。逐帧保护报错包含所有超差关节；被保护拒绝的候选保存在
`rejected_frame`，标记 `sent=false`，不混入 `frames.jsonl` 的已发送帧。
WAIT_SENSORS / PLAN 进度显示最近观测等待原因、观测重试及推理忙状态。

2026-10-27 日志中的右手 ring_MCP_AA / pinky_MCP_AA 反馈停在约-0.04rad附近，
本机 SDK 的这两个关节声明范围为±0.3491rad；现有记录不足以确认真实硬限位，
本次没有据此改限位、裁剪目标或放宽保护。

## 21:25 推理期限修复

slow 的视觉上下文先按原 200ms 采样新鲜度检查采集，完成 slow 后再从原生缓冲区
读取最新触觉并做一次同块 fast，仅对该结果解 IK 后接入。原视觉上下文时间戳
不会被新触觉覆盖。安装时分别检查上下文总年龄 <=750ms、最新触觉结果年龄
<=200ms；实时相机、触觉和关节反馈仍使用原 200ms 断流检查。750ms 是独立的
slow 上下文处理期限，不表示允许接收 750ms 的传感器旧帧。

无损 PNG 使用零压缩级别，避免慢编码；RGB 像素和输入分辨率不变。
report.json 的 last_prediction_timing 会记录 context_age_ms、tactile_age_ms 和
各阶段耗时，过期三次仍然停止。chunk 文件保存两种原始时间基准。

如果在该次退出后重试，使用 --home，省略 --takeover-home，模型 server 无需重启。


## 21:32 触觉采集与组帧修复

原采集绑定 30Hz 命令循环，等待慢手指时还会覆盖快手指未使用的帧。
现在 NativeSampler 是唯一 SDK 读取者，以约200Hz独立轮询；30Hz 主循环只读取
缓冲区并下发命令。退出先停止并 join 采集线程，再执行原 π0.5 失能/读回/断开。

每个物理手指独立保留最多64个原始样本，模型使用各指16个不同原始帧。
采用与既有 ROS 触觉缓冲区相同的逐指历史语义，保留 SDK 时间戳和帧号。
SDK/主机相邻间隔仍限75ms，十指每行跨度仍限100ms，最新采样仍限200ms；
真正断流或超限会清除受影响手指历史并拒绝模型输入，不能重复或补零。

2026-10-27 带载只读验证：真实十指、双臂反馈、三路相机和模型持续30秒，
900次传感器检查、30次slow及146次独立fast结果通过，触觉历史重置0次。
最大 SDK 相邻间隔65.5ms，最大模型上下文年龄529.5ms，触觉结果年龄155.5ms。
双手全程失能，设置前后一致，没有机械臂命令发布者或关节目标下发。
63项测试通过。该结果验证采集和推理稳定性，不等于运动或任务成功验收。


## 对照 π0.5 后的阶段与时序修正

Home 与模型首目标接近沿用 π0.5 SlowApproach 的直线关节步进：0.03rad/s，
Home到位容差0.02rad；模型接近使用原臂0.05rad/手0.15rad接续门限。
稳定变化0.005rad并保持0.5s，按旅行距离设置至少120s期限。
准备阶段始终检查真实关节反馈、网关、范围和发送连续性；满足对应条件后进入
WAIT_SENSORS，丢弃准备使用的旧预测并采集完整新观测。准备阶段不消费触觉模型
输入，因此不以整段触觉历史作为继续回位的条件。

slow构建输入时检查三路RGB和触觉，fast构建输入时检查触觉；RUN每个周期检查
实时关节、触觉和网关。fast使用slow缓存的视觉上下文，不把当前RGB作为fast
输入。所有被消费的传感器输入仍须满足原有200ms新鲜度和相关连续性上限。
暂时无法组成有效观测会丢弃该次输入；规划观测连续不可用3秒或规划整体5秒
没有可安装结果会停止，真正关节断流或RUN触觉失效仍停止。

触觉历史若各取末尾16帧后横向跨度超100ms，会在原始历史内做有序时间匹配，
约束每行SDK/主机跨度<=100ms、每指相邻间隔<=75ms、帧号严格前进，并固定使用
最新可用末帧。不能满足时拒绝输入。不会重复、插值、补零或修改时间戳。

报告新增 home_frames_sent、prepare_frames_sent、policy_frames_sent 和 last_phase；
policy_frames_sent=0不代表没有模型引导的首目标接近运动。
tactile-timing.npz 保存最近4096个原始帧的手指路由、SDK帧号/时间戳和精确主机
接收时钟，支持后续失败回放。不要把退出停采后的样本年龄当作运行时年龄。

最终验证：70项测试；100组异步缺帧模式验证不复用原始帧且不扩大阈值；
Home→首目标接近→重新采集→单块执行的模拟流程通过。
真实只读带载60秒通过1800次实时检查、60个slow和290个fast结果，1次组合观测
不可用被丢弃并重采。双手始终失能、设置前后一致，未发送关节目标。
这仍不能替代有运动时的动作块验收。模型server不需重启，启动参数不变。

### 逆运动学朝向精度

T-Rex 名义 link8 逆解默认要求位置残差不超过 0.002 m（2 mm）、朝向残差不超过 0.02 rad（约 1.146°）。朝向残差是目标与解算姿态之间相对旋转的角度，不是关节实测跟随误差。2026-10-29 应用户要求，朝向容差调整到 0.02 rad，随后位置容差由 1 mm 调整到 2 mm；关节限位、0.2 rad 解码跳变检查与执行跟随检查保持不变。此调整不保证不可达目标或跳变预测能够通过，也不验证接触/摘取效果。新启动的执行进程使用新值，既有进程保持已加载代码。离线复现工具仍使用失败记录中保存的原始门限。

### 逆解失败诊断与离线复现

新启动的执行器遇到 IKError 时，会先保留后台线程中的失败证据，完成硬件清理后
将其写入运行目录的 `ik-failure.json`，并在 `report.json` 的 `ik_failure` 字段关联。
记录包括原始 16×62 动作、58 维固定基准状态、chunk ID、slow/fast 类型和采样时间。
对求解结束后的残差失败，另存失败侧别、从 0 开始的动作行、绝对目标矩阵、
该行求解种子、最终关节解和 FK、float64 精度残差、门限、迭代数、关节限位余量、
URDF SHA256，以及 `line_search_no_improvement` 或 `iteration_limit` 停止原因。
求解前校验或解码跳变错误仍保留原始动作和基准状态，但没有残差诊断。

纯离线复现（无需 ROS、模型服务或手部 SDK）：

```bash
cd /home/descfly/LTT/franka3-sharpa-trex-all
.venv-trex/bin/python -m deploy.fr3_sharpa_runtime.trex.replay_ik /绝对路径/运行目录/ik-failure.json
```

工具核对 URDF 哈希，并分别重算目标/种子逆解和原始整段动作解码。
`reproduced=true` 表示两条路径均匹配，退出码为 0；不匹配返回 1。
URDF 不匹配或没有残差诊断时拒绝复现。此变更保持求解算法和验收门限不变；
保存证据不代表失败动作获准下发。旧日志缺失的失败动作无法追溯补齐。

### 运行时三路相机与十指触觉预览

执行命令增加 `--monitor-port 8765`，浏览器打开 `http://127.0.0.1:8765/`。默认0关闭；端口必须1024..65535且不能与推理端口相同。页面显示cam0/cam1/cam2、左右手各五指240×240灰度形变和F6（Fx/Fy/Fz/Tx/Ty/Tz原始SDK值），以及阶段、当前/已完成/总chunk。监视器只绑定本机，执行结束随进程关闭。需要已有执行进程退出后用新参数启动，无法附加到已经运行的旧进程。

监视器复用本次SensorStore现有缓存，不开SDK连接、不读取或消耗SDK队列、不增加ROS订阅/发布者、不调用模型snapshot；无控制接口。独立线程4Hz预览，最多640×480相机JPEG，获取缓存锁不成功就跳过，编码在锁外执行。各卡片保留原始时间戳/接收年龄；显示各路最新帧，非模型严格对齐输入。超过200ms显示过期，HTTP断开或后台不更新显式提示；预览频率与传输也会增加显示年龄，不代表控制端最新样本同样过期。默认仅预览；增加`--monitor-record`可保存约4Hz观测回放，并非全部原始帧录像。原有控制检查不变。

离线测试覆盖RGB/BGR行padding、10指真实缓存不消费、锁忙跳过、过期样本年龄、HTTP缓存禁用和服务关闭、无效端口。模拟页面已浏览器检查；未运行带预览的真机运动，后台编码仍占少量CPU，应以现场30Hz执行日志为准。

### 观测保存与回放、预览端口冲突

`--monitor-port 8765 --monitor-record`同时启用预览和保存。8765已被占用时自动绑定本机空闲端口，终端打印实际URL；不会结束占用者。仅指定`--monitor-record`也会保存并分配本机预览端口，不必打开网页。执行器不依赖浏览器是否连接来保存。新增参数仅在下一次启动生效。

本次目录`sensor-preview.jsonl`约4Hz逐行保存3路相机JPEG（最多640×480）、十指240×240形变PNG（无损）、F6、原始源时间戳/接收年龄、阶段chunk等。每帧刷新文件缓冲，异常退出通常保留此前完整行；回放忽略未写完的尾行。最大2GiB/轮，达到上限或写盘失败只停止录制，预览和控制检查不变；页面显示录制错误，最终report.json及终端给出录制信息。记录属于低帧率观测副本，不能重建原始全帧触觉历史或用于驱动机器人。

结束后直接回放最近保存的数据（不连接ROS或SDK）：

```bash
cd /home/descfly/LTT/franka3-sharpa-trex-all
bash deploy/fr3_sharpa_runtime/trex/local.sh replay --latest
```

或`local.sh replay --run-dir /绝对路径/execute-时间-进程号`。打开命令打印的实际URL（优先8766，冲突自动换端口），播放/暂停或拖动进度条；按约4Hz顺序播放，各帧保留实际时间及样本年龄。回放只按需读当前帧，不把所有图像加载到内存。Ctrl-C结束回放服务，不影响记录文件。历史未启用保存的运行不能补回完整相机/触觉回放。
