# Idea 实验：具体回答对照

这是所有预测冻结并评分后的诊断。答案文字来自模型选择的原始选项，
不是模型自由生成的解释；不根据例子修改设置。银标签仅指一个参考事件，不能穷尽未来。

## 同一题集的结果

| 方法 | Validation | Test | Test准确率 |
|---|---:|---:|---:|
| full_ICL | 6/32 | 21/57 | 36.84% |
| recent_ICL | 4/32 | 23/57 | 40.35% |
| idea1_recent_only | 14/32 | 24/57 | 42.11% |
| idea1_residual | 14/32 | 24/57 | 42.11% |
| idea1_plain_delayed | 10/32 | 25/57 | 43.86% |
| idea1_reset_residual | 14/32 | 24/57 | 42.11% |
| idea3_direct | 8/32 | 9/57 | 15.79% |
| idea3_search | 6/32 | 6/57 | 10.53% |
| idea3_reset | 4/32 | 3/57 | 5.26% |
| idea3_wrong_direct | 7/32 | 10/57 | 17.54% |
| idea3_wrong_search | 2/32 | 9/57 | 15.79% |
| idea4_teacher | 14/32 | 24/57 | 42.11% |
| idea4_fixed_full | 14/32 | 24/57 | 42.11% |
| idea4_conditioned_full | 14/32 | 24/57 | 42.11% |
| idea4_fixed_warm20 | 14/32 | 24/57 | 42.11% |
| idea4_conditioned_warm20 | 14/32 | 24/57 | 42.11% |
| idea4_fixed_copy | 14/32 | 24/57 | 42.11% |
| idea4_fixed_oracle | 14/32 | 24/57 | 42.11% |
| idea5_low112 | 4/32 | 15/57 | 26.32% |
| idea5_mid224 | 6/32 | 21/57 | 36.84% |
| idea5_high448 | 8/32 | 25/57 | 43.86% |
| idea5_uniform_high | 9/32 | 16/57 | 28.07% |
| idea5_random_high | 7/32 | 17/57 | 29.82% |
| idea5_motion_high | 8/32 | 22/57 | 38.60% |
| idea5_deadline_high_posthoc | 10/32 | 24/57 | 42.11% |

方法预算与实现范围不同，不能直接把此表解释为架构排名。

## 具体题目

例子按配置中的配对比较，依次选字典序最早、尚未展示的改对/改错测试题，再补全部正确/错误例子。
这属于明确按结果选例的事后分析，完整200题见 `answers.jsonl` / `answers.csv`，配对ID见 `summary.json`。

### egolife_023（第6天）

选择原因：idea5_high448 改对 idea5_low112 的错误

问题：结合此前30分钟的历史视频和最近30秒视频，预测佩戴者在接下来的5秒内会做的一个具体动作。请选择一个最符合预测的选项。

**银标签参考：E．我接过了这个床单**

| 选项 | 原文 |
|---|---|
| A | 我拿起了一个抱枕 |
| B | 我整理了一下窗帘 |
| C | 我把被子叠整齐了 |
| D | 我整理了一下枕头 |
| E | 我接过了这个床单 |
| F | 我拿起了一个玩偶 |

| 方法 | 实际选择的答案 | 与参考一致 |
|---|---|---|
| full_ICL | D．我整理了一下枕头 | 否 |
| recent_ICL | E．我接过了这个床单 | 是 |
| idea1_residual | C．我把被子叠整齐了 | 否 |
| idea3_direct | A．我拿起了一个抱枕 | 否 |
| idea3_search | F．我拿起了一个玩偶 | 否 |
| idea4_conditioned_full | C．我把被子叠整齐了 | 否 |
| idea5_low112 | D．我整理了一下枕头 | 否 |
| idea5_mid224 | D．我整理了一下枕头 | 否 |
| idea5_high448 | E．我接过了这个床单 | 是 |
| idea5_uniform_high | D．我整理了一下枕头 | 否 |
| idea5_motion_high | E．我接过了这个床单 | 是 |
| idea5_deadline_high_posthoc | E．我接过了这个床单 | 是 |

### egolife_024（第6天）

选择原因：idea5_high448 改错 idea5_low112 的正确答案

问题：结合此前30分钟的历史视频和最近30秒视频，预测佩戴者在接下来的5秒内会做的一个具体动作。请选择一个最符合预测的选项。

**银标签参考：F．我打开刚才的黑色软件**

| 选项 | 原文 |
|---|---|
| A | 我拿起床边的手机 |
| B | 我摘下头上的耳机 |
| C | 我整理凌乱的被子 |
| D | 我拖动鼠标指针 |
| E | 我拿起桌上的水杯 |
| F | 我打开刚才的黑色软件 |

| 方法 | 实际选择的答案 | 与参考一致 |
|---|---|---|
| full_ICL | B．我摘下头上的耳机 | 否 |
| recent_ICL | D．我拖动鼠标指针 | 否 |
| idea1_residual | F．我打开刚才的黑色软件 | 是 |
| idea3_direct | B．我摘下头上的耳机 | 否 |
| idea3_search | A．我拿起床边的手机 | 否 |
| idea4_conditioned_full | F．我打开刚才的黑色软件 | 是 |
| idea5_low112 | F．我打开刚才的黑色软件 | 是 |
| idea5_mid224 | B．我摘下头上的耳机 | 否 |
| idea5_high448 | B．我摘下头上的耳机 | 否 |
| idea5_uniform_high | B．我摘下头上的耳机 | 否 |
| idea5_motion_high | B．我摘下头上的耳机 | 否 |
| idea5_deadline_high_posthoc | F．我打开刚才的黑色软件 | 是 |

### egolife_027（第7天）

选择原因：idea5_motion_high 改对 idea5_uniform_high 的错误

问题：结合此前30分钟的历史视频和最近30秒视频，预测佩戴者在接下来的5秒内会做的一个具体动作。请选择一个最符合预测的选项。

**银标签参考：B．我拿起另一个杯子**

| 选项 | 原文 |
|---|---|
| A | 我打开一瓶饮料 |
| B | 我拿起另一个杯子 |
| C | 我拿起一瓶酱油 |
| D | 我拿起一块饼干 |
| E | 我拿起一把勺子 |
| F | 我拿起一块奶酪 |

| 方法 | 实际选择的答案 | 与参考一致 |
|---|---|---|
| full_ICL | B．我拿起另一个杯子 | 是 |
| recent_ICL | B．我拿起另一个杯子 | 是 |
| idea1_residual | C．我拿起一瓶酱油 | 否 |
| idea3_direct | E．我拿起一把勺子 | 否 |
| idea3_search | A．我打开一瓶饮料 | 否 |
| idea4_conditioned_full | C．我拿起一瓶酱油 | 否 |
| idea5_low112 | C．我拿起一瓶酱油 | 否 |
| idea5_mid224 | B．我拿起另一个杯子 | 是 |
| idea5_high448 | B．我拿起另一个杯子 | 是 |
| idea5_uniform_high | C．我拿起一瓶酱油 | 否 |
| idea5_motion_high | B．我拿起另一个杯子 | 是 |
| idea5_deadline_high_posthoc | B．我拿起另一个杯子 | 是 |

### egolife_093（第6天）

选择原因：idea5_motion_high 改错 idea5_uniform_high 的正确答案

问题：结合此前30分钟的历史视频和最近30秒视频，预测佩戴者在接下来的5秒内会做的一个具体动作。请选择一个最符合预测的选项。

**银标签参考：D．我喝了一口矿泉水**

| 选项 | 原文 |
|---|---|
| A | 我擦了擦眼镜片 |
| B | 我拿起了桌上的书 |
| C | 我调整了一下耳机 |
| D | 我喝了一口矿泉水 |
| E | 我拍了一下地板 |
| F | 我放下手中的水瓶 |

| 方法 | 实际选择的答案 | 与参考一致 |
|---|---|---|
| full_ICL | D．我喝了一口矿泉水 | 是 |
| recent_ICL | F．我放下手中的水瓶 | 否 |
| idea1_residual | C．我调整了一下耳机 | 否 |
| idea3_direct | B．我拿起了桌上的书 | 否 |
| idea3_search | B．我拿起了桌上的书 | 否 |
| idea4_conditioned_full | C．我调整了一下耳机 | 否 |
| idea5_low112 | C．我调整了一下耳机 | 否 |
| idea5_mid224 | D．我喝了一口矿泉水 | 是 |
| idea5_high448 | D．我喝了一口矿泉水 | 是 |
| idea5_uniform_high | D．我喝了一口矿泉水 | 是 |
| idea5_motion_high | C．我调整了一下耳机 | 否 |
| idea5_deadline_high_posthoc | D．我喝了一口矿泉水 | 是 |

### egolife_090（第6天）

选择原因：idea3_search 改对 idea3_direct 的错误

问题：结合此前30分钟的历史视频和最近30秒视频，预测佩戴者在接下来的5秒内会做的一个具体动作。请选择一个最符合预测的选项。

**银标签参考：F．我接过了快递**

| 选项 | 原文 |
|---|---|
| A | 我举起双手示意 |
| B | 我把包裹放回去了 |
| C | 我拿着手机走了 |
| D | 我走向对面的房子 |
| E | 我坐在椅子上等了 |
| F | 我接过了快递 |

| 方法 | 实际选择的答案 | 与参考一致 |
|---|---|---|
| full_ICL | F．我接过了快递 | 是 |
| recent_ICL | F．我接过了快递 | 是 |
| idea1_residual | B．我把包裹放回去了 | 否 |
| idea3_direct | B．我把包裹放回去了 | 否 |
| idea3_search | F．我接过了快递 | 是 |
| idea4_conditioned_full | B．我把包裹放回去了 | 否 |
| idea5_low112 | F．我接过了快递 | 是 |
| idea5_mid224 | F．我接过了快递 | 是 |
| idea5_high448 | F．我接过了快递 | 是 |
| idea5_uniform_high | F．我接过了快递 | 是 |
| idea5_motion_high | F．我接过了快递 | 是 |
| idea5_deadline_high_posthoc | F．我接过了快递 | 是 |

### egolife_058（第6天）

选择原因：idea3_search 改错 idea3_direct 的正确答案

问题：结合此前30分钟的历史视频和最近30秒视频，预测佩戴者在接下来的5秒内会做的一个具体动作。请选择一个最符合预测的选项。

**银标签参考：C．我打开了盖子**

| 选项 | 原文 |
|---|---|
| A | 我喝了一口水 |
| B | 我站起来吃饭 |
| C | 我打开了盖子 |
| D | 我转身离开了 |
| E | 我站起来走了 |
| F | 我擦了擦嘴 |

| 方法 | 实际选择的答案 | 与参考一致 |
|---|---|---|
| full_ICL | B．我站起来吃饭 | 否 |
| recent_ICL | C．我打开了盖子 | 是 |
| idea1_residual | D．我转身离开了 | 否 |
| idea3_direct | C．我打开了盖子 | 是 |
| idea3_search | A．我喝了一口水 | 否 |
| idea4_conditioned_full | D．我转身离开了 | 否 |
| idea5_low112 | B．我站起来吃饭 | 否 |
| idea5_mid224 | B．我站起来吃饭 | 否 |
| idea5_high448 | B．我站起来吃饭 | 否 |
| idea5_uniform_high | B．我站起来吃饭 | 否 |
| idea5_motion_high | B．我站起来吃饭 | 否 |
| idea5_deadline_high_posthoc | B．我站起来吃饭 | 否 |

### egolife_026（第7天）

选择原因：全部展示方法均错误

问题：结合此前30分钟的历史视频和最近30秒视频，预测佩戴者在接下来的5秒内会做的一个具体动作。请选择一个最符合预测的选项。

**银标签参考：F．我点击手机屏幕呼出键盘**

| 选项 | 原文 |
|---|---|
| A | 我触摸手机屏幕确认地址 |
| B | 我点击手机屏幕发送消息 |
| C | 我操作手机屏幕切换应用 |
| D | 我长按手机屏幕刷新页面 |
| E | 我抬起手机查看导航路线 |
| F | 我点击手机屏幕呼出键盘 |

| 方法 | 实际选择的答案 | 与参考一致 |
|---|---|---|
| full_ICL | A．我触摸手机屏幕确认地址 | 否 |
| recent_ICL | A．我触摸手机屏幕确认地址 | 否 |
| idea1_residual | D．我长按手机屏幕刷新页面 | 否 |
| idea3_direct | B．我点击手机屏幕发送消息 | 否 |
| idea3_search | B．我点击手机屏幕发送消息 | 否 |
| idea4_conditioned_full | D．我长按手机屏幕刷新页面 | 否 |
| idea5_low112 | A．我触摸手机屏幕确认地址 | 否 |
| idea5_mid224 | A．我触摸手机屏幕确认地址 | 否 |
| idea5_high448 | C．我操作手机屏幕切换应用 | 否 |
| idea5_uniform_high | C．我操作手机屏幕切换应用 | 否 |
| idea5_motion_high | C．我操作手机屏幕切换应用 | 否 |
| idea5_deadline_high_posthoc | C．我操作手机屏幕切换应用 | 否 |

### egolife_022（第6天）

选择原因：full_ICL 正确的补充例子

问题：结合此前30分钟的历史视频和最近30秒视频，预测佩戴者在接下来的5秒内会做的一个具体动作。请选择一个最符合预测的选项。

**银标签参考：C．我喝了一口酸奶**

| 选项 | 原文 |
|---|---|
| A | 我喝了一大口牛奶 |
| B | 我把酸奶倒在地上 |
| C | 我喝了一口酸奶 |
| D | 我闻了闻这瓶饮料 |
| E | 我打开了冰箱门 |
| F | 我随手扔掉了瓶子 |

| 方法 | 实际选择的答案 | 与参考一致 |
|---|---|---|
| full_ICL | C．我喝了一口酸奶 | 是 |
| recent_ICL | C．我喝了一口酸奶 | 是 |
| idea1_residual | B．我把酸奶倒在地上 | 否 |
| idea3_direct | E．我打开了冰箱门 | 否 |
| idea3_search | A．我喝了一大口牛奶 | 否 |
| idea4_conditioned_full | B．我把酸奶倒在地上 | 否 |
| idea5_low112 | C．我喝了一口酸奶 | 是 |
| idea5_mid224 | C．我喝了一口酸奶 | 是 |
| idea5_high448 | C．我喝了一口酸奶 | 是 |
| idea5_uniform_high | C．我喝了一口酸奶 | 是 |
| idea5_motion_high | C．我喝了一口酸奶 | 是 |
| idea5_deadline_high_posthoc | C．我喝了一口酸奶 | 是 |
