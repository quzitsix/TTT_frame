# IdeaX 当前研究输入：RL / TTT forecasting

本轮完整输入：**[ideax_rl_input_20261031.md](ideax_rl_input_20261031.md)**。
请读取该文件全文后生成研究提案，输出到 `ideax_rl_output_20261031/`，保留旧 `ideax_output/`。

输入已纳入五个 Idea 的实测结果，以及新服务器的 EgoLife、NextMe、200/185 条
benchmark、context VQA 与 20-case SFT/GRPO 数据位置和审计要求。
重点是 RL 控制细节获取、写入与延迟反馈；RL 尚无已验证收益。

- [新服务器迁移与继续命令](docs/NEW_SERVER_RL_HANDOFF_20261031.md)
- [五方向实验总结与具体回答](docs/IDEAX_EXPERIMENT_SUMMARY_20261026.md)
- [迁移前输入原文备份](docs/archive/ideax_input.pre_migration_20261031.md)

迁移前原文 SHA256：`c5c7396fe7eeffe3715f31f138ab773a41dda92d9aaeb510200b35c5aa5cf8b0`。
本次只更新研究与迁移输入，不启动新的 RL 训练，不改变旧冻结实验源码。
