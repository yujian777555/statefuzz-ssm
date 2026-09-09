# Codex结果报告协议

每轮执行在本目录生成一个JSON结果文件，文件名采用
`result_round_<三位轮次>.json`。结果文件必须使用
`statefuzz.io_atomic.atomic_write_json`写入，并包含以下字段：

- `round`：非负整数轮次。
- `changed_files`：本轮修改文件的仓库相对路径数组。
- `tests`：测试命令、通过数量和总数量等信息。
- `failures`：失败或阻塞项数组；没有时使用空数组。
- `metrics`：可复核的数值或计数指标对象。

示例结构：

```json
{
  "round": 1,
  "changed_files": ["status.json"],
  "tests": {"command": "python -m pytest -q", "passed": 1, "total": 1},
  "failures": [],
  "metrics": {"atomic_write": 1.0}
}
```

