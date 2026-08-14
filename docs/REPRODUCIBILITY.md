# 可复现性

## 最小复现

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
python examples/synthetic_reservoir/run_demo.py
python scripts/audit_release.py
```

最小复现验证以下内容：

- H–V–A节点插值与越界拒绝；
- 共享送出下水电–FPV协同调度的水量保持；
- 生命周期年度账本和累计净碳恒等式；
- 条件水库GHG因子分解；
- 仓库无真实场址标识、绝对路径、密钥指示和超限文件。

研究案例的完整运行需要受限数据和多个隔离环境，因此不属于本公开仓库的最小复现承诺。
