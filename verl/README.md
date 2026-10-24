# Bundled verl

This directory contains TOP-D's working **verl runtime and installation metadata**. Start with the [TOP-D README](../README.md) and install from the repository root with [`install.sh`](../install.sh). No additional verl checkout or submodule is needed.

## Scope

- Runtime code, model integrations, and training configuration are included under [`verl/`](verl/).
- TOP-D launchers and original datasets live in [`../recipe/`](../recipe/).
- Module-level Markdown files retain upstream technical and historical context. Their examples, dependencies, and benchmark settings are **not TOP-D defaults**; upstream example/recipe/CI directories are not bundled.
- Use TOP-D's [dependency pins](../recipe/environment/requirements.txt), not the inherited `[vllm]` extra, which has an older vLLM cap.

## Upstream resources

- [verl repository](https://github.com/verl-project/verl)
- [Documentation](https://verl.readthedocs.io/en/latest/)
- [Additional recipes](https://github.com/verl-project/verl-recipe)
- [HybridFlow paper](https://arxiv.org/abs/2409.19256)

The bundle preserves the working implementation rather than tracking upstream `main`. File provenance and deliberate release changes are recorded in [`source_manifest.json`](../recipe/verification/source_manifest.json).

## License and attribution

The original [Apache-2.0 license](LICENSE), [notices](Notice.txt), and per-file copyright statements remain in effect. Thanks to ByteDance Seed and the verl community for the training infrastructure.

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv:2409.19256}
}
```
