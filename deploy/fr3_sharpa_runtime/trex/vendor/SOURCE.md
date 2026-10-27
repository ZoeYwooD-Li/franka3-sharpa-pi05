# Bundled T-Rex model source

`trex/` contains the model, training, tactile VQ-VAE, and inference Python core from
[`ZhuoyangLiu2005/T-Rex`](https://github.com/ZhuoyangLiu2005/T-Rex), commit
`ba4047ce3e9f3d2013d60de600e136897e90ccb8`. Its root `LICENSE` is preserved.
The original large dataset and hardware examples are omitted because the Sharpa
checkpoint inference path does not import them.

The vendored copy has the Sharpa adapter, LoRA, and inference patches from
[`ZoeYwooD-Li/trex-sharpa-tomato`](https://github.com/ZoeYwooD-Li/trex-sharpa-tomato),
commit `df2a12e70e2596cf0e28bd97bf31fc3b4c47554d`. The exact patch sources
are in `../model_patch/`. `source_manifest.json` records hashes of the inference
modules; `serve.py` checks them before loading a model.

No model weights, optimizer states, robot data, or third-party SDK packages are
bundled here. The user's merged LoRA or full checkpoint remains outside Git.
