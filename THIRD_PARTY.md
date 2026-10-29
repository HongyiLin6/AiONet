# Third-party notices

| Component | Upstream revision | Terms and location |
|---|---|---|
| MetroVision/AiONet source | `ad3ee116741163fa1591de3fe746da90c4fe1433` from `https://gitee.com/wjnwjnwj/metro-vision.git` | MIT, copyright 2025 Jianing Wang; license retained in `training/vendor/metrovision/` and `licenses/` |
| MMDetection3D and included plugins | Version 1.0.0rc6 source in the pinned repository | Apache-2.0 and original component headers; see vendored licenses |
| EMIFF adapted camera modules | `4cea36ea9ecb4dfc73ece47b8f446fbdb179e290` | Apache-2.0; license in `training/scripts/r2_emiff/LICENSE` and `licenses/EMIFF-Apache-2.0.txt` |
| V2X-ViT optional source | `f0e6c13f41e916548b2d8aba61e42a18ce980416` | MIT; fetched separately by `tools/fetch_baselines.py` |
| Select2Col optional source | `5978cc466f294af3252381da4da7d2bee56c9694` | Upstream NoDistrib terms; direct upstream acquisition through the optional fetcher |
| V2X-Seq SPD dataset interface | Official AIR-THU release | Obtain data separately under the dataset terms; `https://github.com/AIR-THU/DAIR-V2X-Seq` |

This source-only release supplies the SPD dataset adapter, selection workflow and independently reconstructed perception experiments. EMIFF adaptations adjust imports, class vocabulary, anchors and data loading for the common camera protocol. The source manifest identifies input source files; release checksums identify distributed files after portability changes. Model weights, datasets, cached detections and experiment outputs are supplied by the person running the code.

Dependency packages installed by pip retain their own licenses. This top-level MIT license applies to original release code and does not replace upstream license terms.
