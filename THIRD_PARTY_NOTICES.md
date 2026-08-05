# Third-Party Notices

This repository contains modified third-party code. The root MIT license applies only to original SRDiff contributions; the components below remain subject to their upstream licenses.

## Meta DiT

- Component: `src/models/dit.py`
- Upstream: [facebookresearch/DiT](https://github.com/facebookresearch/DiT)
- Copyright: Meta Platforms, Inc. and affiliates
- License: [Creative Commons Attribution-NonCommercial 4.0 International](licenses/CC-BY-NC-4.0.txt)
- Changes: adapted for spatiotemporal latent satellite-to-radar translation.

The DiT-derived component is non-commercial. The repository as a whole must not be treated as MIT-only.

## Hugging Face Diffusers

- Components: `src/models/prediff/`
- Upstream: [huggingface/diffusers](https://github.com/huggingface/diffusers)
- Copyright: The Hugging Face Team and contributors
- License: [Apache License 2.0](licenses/Apache-2.0.txt)
- Changes: reduced and adapted VAE building blocks for the released frame-wise VAE.

## Amazon Science Earthformer

- Components: `datasets/sevir/`
- Upstream: [amazon-science/earth-forecasting-transformer](https://github.com/amazon-science/earth-forecasting-transformer)
- Copyright: Amazon.com, Inc. or its affiliates
- License: [Apache License 2.0](licenses/Apache-2.0.txt)
- Notice: [Earthformer NOTICE](licenses/Earthformer-NOTICE.txt)
- Changes: reduced to the four-modality map-style SEVIR translation loader and evaluation path.

## CompVis Stable Diffusion

- Component: `src/utils/distributions.py`
- Upstream: [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
- Copyright: Robin Rombach, Patrick Esser, and contributors
- License: [CreativeML Open RAIL-M](licenses/CreativeML-Open-RAIL-M.txt)
- Changes: reduced to the diagonal-Gaussian posterior required by the VAE.

See each upstream project for additional attribution and usage conditions.
