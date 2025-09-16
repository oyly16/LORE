<!-- <div align="center"> -->
  
# LORE: Latent Optimization for Precise Semantic Control in Rectified Flow-based Image Editing

[Paper](https://arxiv.org/abs/2508.03144) | [Online Demo](https://huggingface.co/spaces/oyly/LORE) |[CyberAgent AILab](https://research.cyberagent.ai/)
-----------------------------|-----------------------------|-----------------------------

![LORE examples](figs/fig_1.png)


# Implementation

Our code is tested with python==3.10.0, torch==2.6.0, transformers==4.49.0, flux-dev model.

Go to [FLUX](https://github.com/black-forest-labs/flux) codebase for more requirements.

To run with local checkpoints, edit model paths in [this file](src/flux/util_lore.py). 


# Image Editing

Examples are provided in /src/run_examples.sh
```
cd src
mkdir outputs_demo
bash run_examples.sh
```

Check [this script](src/demo_lore.py) for tuning edting parameters.

# Gradio Demo

To run our [HF space demo](https://huggingface.co/spaces/oyly/LORE) locally, run app.py
```
cd src
mkdir outputs_gradio
python app.py
```
Samples will be saved to /src/outputs_gradio.

We also support a demo that divide inverse and editing, which saves generation time.

Due to limited computational resources, you may resize the images to 800 pixels (long side) while preserving the aspect ratio.

```
cd src
mkdir outputs_gradio
(run locally) python gradio_lore.py --resize 800
(run sharing) python gradio_lore.py --resize 800 --share
```


# Visualize Attention Maps

We further add a demo to visualize attention maps of ours optimization noise comparing to inversion noise.

Examples are provided in /src/run_vis_attnmaps.sh
```
cd src
mkdir outputs_attnmap
bash run_vis_attnmaps.sh
```

# Citation

If you find our work helpful, please **star 🌟** this repo and **cite 📑** our paper. Thanks for your support!

```
@article{ouyang2025lore,
  title={LORE: Latent Optimization for Precise Semantic Control in Rectified Flow-based Image Editing},
  author={Ouyang, Liangyang and Mao, Jiafeng},
  journal={arXiv preprint arXiv:2508.03144},
  year={2025}
}
```

# Acknowledgements
This work was conducted during an internship at [CyberAgent AILab](https://research.cyberagent.ai/). We thank [FLUX](https://github.com/black-forest-labs/flux/tree/main) and [RF-Edit](https://github.com/wangjiangshan0725/RF-Solver-Edit) for their clean codebase.

# Contact
If you have any questions or concerns, please send emails to [oyly@iis.u-tokyo.ac.jp](oyly@iis.u-tokyo.ac.jp).
