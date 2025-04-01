# MedSAM2
Segment Anything in 3D Medical Images and Videos

[[`Paper`](tbd)] [[`Project`](tbd)] [[`Datasets`]()] [[`3D Slicer`](https://github.com/bowang-lab/MedSAMSlicer/tree/MedSAM2)] [[`Gradio App`]()] [[`CoLab`]()] [[`BibTeX`](#citing-sam-2)]

## Installation 

- Create a virtual environment: `conda create -n medsam2 python=3.12 -y` and `conda activate medsam2` 
- Install [PyTorch](https://pytorch.org/get-started/locally/) 2.5: `conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.4 -c pytorch -c nvidia` (GPU is required)
- Download the code `git clone https://github.com/bowang-lab/MedSAM2.git && cd sam2` and run `pip install -e ".[dev]"`
- Optional: Please install the following dependencies for gradio

```bash
pip install gradio==3.38.0
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt-get update
sudo apt-get install ffmpeg
```

## Inference

### 3D medical image segmentation

```
python medsam2_infer_3D_CT_deeplesion.py -i CT_DeepLesion/images -o './CT_DeepLesion/segmentation'
```


### Medical video segmentation



