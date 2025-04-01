# MedSAM2
Segment Anything in 3D Medical Images and Videos

[[`Paper`](tbd)] [[`Project`](tbd)] [[`Datasets`]()] [[`3D Slicer`](https://github.com/bowang-lab/MedSAMSlicer/tree/MedSAM2)] [[`Gradio App`]()] [[`CoLab`]()] [[`BibTeX`](#citing-sam-2)]

## Installation 

- Create a virtual environment: `conda create -n medsam2 python=3.12 -y` and `conda activate medsam2` 
- Install [PyTorch](https://pytorch.org/get-started/locally/) 2.5: `conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.4 -c pytorch -c nvidia` (GPU is required)
- Download the code `git clone https://github.com/bowang-lab/MedSAM2.git && cd MedSAM2` and run `pip install -e ".[dev]"`
- Download the [checkpoints](https://drive.google.com/drive/folders/1R48RTvrjMcEY3b00T6QYum4dL3o_GIQz?usp=sharing) to `checkpoints`
- Optional: Please install the following dependencies for gradio

```bash
pip install gradio==3.38.0
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt-get update
sudo apt-get install ffmpeg
```


## Inference



### 3D medical image segmentation

```bash
python medsam2_infer_3D_CT_deeplesion.py -i CT_DeepLesion/images -o CT_DeepLesion/segmentation
```


### Medical video segmentation

```bash
python medsam2_infer_video.py -i input_video_path -o output_video_path 
```

### Gradio demo

```bash
python app.py
```

## Training

Specify dataset path in `sam2/configs/sam2.1_hiera_tiny_finetune512.yaml`

```bash
sbatch multi_node_train.sh
```


