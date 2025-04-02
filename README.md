# MedSAM2
Segment Anything in 3D Medical Images and Videos

[[`Paper`](tbd)] [[`Project`](https://medsam2.github.io/)] [[`Dataset List`](https://medsam-datasetlist.github.io/)] [[`3D Slicer`](https://github.com/bowang-lab/MedSAMSlicer/tree/MedSAM2)] [[`Gradio App`](app.py)] [[`CoLab`](https://colab.research.google.com/drive/1MKna9Sg9c78LNcrVyG58cQQmaePZq2k2?usp=sharing)] [[`BibTeX`](#citing-medsam2)]



## Installation 

- Create a virtual environment: `conda create -n medsam2 python=3.12 -y` and `conda activate medsam2` 
- Install [PyTorch](https://pytorch.org/get-started/locally/) 2.5: `conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.4 -c pytorch -c nvidia` (GPU is required)
- Download code `git clone https://github.com/bowang-lab/MedSAM2.git && cd MedSAM2` and run `pip install -e ".[dev]"`
- Download checkpoints: `sh download.sh`
- Optional: Please install the following dependencies for gradio

```bash
pip install gradio==3.38.0
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt-get update
sudo apt-get install ffmpeg
```

## Download annotated datasets

- [CT_DeepLesion-MedSAM2](https://huggingface.co/datasets/wanglab/CT_DeepLesion-MedSAM2)



- [LLD-MMRI-MedSAM2](https://huggingface.co/datasets/wanglab/LLD-MMRI-MedSAM2) 

Note: Please also cite the raw [DeepLesion](https://doi.org/10.1117/1.JMI.5.3.036501) and [LLD-MMRI](https://www.sciencedirect.com/science/article/pii/S0893608025001078) dataset paper when using these datasets. 

## Inference

### 3D medical image segmentation

- CMD

```bash
python medsam2_infer_3D_CT.py -i CT_DeepLesion/images -o CT_DeepLesion/segmentation
```

- [CoLab](https://colab.research.google.com/drive/1MKna9Sg9c78LNcrVyG58cQQmaePZq2k2?usp=sharing): MedSAM2_inference_CT_Lesion_Demo.ipynb

### Medical video segmentation

- CMD

```bash
python medsam2_infer_video.py -i input_video_path -m input_mask_path -o output_video_path 
```

- [CoLab](https://colab.research.google.com/drive/1hyVyGh7qjTbFMuv568YcRoccdkNuDTaW?usp=sharing): MedSAM2_Inference_Video_Demo.ipynb


### Gradio demo

```bash
python app.py
```

## Training

Specify dataset path in `sam2/configs/sam2.1_hiera_tiny_finetune512.yaml`

```bash
sbatch multi_node_train.sh
```

## Acknowledgements

- We highly appreciate all the challenge organizers and dataset owners for providing the public datasets to the community.
- We thank Meta AI for making the source code of [SAM2](https://github.com/facebookresearch/sam2) publicly available. Please also cite this paper when using MedSAM2. 


## Bibtex



