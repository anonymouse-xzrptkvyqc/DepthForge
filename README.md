# DepthForge
## Installation & Environment Setup

Clone the repository:

```
git clone --recursive https://github.com/anonymouse-xzrptkvyqc/DepthForge.git
```

Follow these steps to set up your environment:

```
conda create -n depthforge python=3.11 -y
conda activate depthforge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 #2.6.0
pip install -U openmim
mim install mmengine

#install mmcv
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install -r requirements/optional.txt
pip install -e . -v

pip install mmsegmentation
pip install mmdet
pip install xformers=='0.0.30' # optional for DINOv2
pip install -r requirements.txt
pip install future tensorboard
```

## Dataset Preparation

Prepare the datasets by converting them to the required formats. Run the following commands:

```
cd DepthForge
mkdir data
# Convert GTA dataset (source domain)
python tools/convert_datasets/gta.py data/gta
# Prepare Cityscapes dataset
python tools/convert_datasets/cityscapes.py data/cityscapes
# Convert Mapillary to Cityscapes format (training data)
python tools/convert_datasets/mapillary2cityscape.py data/mapillary data/mapillary/cityscapes_trainIdLabel --train_id
# Resize Mapillary validation images to Cityscapes format
python tools/convert_datasets/mapillary_resize.py data/mapillary/validation/images data/mapillary/cityscapes_trainIdLabel/val/label data/mapillary/half/val_img data/mapillary/half/val_label
```

The final folder structure should look like this:

```
DepthForge
├── ...
├── checkpoints
│   ├── dinov2_vitl14_pretrain.pth
│   ├── dinov2_rein_and_head.pth
│   ├── depth_anything_v2_vitl.pth
│   ├── dinov2_converted.pth
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── bdd100k
│   │   ├── images
│   │   │   ├── 10k
│   │   │   │   ├── train
│   │   │   │   ├── val
│   │   ├── labels
│   │   │   ├── sem_seg
│   │   │   │   ├── masks
│   │   │   │   │   ├── train
│   │   │   │   │   ├── val
│   ├── mapillary
│   │   ├── training
│   │   ├── cityscapes_trainIdLabel
│   │   ├── half
│   │   │   ├── val_img
│   │   │   ├── val_label
│   ├── gta
│   │   ├── images
│   │   ├── labels
├── ├── adac
│   │   ├── gt
│   │   │   ├── fog
│   │   │   ├── night
│   │   │   ├── rain
│   │   │   ├── snow
│   │   ├── rgb_anon
│   │   │   ├── fog
│   │   │   ├── night
│   │   │   ├── rain
│   │   │   ├── snow
├── ...

```

## Pre-trained Weights & Dataset Downloads

* **Download:** 
  Download the pre-trained weights for testing from [facebookresearch](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth). Ensure the file name remains unchanged and place it in the project directory. You can also download the DepthAnything weights from [DepthAnything GitHub](https://github.com/DepthAnything/Depth-Anything-V2).

* **Convert:** 
  Convert the pre-trained weights for training or evaluation by running:
  
  ```bash
  python tools/convert_models/convert_dinov2.py checkpoints/dinov2_vitl14_pretrain.pth checkpoints/depth_anything_v2_vitl.pth checkpoints/dinov2_depth_converted.pth

* For 1024x1024 resolution (optional), run:

  ```bash
  python tools/convert_models/convert_dinov2.py checkpoints/dinov2_vitl14_pretrain.pth checkpoints/depth_anything_v2_vitl.pth checkpoints/dinov2_depth_converted_1024x1024.pth --height 1024 --width 1024
  ```

## Training

Use the following commands to start training with different configurations. If you need to resume training from a checkpoint, simply append `--resume` to the command.

*Tips: If resuming training appears to hang or shows no response for a long time, please refer to [this issue](https://github.com/open-mmlab/mmsegmentation/issues/3671) for potential solutions.*

- **Cityscapes → BDD100K + Mapillary + ADAC (fog, night, rain, snow):**

  ```
  python tools/train.py configs/dinov2/depthforge_dinov2_mask2former_512x512_bs1x4_citys.py
  # To resume training, use:
  # python tools/train.py configs/dinov2/depthforge_dinov2_mask2former_512x512_bs1x4_citys.py --resume
  ```

- **GTAV → BDD100K + Mapillary + Cityscapes:**

  ```
  python tools/train.py configs/dinov2/depthforge_dinov2_mask2former_512x512_bs1x4.py
  # To resume training, use:
  # python tools/train.py configs/dinov2/depthforge_dinov2_mask2former_512x512_bs1x4.py --resume
  ```

For the updated DepthForge V2 architecture, use these commands:

- **Cityscapes Configuration (DepthForge V2):**

  ```
  python tools/train.py configs/dinov2/depthforgev2_dinov2_mask2former_512x512_bs1x4_citys.py
  # To resume training, use:
  # python tools/train.py configs/dinov2/depthforgev2_dinov2_mask2former_512x512_bs1x4_citys.py --resume
  ```

- **GTAV Configuration (DepthForge V2):**

  ```
  python tools/train.py configs/dinov2/depthforgev2_dinov2_mask2former_512x512_bs1x4.py
  # To resume training, use:
  # python tools/train.py configs/dinov2/depthforgev2_dinov2_mask2former_512x512_bs1x4.py --resume
  ```



## Evaluation

To evaluate a trained model, replace `<DepthForge model>.pth` with your model file and run the corresponding command. The backbone checkpoint `checkpoints/dinov2_converted.pth` is used in all evaluations:

- **Evaluation with GTAV-based Configuration:**

  ```
  python tools/test.py configs/dinov2/depthforge_dinov2_mask2former_512x512_bs1x4.py <DepthForge model>.pth --backbone checkpoints/dinov2_converted.pth
  ```

- **Evaluation with Cityscapes-based Configuration:**

  ```
  python tools/test.py configs/dinov2/depthforge_dinov2_mask2former_512x512_bs1x4_citys.py <DepthForge model>.pth --backbone checkpoints/dinov2_converted.pth
  ```

- **Evaluation with DepthForge V2 Cityscapes Configuration:**

  ```
  python tools/test.py configs/dinov2/depthforgev2_dinov2_mask2former_512x512_bs1x4_citys.py <DepthForge model>.pth --backbone checkpoints/dinov2_converted.pth
  ```

- **Evaluation with DepthForge V2 GTAV Configuration:**

  ```
  python tools/test.py configs/dinov2/depthforgev2_dinov2_mask2former_512x512_bs1x4.py <DepthF
  ```

## Acknowledgment

Our implementation is mainly based on following repositories. Thanks for their authors.

- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [Rein](https://github.com/w1oves/Rein)
