# Overview
This repository is written for the "CVPR 2025: Foundation Models for Interactive 3D Biomedical Image Segmentation"([link](https://www.codabench.org/competitions/5263/)) challenge. It
is based on VISTA3D ([link](https://github.com/Project-MONAI/VISTA/tree/main/vista3d/cvpr_workshop)). You can find the VISTA3D weights fine-tuned on the CVPR challenge dataset in their code repository ([link](https://github.com/Project-MONAI/VISTA/tree/main/vista3d/cvpr_workshop)).

<!-- We alternately use VISTA3D’s point sampling and our Gaussian edge-center point sampling strategy to form a two-stage fine-tuning strategy, and adjust the crop size, downsampling, and number of hints according to the input size. -->

# Environments and Requirements

- Ubuntu 20.04.6 LTS
- CPU: Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz
- RAM: 8×64GB; 3200 MT/s
- GPU: 4× NVIDIA A800-SXM4-80GB and A100-SXM4-80GB
- CUDA version: 12.2
- python version: 3.10

To install requirements:

```setup
pip install -r requirements.txt
```

# Dataset

For dataset details and download, please refer to [challenge website]((https://www.codabench.org/competitions/5263/)).

# Preprocessing
The dataset provided by the challenge has been preprocessed. For details, please refer to [challenge website]((https://www.codabench.org/competitions/5263/)).


# Training
Download the fine-tuned challenge subset checkpoints or the VISTA3D original checkpoints provided in the VISTA3D code repository ([link](https://github.com/Project-MONAI/VISTA/tree/main/vista3d/cvpr_workshop)).
```
nohup torchrun --nnodes=1 --nproc_per_node=4 vista_train_cvpr.py > coreset_2e5_gauss.log 2>&1 &
```
The checkpoint saved by train_cvpr.py can be updated by `vista_train_cvpr.py` to remove the additional `module` key due to multi-gpu training.


# Inference
You can directly download the [docker file](https://www.codabench.org/competitions/5263/) for the challenge baseline.
For more details, refer to the [challenge website]((https://www.codabench.org/competitions/5263/)).
```
docker container run --gpus "device=0" -m 32G --name teamname --rm -v $PWD/PathToTestSet/:/workspace/inputs/ -v $PWD/teamname_outputs/:/workspace/outputs/ teamname:latest /bin/bash -c "sh predict.sh"
```
You can also directly run `predict.sh`. Download the finetuned checkpoint and modify the `--model=/your_downloaded_checkpoint`. Change `save_data=True` in `infer_cvpr.py` to save predictions to nifti files for visualization.



# Evaluation


To calculate evaluation metrics, please submit your inference results to the [challenge website]((https://www.codabench.org/competitions/5263/)).


# Results

Our method achieves the following performance on [CVPR 2025: Foundation Models for Interactive 3D Biomedical Image Segmentation](https://www.codabench.org/competitions/5263/) coreset track.

| Modality   | Model name  |  DSC_AUC  |  NSD_AUC |  DSC_Final |  NSD_Final |
| ---------- | ----------- | :-------: | :------: | :--------: | :--------: |
| CT         | GS_MedSegFM | 2.9638    | 3.0037   | 0.7562     | 0.7739     |
| MRI        | GS_MedSegFM | 2.3980    | 2.7477   | 0.6046     | 0.6944     |
| Microscopy | GS_MedSegFM | 1.8252    | 2.8487   | 0.4665     | 0.6827     |
| PET        | GS_MedSegFM | 2.3949    | 2.1673   | 0.6089     | 0.5501     |
| Ultrasound | GS_MedSegFM | 2.5513    | 2.5333   | 0.7486     | 0.7497     |

# Acknowledgement

> We thank [CVPR 2025: Foundation Models for Interactive 3D Biomedical Image Segmentation](https://www.codabench.org/competitions/5263/), the contributors of the public dataset, and [VISTA3D](https://github.com/Project-MONAI/VISTA/tree/main/vista3d/cvpr_workshop).


