# SEN_CSLR
This repo holds codes of the paper: Self-Emphasizing Network for Continuous Sign Language Recognition.(AAAI 2023) [[paper]](https://arxiv.org/abs/2104.02330)

This repo is based on [VAC (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html). Many thanks for their great work!

## Prerequisites

- This project is implemented in Pytorch (>1.8). Thus please install Pytorch first.

- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode.

- sclite [[kaldi-asr/kaldi]](https://github.com/kaldi-asr/kaldi), install kaldi tool to get sclite for evaluation. After installation, create a soft link toward the sclite:    
  `ln -s PATH_TO_KALDI/tools/sctk-2.4.10/bin/sclite ./software/sclite`

- [SeanNaren/warp-ctc](https://github.com/SeanNaren/warp-ctc) for ctc supervision.

## Implementation
The implementation for the SSEM (line 47) and TSEM (line 23) is given in [./modules/resnet.py](https://github.com/hulianyuyy/SEN_CSLR/blob/main/modules/resnet.py).  

They are then equipped with the BasicBlock in ResNet in line 93 [./modules/resnet.py](https://github.com/hulianyuyy/SEN_CSLR/blob/main/modules/resnet.py).

## Data Preparation
You can choose any one of following datasets to verify the effectiveness of SEN.

### PHOENIX2014 dataset
1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/). Our experiments based on phoenix-2014.v3.tar.gz.

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET/phoenix2014-release ./dataset/phoenix2014`

3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python data_preprocess.py --process-image --multiprocessing
   ```

### PHOENIX2014-T dataset
1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET/PHOENIX-2014-T-release-v3/PHOENIX-2014-T ./dataset/phoenix2014-T`

3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python data_preprocess-T.py --process-image --multiprocessing
   ```

### CSL dataset

1. Request the CSL Dataset from this website [[download link]](https://ustc-slr.github.io/openresources/cslr-dataset-2015/index.html)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET ./dataset/CSL`

3. The original image sequence is 1280x720, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python data_preprocess-CSL.py --process-image --multiprocessing
   ``` 

### CSL-Daily dataset

1. Request the CSL-Daily Dataset from this website [[download link]](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET ./dataset/CSL-Daily`

3. The original image sequence is 1280x720, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python data_preprocess-CSL-Daily.py --process-image --multiprocessing
   ``` 

## Inference

### PHOENIX2014 dataset

| Backbone | Dev WER  | Test WER  | Pretrained model                                             |
| -------- | ---------- | ----------- | --- |
| Baseline | 21.2%      | 22.3%       |  --- | 
| ResNet18 | 19.7%      | 20.8%       | [[Baidu]](https://pan.baidu.com/s/1QRws8gylNzlpXvU52VCLww) (passwd: tsa2)<br />[[Google Drive]](https://drive.google.com/file/d/1uCIYCz0O7twKG1k_BE9sZ5Q1hga4DXRI/view?usp=sharing) |

### PHOENIX2014-T dataset

| Backbone | Dev WER  | Test WER  | Pretrained model                                             |
| -------- | ---------- | ----------- | --- |
| Baseline | 21.1%      | 22.8%       |  --- | 
| ResNet18 | 19.4%      | 21.2%       | [[Baidu]](https://pan.baidu.com/s/1o8IvZhFuTWM9pZI1U8Y2YQ) (passwd: c6cq)<br />[[Google Drive]](https://drive.google.com/file/d/1xFv0ttMQdU6SMvncEnHT0OT6osUCSXVK/view?usp=sharing) |

### CSL-Daily dataset

| Backbone | Dev WER  | Test WER  | Pretrained model                                            |
| -------- | ---------- | ----------- | --- |
| Baseline | 21.1%      | 22.8%       |  --- | 
| ResNet18 | 19.4%      | 21.2%       | [[Baidu]](https://pan.baidu.com/s/1o8IvZhFuTWM9pZI1U8Y2YQ) (passwd: c6cq)<br />[[Google Drive]](https://drive.google.com/file/d/1xFv0ttMQdU6SMvncEnHT0OT6osUCSXVK/view?usp=sharing) |

​	To evaluate the pretrained model, run the command below：   
`python main.py --device your_device --load-weights path_to_weight.pt --phase test`

### Training

The priorities of configuration files are: command line > config file > default values of argparse. To train the SLR model, run the command below:

`python main.py --device your_device`

Note that you can choose the target dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in line 3 in ./config/baseline.yaml.
 
### Citation

If you find this repo useful in your research works, please consider citing:

```latex
@inproceedings{hu2022temporal,
  title={Temporal Lift Pooling for Continuous Sign Language Recognition},
  author={Lianyu Hu, Liqing Gao, Zekang Liu and Wei Feng},
  booktitle={European conference on computer vision},
  year={2022},
  organization={Springer}
}
```