# Ther2RGB-Translation

해당 코드는 2021-2학기 세종대학교 컴퓨터비전 수업 텀프로젝트 베이스라인 코드입니다.

## Getting Started
1. 해당 레포를 깃 클론합니다.
```sh
git clone https://github.com/sjmin99/Ther2RGB-Translation.git
cd Ther2RGB-Translation
```
2. 학습을 위한 필요한 라이브러리를 설치합니다.
```
pip install -r requirements.txt
```
3. 학습을 위한 데이터 셋을 다운받습니다.


### Training
```
python train.py --name $YOUR_MODEL_NAME --dataroot $YOUR_DATA_PATH
```

### Multi-GPU training
```
python train.py --name $YOUR_MODEL_NAME --dataroot $YOUR_DATA_PATH --gpu_ids 0,1,2,3
```

### Baseline Setting
```
sh scripts/train.sh
```

### Evaluating
```
sh scripts/test.sh
```
@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```
