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
```
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1479OIpNdDe2e3DaXVg0UV3aNPX_ST-wD' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1479OIpNdDe2e3DaXVg0UV3aNPX_ST-wD" -O dataset.zip && rm -rf ~/cookies.txt
unzip dataset.zip
```
데이터 셋 구성은 다음과 같습니다.
```
├── T2R_Dataset
   ├── train_A                                 # Source domain(Thermal) dataset for training
   ├── train_B                                 # Target domain(RGB) dataset for training
   └── test_A                                  # Source domain(Thermal) dataset for training
```
### Training
다음 명령어를 실행시키면 모델을 학습할 수 있습니다.
```
python train.py --name $YOUR_MODEL_NAME --dataroot $YOUR_DATA_PATH
```
위에 명령어를 처음 실행시킬 경우 wandb 로그인을 해야하는 과정을 거치게 됩니다. wandb login 및 사용방법은 아래 링크를 참고해주세요.(해당 코드 내부에는 wandb 로그인만 하셔도 기본적인 기능들을 사용하실 수 있으니 로그인만 하셔도 무방합니다.)
https://docs.wandb.ai/quickstart

### Multi-GPU training
Multi-GPU가 사용이 가능하다면 아래 명령어를 통해 Multi-GPU로 학습시킬 수 있습니다.
```
python train.py --name $YOUR_MODEL_NAME --dataroot $YOUR_DATA_PATH --gpu_ids 0,1,2,3
```
참고 : GAN은 학습의 불안정성 때문에 Batch를 작게 하는 것이 일반적입니다. 너무 큰 배치 사이즈로 학습 시 어떤 성능이 나오는지 평가해보지 않았으므로, 사용자분들의 재량에 맞춰 세팅해주세요.
### Baseline Setting
리더보드 베이스라인에 사용한 arguments입니다.
```
sh scripts/train.sh
```
보다 자세한 argument 종류 및 설명은 options 폴더 내 script에서 확인하실 수 있습니다.
### Evaluating
다음 명령어를 실행시키면 학습한 모델을 평가할 수 있습니다.
```
sh scripts/test.sh
```
## Citation
```
@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
@inproceedings{wang2018pix2pixHD,
  title={High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs},
  author={Ting-Chun Wang and Ming-Yu Liu and Jun-Yan Zhu and Andrew Tao and Jan Kautz and Bryan Catanzaro},  
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
```
