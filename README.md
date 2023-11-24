## openSW 수업 관련 레포지토리 

1. 구조소개

```
메인 <- 모든게 다 들어간 폴더 
브랜치 <- 각 주제별로 들어간 브랜치 
코드 리포지토리는 별도로 작성 (간단한건 같이 들어감) 

문서는 거의 마크다운 형식 아니면 스프레드시트 파일 
```


2. 코드 레포지토리 목록
* Telegram_bot
텔레그램 알림봇 + 채팅 요청에 맞추어 openai 응답하는 봇 (api key, env 미포함) 과제 2, 3
* mask_rcnn_mmdection
mmdection을 통해 수행한 과제 1 - 로컬 머신으로 실행하여 추론 및 학습 (샘플데이터인 coco dataset에 대한 1회 추론 전부 진행) 
* mask_rcnn
Mask_rcnn 깃허브 레포지토리의 포크 버전 
* mask_rcnn_update  
mask_rcnn레포지토리 통해 수행한 과제 1 - 이 버전으로 진행 


3. 레포지토리 베이스라인 및 형상 기준

과제의 요소 혹은 특정 기간에 맞춰 기준을 잡음 

커밋은 꾸준히 많이 들어감. (틈틈히 백업 혹은 롤백용으로 진행할 수 있도록 함)
vscode로 깃허브 커밋 풀 푸쉬 등등 진행함. (브랜치 추가도 동일) 

4. 관련 글 덩어리

작동환경 
proxmox_virtual_machine : qemu-kvm
hardware-virtualiztion on (for AVX/AVX2)
24GiB Memory
ubuntu 22.04 with EFI system
16processor
80GB disk 

#### mmdection 과정에서 발생한 문제 

note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for mmcv-full
  Running setup.py clean for mmcv-full

installation 과정에서 mmcv 문제로 직접 빌드가 필요할 때 아래 문서를 참고하여 mmcv custom 빌드 후 다시 실행 
https://mmcv.readthedocs.io/en/latest/get_started/build.html 


### MASK- rcnn

https://m.blog.naver.com/adamdoha/221906246483 

pip requirements 

pip install tensorboard==1.15.0 tensorflow==1.15.0 tensorflow-estimator==1.15.1 tensorflow-gpu==1.15.2 tensorflow-gpu-estimator==2.1.0 Keras==2.2.5 Keras-Applications==1.0.8 Keras-Preprocessing==1.1.0

train
python3 'balloon.py의 절대경로' train --dataset='Mask_RCNN/datasets/balloon_dataset의 절대경로' --weights=coco

python3 './Mask_RCNN/samples/balloon/balloon.py' splash --weights='./Mask_RCNN/mask_rcnn_balloon.h5' --image='./Mask_RCNN/balloon/datasets/val/5555705118_3390d70abe_b.jpg'

가끔 먼가 pip 하라는 것도 같이 pip 
