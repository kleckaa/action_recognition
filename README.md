# action_recognition


## Kinetics-400
Předtrénované váhy pro I3D síť je možné stáhnout [zde]
Dataset Kinetics-400 je možné stáhnout [zde](https://github.com/IBM/action-recognition-pytorch/releases/tag/weights-v0.1). <br />
[Data_loader_kin400.py](Kinetics-400/Data_loader_kin400.py) -> stažení datasetu pomocí knihovny **youtube-dl** a **ffmpeg** <br />
[Data_loader_kin400_2.py](Kinetics-400/Data_loader_kin400_2.py) -> stažení datasetu pomocí knihovny **pytube** a **moviepy** <br />
[kin400_analysis.ipynb](Kinetics-400/kin400_analysis.ipynb) -> analýza datasetu

## HAA
* [HAA500](HAA500/) -> složka obsahující tooly pro zpracování a předzpracování datasetu HAA500<br />
* Dataset HAA500 je možné stáhnout [zde](https://www.cse.ust.hk/haa/).<br />
* [video_parser.py](HAA500/video_parser.py) -> Pro zpracování RGB/segmentovaných videí <br />
<img src="img/segm_1.jpg " width="500" > <br />
* [optical_flow.py](HAA500/optical_flow.py) -> Pro zpracování optického toku <br />
<img src="img/of_1.jpg " width="500" > <br />
* [mean_std.py](HAA500/mean_std.py) -> Výpočet mean a std z datasetů <br />

## I3D
* Originální repozitář: [https://github.com/IBM/action-recognition-pytorch](https://github.com/IBM/action-recognition-pytorch)<br />
* [train_main.py](I3D/train_main.py) -> Pro trénování I3D <br />
* [test_main.py](I3D/test_main.py) -> Pro testování I3D <br />
* Nejlépe natrénované modely z DP na RGB, segmentaci, optickém toku je možné stáhnou [zde](https://drive.google.com/drive/folders/1SlKmSZPQsmyVRqeRIeWmiIoNv5GWIk4M?usp=sharing)<br />

## Prace s natrénovanými modely
* [model](model/) -> folder obsahující užitečné tooly pro zhodnocení modelů, spojení modelů a natrénování klasifikátorů <br />
* [model_analysis.ipynb](model/model_analysis.ipynb) -> Analýza průběhu trénování modelů + vyhodnocení Top1, Top3, Top5 <br />
* [analysis.py](model/analysis.py) -> Hlubší analýza modelů na základě skupin činností<br />
* [softmax.py](model/softmax.py) -> Spojení A a Spojení B, [softmax_find_best.py](model/softmax_find_best.py) -> Pro nalezení idealnich parametru <br />
* [test_classifier.py](model/test_classifier.py) -> použití klasifikátorů (natrénovaný model: [modelSVM.npy](model/classif_m/)) , [train_classifier.py](model/train_classifier.py) -> trénování klasifikátorů


