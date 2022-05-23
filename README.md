# action_recognition

[documents](documents/) -> pdf dokumentů


Dataset Kinetics 400 je možné stáhnout [zde](https://deepmind.com/research/open-source/kinetics).
[Data_loader_kin400.py](dataset/Data_loader_kin400.py) -> stažení datasetu pomocí knihovny **youtube-dl** a **ffmpeg** <br />
[Data_loader_kin400_2.py](dataset/Data_loader_kin400_2.py) -> stažení datasetu pomocí knihovny **pytube** a **moviepy** <br />
[kin400_analysis.ipynb](dataset/kin400_analysis.ipynb) -> analýza datasetu

[HAA500](HAA500/) -> složka obsahující tooly pro zpracování a předzpracování datasetu HAA500<br />
Dataset HAA500 je možné stáhnout [zde](https://www.cse.ust.hk/haa/).<br />
[video_parser.py](HAA500/video_parser.py) -> Pro zpracování RGB/segmentovaných videí <br />
[optical_flow.py](HAA500/optical_flow.py) -> Pro zpracování optického toku <br />

[model](model/) -> folder obsahující užitečné tooly pro zhodnocení modelů, spojení modelů a natrénování klasifikátorů <br />
[model_analysis.ipynb](model/model_analysis.ipynb) -> Analýza průběhu trénování modelů + vyhodnocení Top1, Top3, Top5 <br />
[softmax.py](model/softmax.py) -> Spojení A a Spojení B, [softmax_find_best.py](model/softmax_find_best.py) -> Pro nalezení idealnich parametru <br />
[test_classifier.py](model/test_classifier.py) -> použití klasifikátorů (natrénovaný model: [modelSVM.npy](model/classif_m/)) , [train_classifier.py](model/train_classifier.py) -> trénování klasifikátorů


