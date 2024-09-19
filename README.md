# Inspection of imprint defects in stamped metal surfaces using deep learning and tracking

## Overview

Detection and classification for imprints in stamped metal sheets
This research focuses on the automatic detection and classification of imprint defects on the surface of metal parts. This innovative research had collaboration with a multinational industry, which provided a system to capture images of vehicle parts as well as information about defects, their frequency, and quality requirements. As our main contribution, we proposed a framework that combines detection, classification, and tracking in a synergistic way to assist the automotive industry. We used a state-of-the-art convolutional neural network, known as RetinaNet, to detect and classify imprint defects. We explored the temporal coherence in consecutive frames by tracking detected regions so as to reduce false alarms --- unstable candidate regions that are rarely (re)detected many times --- or to fix the classification of regions that are alternated classified as mild or severe imprint across frames. In our experiments, we achieved a mean average precision of 76% to detect and classify mild and severe defects, outperforming state-of-the-art detectors for static images. For severe imprints only, we achieved precision and recall values of 90% and 92%, respectively. These are promising results that could also benefit other industrial applications such as inspection of fissures, holes, wrinkles and scratches, that also use image sequences. 

![](/imprint.jpg)

## Source code

Link: [Google Drive](https://drive.google.com/file/d/12CHtJo52kvEnSPuHxwZp5fHU9w9UrH1K/view?usp=sharing)


## Dataset download

Link: [Google Drive](https://drive.google.com/drive/folders/1-kcj73qL-nTVR63k2gz9ZpJ8L-b5wRg2?usp=sharing).

## Citing

If you use our dataset or code in your research, please cite our paper to acknowledge this work. 

Link: [IEEE Access](https://ieeexplore.ieee.org/abstract/document/9062515).

```
@ARTICLE{9062515,
  author={Block, Sylvio Biasuz and da Silva, Ricardo Dutra and Dorini, Leyza Baldo and Minetto, Rodrigo},
  journal={IEEE Transactions on Industrial Electronics}, 
  title={Inspection of Imprint Defects in Stamped Metal Surfaces Using Deep Learning and Tracking}, 
  year={2021},
  volume={68},
  number={5},
  pages={4498-4507},
  doi={10.1109/TIE.2020.2984453}
}
```
