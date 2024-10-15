# Prompt-guided Multi-modal contrastive learning for Cross-compression-rate Deepfake Detection

## Overview

- Framework
![image](https://hackmd.io/_uploads/S1P8QIzA0.png)

- Cross-Quality Similarity Learning(CQSL)
![image](https://hackmd.io/_uploads/r19_XIMCR.png)

- Cross-Modality Consistency Learning(CMCL)
![image](https://hackmd.io/_uploads/Hy2KXLGAR.png)



## Highlights

- The proposed Contrastive Physio-inspired Multi-modalities with Language guidance (CPML) framework represents an approach to addressing CCR scenarios in deepfake detection.

- We propose the Cross-Quality Similarity Learning (CQSL) strategy that adopts contrastive learning based on the intrinsic features of rPPG signals.

- We propose the Cross-Modality Consistency Learning (CMCL) strategy that aligns the unified features extracted from multi-modal with class prompts. To the best of our knowledge, this is the first multi-modal prompt-guided learning approach for deepfake detection.

# Setup

- Get Code
`git clone https://github.com/k-bigboss99/CPML.git`

- Dataset Pre-Processing
    - Benchmark Datasets: FF++, Celeb-DF, DFD
    - For benchmark dataset, follow the preprocessing step of SSDG to detect and align the faces using [MTCNN](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection).
    - For each video, we sample 160 frames (30fps) to ensure an adequate amount of rPPG signals for analysis.
    - Input the sample frames into MTCNN to detect, align and crop the images. Save the frames into Deepfake_dataset/FF++/ with the following directory structure:
    ```
    Deepfake_dataset/FF++/

    |-- original_sequences
        |-- youtube
        |   |--raw
        |   |  |--crop_MTCNN
        |   |  |  |--000, 001, 002...
        |   |  |  |  |--face0
        |   |  |  |  |  |--0000.jpg, 0001.jpg, 0002.jpg... 
        |   |--c23
        |   |  |--crop_MTCNN
        |   |  |  |--000, 001, 002...
        |   |  |  |  |--face0
        |   |  |  |  |  |--0000.jpg, 0001.jpg, 0002.jpg...   
        |   |--c40
        |   |  |--crop_MTCNN
        |   |  |  |--000, 001, 002...
        |   |  |  |  |--face0
        |   |  |  |  |  |--0000.jpg, 0001.jpg, 0002.jpg...

    |-- manipulated_sequences
        |-- Deepfakes
        |   |--raw
        |   |  |--crop_MTCNN
        |   |  |  |--000_003, 001_870, 002_006...
        |   |  |  |  |--face0
        |   |  |  |  |  |--0000.jpg, 0001.jpg, 0002.jpg... 
        |   |--c23
        |   |  |--crop_MTCNN
        |   |  |  |--000_003, 001_870, 002_006...
        |   |  |  |  |--face0
        |   |  |  |  |  |--0000.jpg, 0001.jpg, 0002.jpg...   
        |   |--c40
        |   |  |--crop_MTCNN
        |   |  |  |--000_003, 001_870, 002_006...
        |   |  |  |  |--face0
        |   |  |  |  |  |--0000.jpg, 0001.jpg, 0002.jpg...

    ```

- Training and Inference
    - Please refer to [train.md](/UgsYqvB_TSu4bi2JvmAQIg) for training the models.



## Results

- In- and Cross-dataset Evaluations
![image](https://hackmd.io/_uploads/B1inW8zRR.png)

- Cross-manipulation Evaluation
![image](https://hackmd.io/_uploads/Byk5W8GR0.png)

-  Cross-compression-rate Evaluation
![image](https://hackmd.io/_uploads/HJ29bUf0C.png)
![image](https://hackmd.io/_uploads/SkspbUMAR.png)

- Ablation study
![image](https://hackmd.io/_uploads/rJ_AZIzRC.png)



## Visualizations

## Citation
If you use the FaceForensics++ data or code please cite:
```
  @InProceedings{Srivatsan_2023_ICCV,
    author    = {Ching-Yi Lai, Chiou-Ting Hsu, Chia-Wen Lin and Chih-Chung Hsu},
    title     = {Prompt-guided Multi-modal contrastive learning for Cross-compression-rate Deepfake Detection},
    booktitle = {},
    month     = {November},
    year      = {2024},
    pages     = {}
}
```

## Acknowledgement
Our code is built on top of the [FLIP](https://github.com/koushiksrivats/FLIP?tab=readme-ov-file) repository. We thank the authors for releasing their code.

## Help
If you have any questions, please contact us at ching1999.work@gmail.com
