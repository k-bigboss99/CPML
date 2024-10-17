# Prompt-guided Multi-modal contrastive learning for Cross-compression-rate Deepfake Detection

## Overview

- Framework
    - ![image](https://github.com/user-attachments/assets/024aaabb-212f-4119-9043-f23d539b51a9)
  
- Cross-Quality Similarity Learning(CQSL)
    - ![image](https://github.com/user-attachments/assets/03b5633c-9430-4f07-84de-28e9af01dea8)

- Cross-Modality Consistency Learning(CMCL)
    - ![image](https://github.com/user-attachments/assets/6af96756-9f53-4a3f-9481-eb6716203297)


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

- Cross-compression-rate Evaluation
![image](https://hackmd.io/_uploads/HJ29bUf0C.png)
![image](https://hackmd.io/_uploads/SkspbUMAR.png)

- Ablation study
![image](https://hackmd.io/_uploads/rJ_AZIzRC.png)



## Visualizations

## Citation
If you use the FaceForensics++ data or code please cite:
```

```

## Acknowledgement
Our code is built on top of the [FLIP](https://github.com/koushiksrivats/FLIP?tab=readme-ov-file) repository. We thank the authors for releasing their code.

## Help
If you have any questions, please contact us at ching1999.work@gmail.com
