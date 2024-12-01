# Prompt-guided Multi-modal contrastive learning for Cross-compression-rate Deepfake Detection

## Overview

- Framework
    - ![image](https://github.com/user-attachments/assets/96a96bfd-dc3f-4bd8-b7f9-410cbd685ae2)

- Cross-Quality Similarity Learning(CQSL)
    - ![image](https://github.com/user-attachments/assets/a67dde90-e458-45f3-acc0-294b5bee1244)


- Cross-Modality Consistency Learning(CMCL)
    - ![image](https://github.com/user-attachments/assets/1bb426ee-612c-4d86-abe6-8b1bee8cdbb7)


## Highlights

- The proposed Contrastive Physio-inspired Multi-modalities with Language guidance (CPML) framework represents an approach to addressing CCR scenarios in deepfake detection.

- We propose the Cross-Quality Similarity Learning (CQSL) strategy that adopts contrastive learning based on the intrinsic features of rPPG signals.

- We propose the Cross-Modality Consistency Learning (CMCL) strategy that aligns the unified features extracted from multi-modal with class prompts. To the best of our knowledge, this is the first multi-modal prompt-guided learning approach for deepfake detection.

# Setup

- Get Code
`git clone https://github.com/k-bigboss99/CPML.git`

- Dataset Pre-Processing
    - Benchmark Datasets: FF++, Celeb-DF, DFD
    - For the benchmark dataset, follow the preprocessing step of SSDG to detect and align the faces using [MTCNN](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection).
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



## Results

- In- and Cross-dataset Evaluations
    - ![image](https://github.com/user-attachments/assets/a871d03e-dced-44bd-b11f-3a96399abf68)

- Cross-manipulation Evaluation
    - ![image](https://github.com/user-attachments/assets/9ae29359-5533-4918-a33c-913348717952)

-  Cross-compression-rate Evaluation
    - ![image](https://github.com/user-attachments/assets/65030c89-9946-4203-9c0e-fe7e76328c1a)

- Ablation study
    - ![image](https://github.com/user-attachments/assets/9bcf45ce-6b81-4558-93ea-93e3d8380159)

## Citation
If you use the CPML data or code please cite:
```

```

## Acknowledgement
Our code is built on top of the [FLIP](https://github.com/koushiksrivats/FLIP?tab=readme-ov-file) repository. We thank the authors for releasing their code.

## Help
If you have any questions, please contact us at ching1999.work@gmail.com
