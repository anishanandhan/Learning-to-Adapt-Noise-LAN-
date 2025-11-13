Learning to Adapt Noise (LAN)
(Image Denoising via Test-Time Noise Adaptation)

1. Project Abstract

Real-world image denoising remains challenging due to highly variable and unpredictable noise distributions present across different imaging devices and conditions. Deep learning denoising models often fail when the test noise distribution differs from the training set. This project implements LAN (Learning to Adapt Noise), a test-time adaptation technique that enhances denoising performance without modifying model weights. LAN learns a small pixel-wise noise offset that, when added to the noisy input, shifts the noise distribution toward what the pretrained denoising network expects. This significantly improves performance on real-world unseen datasets.
We integrate LAN with Restormer, NAFNet, and Neighbor2Neighbor/ZS-N2N, and evaluate SIDD-pretrained models on PolyU and Nam datasets. The method demonstrates improved PSNR, SSIM, and perceptual quality. LAN is lightweight, computationally inexpensive, and effective for robust real-world denoising.

2. Team Members
Name	Register Number
Anish Anadhan A L	22MIS1190
Laksharaa A S	23MIA1053
Sandheep S S	23MIA1161

3. Base Paper Reference

Title: LAN: Learning to Adapt Noise for Image Denoising
Authors: Changjin Kim, Tae Hyun Kim, Sungyong Baik
Conference: CVPR 2023

BibTeX:
@inproceedings{kim2023lan,
  title={LAN: Learning to Adapt Noise for Image Denoising},
  author={Kim, Changjin and Kim, Tae Hyun and Baik, Sungyong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}

4. Tools and Libraries Used
Python 3.8+

PyTorch

NumPy

OpenCV

Pillow

basicsr

NAFNet

Restormer

Neighbor2Neighbor / ZS-N2N

tqdm

5. Steps to Execute the Code
Install Dependencies
pip install -r requirements.txt

Add Input Images

Put noisy images inside:

dataset/input/

Run LAN + Denoising

Using Restormer:

python src/main.py --image dataset/input/sample.png --model restormer


Using NAFNet:

python src/main.py --image dataset/input/sample.png --model nafnet

View Results

Outputs are saved to:

results/

6. Dataset Description
SIDD Dataset

Used for pretraining
🔗 https://www.eecs.yorku.ca/~kamel/sidd/

PolyU Real-World Dataset

Used for unseen real-noise evaluation
🔗 https://github.com/csj4032/Real-World-Noisy-Image-Denoising

Nam / RENOIR Dataset

Real noisy smartphone images
🔗 https://www.eecs.yorku.ca/~kamel/RENOIR/

Datasets too large → only links provided.

7. Output Screenshots / Result Summary

results/

Typical images:

noisy_input.png

pretrained_output.png

lan_output.png

comparison_grid.png

8. YouTube Demo Link

Add your demo link here:

[YouTube Demo Link]





