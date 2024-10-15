A super-resolution method based on gastroscopic images.
The dataset can be download in https://davidrecasens.github.io/EndoDepthAndMotion/   

Installation：

Dependencies:
  PyTorch>1.10
  OpenCV
  Matplotlib 3.3.4
  opencv-python
  pyyaml
  tqdm
  numpy
  torchvision

Evaluate Pretrained Models：
Example: evaluate the model trained with DF2K@X4:
  Step 1, the following cmd will report a performance evaluated with python script, and generated images are placed in ./SR
python test.py -v "OmniSR_X4_DF2K" -s 994 -t tester_Matlab --test_dataset_name "test"
  Step2, please execute the script in the root directory to obtain the results reported in the paper. Please modify and to match the model/dataset name evaluated above.Evaluate_PSNR_SSIM.mLine 8 (Evaluate_PSNR_SSIM.m): methods = {'OmniSR_X4_DF2K'};Line 10 (Evaluate_PSNR_SSIM.m): dataset = {'test'};

Training：
Step1, please download training dataset from DIV2K ( and ), then set the dataset root path in Train Data Track 1 bicubic downscaling x? (LR images)Train Data (HR images)./env/env.json: Line 8: "DIV2K":"TO YOUR DIV2K ROOT PATH"
Step2, please download benchmark (baidu cloud (passwd: sjtu) , Google driver), and copy them to . If you want to generate the benchmark by yourself, please refer to the official repository of RCAN../benchmark/
Step3, training with DIV2K dataset:
python train.py -v "OmniSR_X4_DIV2K" -p train --train_yaml "train_OmniSR_X4_DIV2K.yaml"

Modified from:Omni Aggregation Networks for Lightweight Image Super-Resolution (OmniSR)，and can be downloaded from https://arxiv.org/pdf/2304.10244



