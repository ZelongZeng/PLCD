# Geo-localization via ground-to-satellite cross-view image retrieval  
The official code of IEEE Transactions on Multimedia paper "Geo-localization via ground-to-satellite cross-view image retrieval"  
  
## What's New?  
**24 Jun 2024**  
- Fix a bug in ```test_PL_G2D.py``` for ```--low_altitude```. 

**22 Mar 2023**  
- We shared the pre-trained models.   Please check [here](#pretrained-models). 
- We fixed some bugs. 

**12 Feb 2023**  
- We updated the structure of PLCD to achieve better performace.  
- We re-adjusted the values of the hyper-parameters to achieve better performance.  
- We removed the detection part for higher training efficiency.  
- We only used global features during test for higher retrieval efficiency.  
- We have fixed a big error in evaluate_dfs.py, which causes the performance (ground-to-satellite retrieval) in our paper are incorrect.  
  
## Preparation  
  
### Prerequisites  
  
- Python 3.6  
- Numpy > 1.19.2  
- Pytorch 1.10+  
  
### Dataset  
  
Please download the dataset from [here](https://github.com/layumi/University1652-Baseline)  
Then, put all files from the folder "Index" of this project into the folder "train" of the dataset.  
  
More detailed file structure:  
```  
├── University-1652/  
│   ├── readme.txt  
│   ├── train/  
│       ├── drone_id.txt                /* drone-view training images' id  
│       ├── drone_path.txt              /* drone-view training images' path  
│       ├── drone_view_id.txt           /* drone-view training images' shooting direction id  
│       ├── street_path.txt             /* ground-view training images' path  
│       ├── satellite_id.txt            /* satellite-view training images' id  
│       ├── satellite_path.txt          /* satellite-view training images' path  
│       ├── drone/  
│           ├── 0001  
|           ├── 0002  
|           ...  
│       ├── street/  
│       ├── satellite/  
│       ├── google/  
│   ├── test/  
│       ├── query_drone/  
│       ├── gallery_drone/  
│       ├── query_street/  
│       ├── gallery_street/  
│       ├── query_satellite/  
│       ├── gallery_satellite/  
│       ├── 4K_drone/  
```  
  
### Installation  
Install this project:  
```  
git clone https://github.com/ZelongZeng/PLCD.git  
cd PLCD  
mkdir model  
```  
  
Divide the drone-view gallery set into 3 parts based on the caputed altitude:  
```  
prepare_limited_view.py --root_path 'Input Your Root Path Of The Dataset'.  
```  
  
We found that use the low-altitude drone-view images can reach the best performance!  
  
## Train & Evaluation  
### Train & Evaluation for the Ground-Drone representation  
**Different from our original paper, in this updated structure, we only use the global feature for knowledge distillation in the Step 2, and use the local feature in the Step 3.**  
  
**Train Step 1:**  
```  
python train_PL_G2D.py --train_all --batchsize 8 --pool max --data_dir $Your_Data_Root$ --gen 0 --seed 0 --name $Name_of_Step-1$ --gpu 0 --lambda1 1.0 --lambda2 0.1 --easy_pos  
```  
  
**Train Step 2:**  
```  
python train_PL_G2D.py --train_all --batchsize 8 --pool max --data_dir $Your_Data_Root$ --gen 1 --seed 0 --old_name $Name_of_Step-1$ --name $Name_of_Step-2$ --gpu 0 --lambda1 1.0 --lambda2 0.1 --lambda3 1.0 --tau 0.1 --easy_pos  
```  
  
**Train Step 3:**  
```  
python train_PL_G2D.py --train_all --batchsize 8 --pool max --data_dir $Your_Data_Root$ --gen 2 --seed 0 --old_name $Name_of_Step-2$ --name $Name_of_Step-3$ --gpu 0 --lambda1 1.0 --lambda2 0.1 --lambda3 3.0 --tau 0.1 --hiera 13 --easy_pos  
```  
  
**Evaluation for ground-to-drone retrieval:**  
 ```  
python test_PL_G2D.py --batchsize 8 --test_dir $Your_Data_Root$ --name $Name_of_Step-3$ --gpu_ids 0 --low_altitude 
python evaluate_h5.py --name $Name_of_Step-3$  
```  
  
### Train & Evaluation for the Satellite-Drone representation  
  
**Train:**  
```  
python train_D2S.py --train_all --batchsize 8 --pool max --data_dir $Your_Data_Root$ --seed 0 --name $Name_of_D2S$ --gpu 0 --lambda1 5.0 --lambda2 0.5  
```  
  
**Evaluation for drone-to-satellite retrieval:**  
 ```  
python test_D2S.py --batchsize 8 --test_dir $Your_Data_Root$ --name $Name_of_D2S$ --gpu_ids 0  
python evaluate_h5.py --name $Name_of_D2S$  
```  
  
### Evaluation for ground-to-satellite retrieval by using cross-diffusion:  
```  
python test_D2S_for_diffusion.py --batchsize 8 --test_dir $Your_Data_Root$ --name $Name_of_D2S$ --gpu_ids 0 --low_altitude
cd diffusion  
python evaluate_dfs.py --name_g2d $Name_of_Step-3$ --name_d2s $Name_of_D2S$ --kq 20 --kd 70 --n_trunc 500 --hiera 0  
```  
  
### Results  
  
**Ground-to-Drone Retrieval:** R@1: 8.07 R@5: 13.18 R@10: 17.14 R@1%: 45.87 mAP: 6.09  
  
**Drone-to-Satellite Retrieval:** R@1: 70.09 R@5: 87.63 R@10: 91.38 R@1%: 91.80 mAP: 74.05  
  
**Ground-to-Satellite Retrieval:** R@1: 6.63 R@5: 14.73 R@10: 18.53 R@1%: 19.31 mAP: 8.80  
  
## Pretrained models
We also provide the pretrained models at [Dropbox](https://www.dropbox.com/sh/qtuecfcehkxaugq/AADgFFLfWikXhouFbvRxDCpfa?dl=0). 
If you want to use them, please download and put them into the 'PLCD/model/', then run the following commands: 

- Ground-to-Drone Retrieval:
```
python test_PL_G2D.py --batchsize 8 --test_dir $Your_Data_Root$ --name G2D --gpu_ids 0 --low_altitude 
python evaluate_h5.py --name G2D 
```
- Drone-to-Satellite Retrieval:
```
python test_D2S.py --batchsize 8 --test_dir $Your_Data_Root$ --name D2S --gpu_ids 0  
python evaluate_h5.py --name D2S  
```

- Ground-to-Satellite Retrieval:
```
python test_PL_G2D.py --batchsize 8 --test_dir $Your_Data_Root$ --name G2D --gpu_ids 0 --low_altitude 
python test_D2S_for_diffusion.py --batchsize 8 --test_dir $Your_Data_Root$ --name D2S --gpu_ids 0 --low_altitude
cd diffusion  
python evaluate_dfs.py --name_g2d G2D --name_d2s D2S --kq 20 --kd 70 --n_trunc 500 --hiera 0 
```
  
## Citation  
```bibtex  
@article{zeng2022geo,  
  title={Geo-localization via ground-to-satellite cross-view image retrieval},  
  author={Zeng, Zelong and Wang, Zheng and Yang, Fan and Satoh, Shin'ichi},  
  journal={IEEE Transactions on Multimedia},  
  year={2022},  
  publisher={IEEE}  
}  
```
