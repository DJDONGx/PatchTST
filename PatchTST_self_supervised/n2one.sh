
# N-to-One Setting script

device=5
root_path_name=./saved_models/
model_suffixes=/masked_patchtst/based_model/patchtst_pretrained_cw512_patch12_stride12_epochs-pretrain10_mask0.4_model1.pth

pretrain_dataset=etth1_etth2
target_dataset=electricity

model_path_name=$root_path_name$pretrain_dataset$model_suffixes


#python patchtst_pretrain.py --multi_dset pretrain_dataset
#python patchtst_finetune.py --dset target_dataset --pretrained_model model_path_name --is_finetune 1



















