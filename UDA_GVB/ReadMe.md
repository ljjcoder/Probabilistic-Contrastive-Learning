### Declaim
This project is developped based on [GVB](https://github.com/cuishuhao/GVB) 


################core code###########################
Our core code is located at the './clsda/trainers/trainer_gvb.py'
```
if self.using_NCE:
    outputs_target_strong = outputs_all[(src_batchsize + tgt_batchsize):]
    w_soft = F.softmax(outputs_target[0:self.num_NCE_instance], dim=1)
    s_soft = F.softmax(outputs_target_strong[0:self.num_NCE_instance], dim=1)
    w_s_soft = torch.cat([w_soft, s_soft])
    # print(w_s_soft.shape)
    # exit()
    targets_NCE = torch.from_numpy(np.array([x for x in range(self.num_NCE_instance)]))
    out1_x = w_s_soft
    NCE_2 = torch.mm(out1_x, out1_x.transpose(0, 1).contiguous())
    unit_1 = torch.eye(self.num_NCE_instance * 2).cuda()
    #
    NCE_2 = NCE_2 * (1 - unit_1) + (-100000) * unit_1
    gt_labels_cls_cross = torch.cat([targets_NCE + self.num_NCE_instance, targets_NCE]).cuda()
    nce_loss = F.cross_entropy(self.NCE_scale * NCE_2, gt_labels_cls_cross)
    loss += self.NCE_weight * nce_loss
    nce_loss_val = nce_loss.item()
else:
    nce_loss_val = 0
```

### Requirements

```
CUDA 10.1
torch>=1.8.1
torchvision>=0.9.1
pyyaml
```

set the project root as: /ghome/your_name/DA/UDA_test

data prepare: Images are stored in the folder with the same name as dataset name, and train/test split are stored in txt folder. Below are folders under "/ghome/your_name/DA/UDA_test/data/"

```
│officehome/
├──Art/
│  ├── Alarm_Clock
│  │   ├── 00001.jpg
│  │   ├── 00002.jpg
│  │   ├── ......
│  ├── Backpack
│  │   ├── 00001.jpg
│  │   ├── 00002.jpg
│  │   ├── ......
│  ├── ......
├──Clipart/
│  ├── Alarm_Clock
│  │   ├── 00001.jpg
│  │   ├── 00002.jpg
│  │   ├── ......
│  ├── Backpack
│  │   ├── 00001.jpg
│  │   ├── 00002.jpg
│  │   ├── ......
│  ├── ......
│txt/
├──officehome/
│  ├── labeled_source_images_Art.txt
│  ├── unlabeled_target_images_Clipart_0.txt
```

### Training and testing

Command line for training model on 1 GPU
```bash
CUDA_VISIBLE_DEVICES=0 bash ./experiments/scripts/uda_gvb_train_release.sh exp ./configs/gvb/gvb_officehome_A_C_fixmatch_nce.py
```


### License
This repository is released under the MIT License as found in the [LICENSE](LICENSE) file. For commercial use, please contact with the authors.

