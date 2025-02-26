# DT2GS
[NeurIPS 2023] -- Decompose a Task into Generalizable Subtasks in Multi-Agent Reinforcement Learning ([DT2GS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f7d3cef7ff579f2f903c8f458e730cae-Abstract-Conference.html))

## 1.Installation
---
```shell
# install on-policy package
cd DT2GS
pip install -e .
```
## 2.Train/Transfer
---
### run mappo
```shell
cd onpolicy/scripts/train_mappo_sh
chmod +x ./train_smac_3s5z_vs_3s6z.sh
./train_smac_3s5z_vs_3s6z.sh
```

### run DT2GS
```shell
# run DT2GS or DT2GS_transfer
cd onpolicy/scripts/train_DT2GS_sh
chmod +x ./train_smac_subtask.sh
./train_smac_subtask.sh

# or
chmod +x ./transfer_smac_subtask.sh
./transfer_smac_subtask.sh
```
### run UPDeT
```shell
# run UPDeT or UPDeT_transfer
cd onpolicy/scripts/train_DT2GS_sh
chmod +x ./train_smac_entity.sh
./train_smac_entity.sh

# or
chmod +x ./transfer_smac_entity.sh
./transfer_smac_entity.sh
```
### run ASN_G or ASN_G_transfer
```shell
# run ASN_G or ASN_G_transfer
cd onpolicy/scripts/train_DT2GS_sh
chmod +x ./train_smac_asn_gatten.sh
./train_smac_asn_gatten.sh

# or
chmod +x ./transfer_smac_asn_gatten.sh
./transfer_smac_asn_gatten.sh
```

### run ASN
```shell
cd onpolicy/scripts/train_DT2GS_sh
chmod +x ./train_smac_asn.sh
./train_smac_asn.sh
```


### BibTeX
---
If you find our models useful, please consider citing our paper!
```
@article{tian2023decompose,
  title={Decompose a task into generalizable subtasks in multi-agent reinforcement learning},
  author={Tian, Zikang and Chen, Ruizhi and Hu, Xing and Li, Ling and Zhang, Rui and Wu, Fan and Peng, Shaohui and Guo, Jiaming and Du, Zidong and Guo, Qi and others},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={78514--78532},
  year={2023}
}
```