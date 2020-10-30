# reference
- https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
- [attention model code](https://github.com/ozan-oktay/Attention-Gated-Networks)
- [A Survey of the Recent Architectures of Deep Convolutional Neural Networks](https://arxiv.org/ftp/arxiv/papers/1901/1901.06032.pdf)
- [network architecture search for semantic segmentation](https://github.com/MenghaoGuo/AutoDeeplab)
- [用Attention玩转CV，一文总览自注意力语义分割进展] 机器之心 微信收藏
# files

# motion seg
## dataset
- dataset in ['FBMS','cdnet2014','segtrackv2','BMCnet','all','all2','all3']
- frame_gap can be zero for random frame gap, but cannot use random frame gap with optical flow
- use_part_number should be 0 when used for benchmark
- attention
there are still room for resdual attention, merge method for spatial and channel attention, share the weight for all attention networks or not

| parameter | choices | note |
| fusion_type | HR,LR,all,first,last | fusion feature from main backbone and aux backbone |
| filter_type | main,all | apply attention on main/all feature |
| filter_feature | aux,all | generate attention from aux/all feature |
| attention_type | s,c,sc,cs | s for spatial attention, c for channel attention, sc for spatial + channel attention and cs for channel + spatial attention |

## input format
- the aux input format, main input will always in.