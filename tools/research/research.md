# research

### the influence of downsample and upsample on miou

| scale | miou |
| -      | -     |
| 2 | 0.9843 |
| 3 | 0.9551 |
| 4 | 0.9540 |
| 5 | 0.9442 |
| 6 | 0.9132 |
| 7 | 0.9043 |
| 8 | 0.9019 |

| scale | miou |
| -      | -     |
| 0.9 | 0.9718 |
| 0.8 | 0.9642 |
| 0.7 | 0.9730 |
| 0.6 | 0.9633 |
| 0.5 | 0.9843 |


| size | miou |
| -      | -     |
| (224, 224) | 0.8936 |
| (256, 256) | 0.9234 |
| (256, 512) | 0.9354 |
| (512, 1024) | 0.9723 |

### optimizer
- https://github.com/ivallesp/awesome-optimizers

SGD+Momentum (momentum < 1, default value 0.9, for pytorch, momentum=0 means not use)
SGD+Nesterov Momentum (set nesterov=True)