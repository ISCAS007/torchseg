# parameters
- add accumulate for batch size
- remove cascade config (config.dataset, config.aug, config.model, config.args)
- rename some variable to make config.xxx==args.xxx
- rename augmentations_rotate as use_rotate

# PSPUNet
- suggest upsample_layer=1, deconv_layer=5, assert upsample_layer < deconv_layer
- when additional_upsample is False, assert midnet_scale%2==0 for default pool_sizes [6,3,2,1]

# input_shape
- for pspnet: input_shape=f(midnet_scale,midnet_pool_sizes,upsample_layer,use_none_layer)
- for PSPUNet: input_shape=f(midnet_scale,midnet_pool_sizes,deconv_layer,use_none_layer)