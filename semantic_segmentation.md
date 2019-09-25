# parameters
- add accumulate for batch size
- remove cascade config (config.dataset, config.aug, config.model, config.args)
- rename some variable to make config.xxx==args.xxx

# PSPUNet
- suggest upsample_layer=1, deconv_layer=5, assert upsample_layer < deconv_layer
- when additional_upsample is False, assert midnet_scale%2==0