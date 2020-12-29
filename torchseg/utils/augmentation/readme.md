# augmentation

### input_shape, output_shape
input_shape ==> network ==> output_shape 

input_shape: network input shape, 
    for vgg/resnet, the default input_shape is (224,224)

output_shape: network output shape,
    for most segmenation network, the output_shape == input_shape 
    except for upsample_type==lossless, output_shape maybe 4*input_shape 
     
### resize_shape
origin different image size ==> resize_shape

if the origin image size in dataset not a const, we need resize the origin image 
to the same size (resize_shape), and we may pad_for_crop
but in fact, the image size for cityscapes is const (1024,2048)

### min_crop_size, max_crop_size
min_crop_size < crop_size < max_crop_size 

we can set the lower and upper bounds for crop_size 
use crop_ratio or crop_size_step to generate different image crop

### aug_library 

semseg: image_size ==>(*) [0.5,2] ==>(crop) [max_crop_size] 
    max_crop_size = image_size/2  <==> crop_ratio = [0.25, 1]
    
    image size change, crop_size=max_crop_size not change
    
imgaug/pillow:  min_crop_size < crop_size < max_crop_size, 
    image_size not change, crop_size change