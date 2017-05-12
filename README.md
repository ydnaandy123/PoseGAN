# Pose GAN

## Training note
- size of dataset matter
- gamma and learning rate is sensitive in BEGAN

### DCGAN Record
- DCGAN_city
    - low_diversity (6509 from 194 scenes)
    - batch_size: 32
    - image_size: 128x256
    
![result](./assests/DCGAN_city/test_000000.png)

![log](./assests/DCGAN_city/tensorbaord_city.png)
- DCGAN_city_coarse
    - high_diversity (19998 scenes)
    - batch_size: 32
    - image_size: 256x256
    
![result](./assests/DCGAN_city_coarse/test_000000.png)

![log](./assests/DCGAN_city_coarse/tensorbaord_city_coarse.png)

- DCGAN_lsp
    - high_diversity (11000 scenes)
    - batch_size: 32
    - image_size: 256x256
    
![result](./assests/DCGAN_lsp/test_000000.png)

![log](./assests/DCGAN_lsp/tensorboard_lsp.png)

- MPII
    - (28883)
### BEGAN Record

- BEGAN_city
    - gamma = 0.5
    - d_lr = 0.00008
    - g_lr = 0.00008
    - lr_lower_boundary = 0.00002
![result](./assests/BEGAN_city/290500_G.png)

![result](./assests/BEGAN_city/tensorboard_city_scale.png)

![result](./assests/BEGAN_city/tensorboard_city_misc.png)

- BEGAN_city_coarse
    - gamma = 0.5
    - d_lr = 0.00008
    - g_lr = 0.00008
    - lr_lower_boundary = 0.00002

![result](./assests/BEGAN_city_coarse/300500_G.png)

![result](./assests/BEGAN_city_coarse/tensorboard_city_coarse_scale.png)

![result](./assests/BEGAN_city_coarse/tensorboard_city_coarse_misc.png)

- BEGAN_city_coarse
    - gamma = 0.9
    - d_lr = 0.00008
    - g_lr = 0.00008
    - lr_lower_boundary = 0.00002

![result](./assests/BEGAN_city_coarse/gamma_0_9.png)

![result](./assests/BEGAN_city_coarse/tensorboard_city_coarse_0_9_scale.png)

![result](./assests/BEGAN_city_coarse/tensorboard_city_coarse_0_9_misc.png)

- BEGAN_lsp
    - gamma = 0.5    
    - d_lr = 0.00008
    - g_lr = 0.00008
    - lr_lower_boundary = 0.00002

![result](./assests/BEGAN_lsp/212000_G.png)

![result](./assests/BEGAN_lsp/tensorboard_lsp_scale.png)

![result](./assests/BEGAN_lsp/tensorboard_lsp_misc.png)


- BEGAN_lsp
    - gamma = 0.9    
    - d_lr = 0.00008
    - g_lr = 0.00008
    - lr_lower_boundary = 0.00002
    
![result](./assests/BEGAN_lsp/gamma_0_9.png)

![result](./assests/BEGAN_lsp/tensorboard_lsp_0_9_scale.png)

![result](./assests/BEGAN_lsp/tensorboard_lsp_0_9_misc.png)