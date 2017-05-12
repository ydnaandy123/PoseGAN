# Pose GAN

## Training note
- size of dataset matter
- gamma and learning rate is sensitive in BEGAN

### DCGAN Record

- DCGAN_face
    - **for comparison to BEGAN**    
    - **similar scenes with slightly differences**
    - **g_loss not yet converge?**
    - batch_size: 32
    - image_size: 218x178
    
![result](./assests/DCGAN_face/test_000000.png)

![log](./assests/DCGAN_face/tensorboard_face.png)

- DCGAN_city
    - **low_diversity (6509 from 194 scenes)**    
    - **similar scenes with slightly differences**
    - **g_loss can't converge**
    - batch_size: 32
    - image_size: 128x256
    
![result](./assests/DCGAN_city/test_000000.png)

![log](./assests/DCGAN_city/tensorbaord_city.png)

- DCGAN_city_coarse
    - **high_diversity (19998 scenes)**
    - **but view point is similar**
    - **d_real is small**
    - batch_size: 32
    - image_size: 256x256
    
![result](./assests/DCGAN_city_coarse/test_000000.png)

![log](./assests/DCGAN_city_coarse/tensorbaord_city_coarse.png)

- DCGAN_lsp
    - **wild high_diversity (11000 scenes)**
    - **working well, but too difficult?**
    - batch_size: 32
    - image_size: 256x256
    
![result](./assests/DCGAN_lsp/test_000000.png)

![log](./assests/DCGAN_lsp/tensorboard_lsp.png)

### BEGAN Record

- BEGAN_face
    - **balanced, k_t not going to zero!**
    - **d_fake keep pulling over each other**
    - **results keep both diversity and realistic balanced**
    - gamma = 0.5
    - d_lr = 0.00008
    - g_lr = 0.00008
    - lr_lower_boundary = 0.00002
    
![result](./assests/BEGAN_face/29500_G.png)

![result](./assests/BEGAN_face/tensorboard_face_scale.png)

![result](./assests/BEGAN_face/tensorboard_face_misc.png)

- BEGAN_city
    - **low diversity result**
    - **loss vibrate slightly**
    - gamma = 0.5
    - d_lr = 0.00008
    - g_lr = 0.00008
    - lr_lower_boundary = 0.00002
    
![result](./assests/BEGAN_city/290500_G.png)

![result](./assests/BEGAN_city/tensorboard_city_scale.png)

![result](./assests/BEGAN_city/tensorboard_city_misc.png)

- BEGAN_city_coarse
    - **thougt dataset has more diversity**
    - **results still are low diversity**
    - **loss vibrate tensely**
    - gamma = 0.5
    - d_lr = 0.00008
    - g_lr = 0.00008
    - lr_lower_boundary = 0.00002
    
![result](./assests/BEGAN_city_coarse/300500_G.png)

![result](./assests/BEGAN_city_coarse/tensorboard_city_coarse_scale.png)

![result](./assests/BEGAN_city_coarse/tensorboard_city_coarse_misc.png)

- BEGAN_city_coarse
    - **higher gamma -> higher k_t -> focus d_fake -> d_fake larger 
    -> higher diversity results -> not very realistic**
    - **gamma = 0.9**
    - d_lr = 0.00008
    - g_lr = 0.00008
    - lr_lower_boundary = 0.00002

![result](./assests/BEGAN_city_coarse/gamma_0_9.png)

![result](./assests/BEGAN_city_coarse/tensorboard_city_coarse_0_9_scale.png)

![result](./assests/BEGAN_city_coarse/tensorboard_city_coarse_0_9_misc.png)

- BEGAN_lsp
    - **almost same with city_coarse**
    - **thougt dataset has more diversity, results still are low diversity**
    - **loss vibrate tensely**
    - gamma = 0.5    
    - d_lr = 0.00008
    - g_lr = 0.00008
    - lr_lower_boundary = 0.00002

![result](./assests/BEGAN_lsp/212000_G.png)

![result](./assests/BEGAN_lsp/tensorboard_lsp_scale.png)

![result](./assests/BEGAN_lsp/tensorboard_lsp_misc.png)


- BEGAN_lsp
    - **differ from city_coarse, the model seems to collapse**
    - **higher gamma -> higher k_t -> focus d_fake -> d_fake larger 
    -> higher diversity results -> not very realistic**
    - **gamma = 0.9** 
    - d_lr = 0.00008
    - g_lr = 0.00008
    - lr_lower_boundary = 0.00002
    
![result](./assests/BEGAN_lsp/gamma_0_9.png)

![result](./assests/BEGAN_lsp/tensorboard_lsp_0_9_scale.png)

![result](./assests/BEGAN_lsp/tensorboard_lsp_0_9_misc.png)