# Pose GAN

## Training note
- DCGAN
    1. dataset better contain enough diversity
        - city
    2. if dataset is too messy,  G may not learn well
        - lsp
    3. heatmap...?
    4. bigger batch size better
        - heatmap_joint
- BEGAN    
    1. good training: balanced, k_t != 0, d_fake pulling around
        - balance = gamma * d_loss_real - g_loss(d_loss_fake)
        - k_t += lambda_k * balance
        - d_loss = d_loss_real - k_t * d_loss_fake
    2. D too weak -> **balance grow up** -> k_t grow up (D will focus on d_fake)
    3. G too weak (D too strong) -> **balance drop down** -> k_t drop down (avoid D mistakenly think he is strong)
    3. higher gamma -> higher diversity
    - summary:
        1. learning better lower (around 4e-4)
        2. 
    
### DCGAN Record

- DCGAN_face    
    - **not yet converge?**
    - batch_size: 32
    - image_size: 218x178
    
![result](./assests/DCGAN_face/test_000000.png)

![log](./assests/DCGAN_face/tensorboard_face.png)

- DCGAN_city
    - **low_diversity (6509 from 194 scenes)**
    - **g_loss can't converge, becuase similar scenes with slightly differences?**
    - batch_size: 32
    - image_size: 128x256
    
![result](./assests/DCGAN_city/test_000000.png)

![log](./assests/DCGAN_city/tensorbaord_city.png)

- DCGAN_city_coarse
    - **high_diversity (19998 scenes)**
    - **d_real is small, because view point is still similar?**
    - batch_size: 32
    - image_size: 256x256
    
![result](./assests/DCGAN_city_coarse/test_000000.png)

![log](./assests/DCGAN_city_coarse/tensorbaord_city_coarse.png)

- DCGAN_lsp
    - **wild high_diversity (11000 scenes)**
    - **too difficult for d_real?**
    - batch_size: 32
    - image_size: 256x256
    
![result](./assests/DCGAN_lsp/test_000000.png)

![log](./assests/DCGAN_lsp/tensorboard_lsp.png)


- DCGAN_heatmap
    - batch_size: 32
    - image_size: 128x256
    
![result](./assests/DCGAN_heatmap/test_000000.png)

![log](./assests/DCGAN_heatmap/tensorboard_heatmap.png)


- DCGAN_heatmap_joint
    - **accidentally collapse**
    - batch_size: 32
    - image_size: 64x64
    
![result](./assests/DCGAN_heatmap_joint/train_127801.png)

![result](./assests/DCGAN_heatmap_joint/train_147701.png)

![log](./assests/DCGAN_heatmap_joint/tensorboard_heatmap_joint.png)


- DCGAN_heatmap_joint
    - **batch_size: 256**
    - image_size: 64x64
    
![result](./assests/DCGAN_heatmap_joint/test_000000_batch_256.png)

![log](./assests/DCGAN_heatmap_joint/tensorboard_heatmap_joint_batch_256.png)

### BEGAN Record

- BEGAN_face
    - **d_fake keep pulling over each other**
    - **results keep both diversity and realistic**
    - **balanced, k_t not going to zero!**
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
    - **thougt dataset has more diversity, results still are low diversity**
    - **loss vibrate tensely**
    - gamma = 0.5
    - d_lr = 0.00008
    - g_lr = 0.00008
    - lr_lower_boundary = 0.00002
    
![result](./assests/BEGAN_city_coarse/300500_G.png)

![result](./assests/BEGAN_city_coarse/tensorboard_city_coarse_scale.png)

![result](./assests/BEGAN_city_coarse/tensorboard_city_coarse_misc.png)

- BEGAN_city_coarse_gamma
    - **higher gamma -> higher k_t -> focus d_fake -> d_fake larger 
    -> higher diversity results -> not very realistic**
    - **d_fake keep pulling over each other**
    - **gamma = 0.9**
    - d_lr = 0.00008
    - g_lr = 0.00008
    - lr_lower_boundary = 0.00002

![result](./assests/BEGAN_city_coarse/gamma_0_9.png)

![result](./assests/BEGAN_city_coarse/tensorboard_city_coarse_0_9_scale.png)

![result](./assests/BEGAN_city_coarse/tensorboard_city_coarse_0_9_misc.png)

- BEGAN_city_coarse_lr
    - **lower learning rate seems perform better**
    - **d_fake keep pulling over each other**
    - gamma = 0.5
    - **d_lr = 0.00004**
    - **g_lr = 0.00004**
    - lr_lower_boundary = 0.00002

![result](./assests/BEGAN_city_coarse/lr_half.png)

![result](./assests/BEGAN_city_coarse/tensorboard_city_coarse_lr_half_scale.png)

![result](./assests/BEGAN_city_coarse/tensorboard_city_coarse_lr_half_misc.png)

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


- BEGAN_lsp_gamma
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

- BEGAN_lsp_lr
    - **lower learning rate seems perform better**
    - gamma = 0.5
    - **d_lr = 0.00004**
    - **g_lr = 0.00004**
    - lr_lower_boundary = 0.00002
    
![result](./assests/BEGAN_lsp/lr_half.png)

![result](./assests/BEGAN_lsp/tensorboard_lsp_lr_half_scale.png)

![result](./assests/BEGAN_lsp/tensorboard_lsp_lr_half_misc.png)

- BEGAN_heatmap
    - **failed**
    - gamma = 0.5
    - **d_lr = 0.00004**
    - **g_lr = 0.00004**
    - lr_lower_boundary = 0.00002
    
![result](./assests/BEGAN_heatmap/71000_G.png)

![result](./assests/BEGAN_heatmap/tensorboard_heatmap_scale.png)

![result](./assests/BEGAN_heatmap/tensorboard_heatmap_misc.png)


- BEGAN_heatmap_joint
    - **lower learning rate seems perform better**
    - gamma = 0.5
    - **d_lr = 0.00004**
    - **g_lr = 0.00004**
    - lr_lower_boundary = 0.00002
    
![result](./assests/BEGAN_heatmap_joint/71000_G.png)

![result](./assests/BEGAN_heatmap_joint/tensorboard_heatmap_joint_scale.png)

![result](./assests/BEGAN_heatmap_joint/tensorboard_heatmap_joint_misc.png)