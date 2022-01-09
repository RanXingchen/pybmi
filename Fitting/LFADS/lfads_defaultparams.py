default_hyperparams = {
    # ***MODEL PARAMETERS***
    # ==========================================
    # Hidden size of the generator.
    'g_dim':                        100,
    # Dimensionality of the inferred inputs to the generator.
    'u_dim':                        1,
    # Hidden size of the encoder for the generator.
    'enc_g_dim':                    100,
    # Hidden size of the encoder for the controller.
    'enc_c_dim':                    100,
    # Hidden size of the controller.
    'c_dim':                        100,
    # Variance of control/g0 input prior distribution.
    'kappa':                        0.1,
    # The dropout rate.
    'dropout':                      0.0,
    # Clips the hidden unit activity to be less than this value.
    'clip_val':                     5.0,
    # Maximum gradient norm.
    'max_norm':                     200,

    # ***OPTIMIZER PARAMETERS***
    # ==========================================
    # Batch size of each training step.
    'batch_size':                   64,
    'epoch':                        100,
    # Learning rate for ADAM optimizer.
    'lr':                           0.01,
    # Minimum learning rate. Stop training when the learning rate reaches
    # this threshold.
    'lr_min':                       1e-5,
    # Factor by which to decrease the learning rate if progress
    # isn't being made.
    'lr_decay':                     0.95,
    # Apply scheduler if True.
    'scheduler_on':                 True,
    # Number of steps without loss decrease before weight decay.
    'scheduler_patience':           6,
    # Number of steps after weight decay to wait before next weight decay.
    'scheduler_cooldown':           6,
    # Epsilon value form ADAM optimizer.
    'epsilon':                      0.1,
    # Beta values for ADAM optimizer.
    'betas':                        (0.9, 0.999),
    # Scaling factor for regularizing l2 norm of generator hidden weights.
    'l2_g_scale':                   500,
    # Scaling factor for regularizing l2 norm of controller hidden weights.
    'l2_c_scale':                   0.0,
    # Optimization step to start kl_weight increase.
    'w_kl_start':                   0,
    # Number of optimization steps to increase kl_weight to 1.0.
    'w_kl_dur':                     2000,
    # Optimization step to start l2_weight increase.
    'w_l2_start':                   0,
    # Number of optimization steps to increase l2_weight to 1.0.
    'w_l2_dur':                     2000
}
