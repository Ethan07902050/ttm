config = dict(
    
    # training
    load_saved_model = True,
    num_epoch = 200,
    batch_size = 8,
    learning_rate = 3e-4,

    # video
    in_channels = 3, # image
    image_H = 96,
    image_W = 96,

    # audio 
    audio_dim = 64, # number of mfcc coef

    # model
    dim = 128, 
    patch_size_h = 32, # patch size must be devisive by image size
    patch_size_w = 32,
    max_num_frames = 256, # input max seq len
    depth = 6,
    heads = 4,
    pool = 'cls',
    dim_head = 64,
    dropout = 0.2,
    emb_dropout = 0.2,
    scale_dim = 4,
    audio_scale = 4, # for vivit_w_audio_v2

    # loader
    num_workers = 4,
)