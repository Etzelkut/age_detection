from argparse import Namespace

re_dict = {
    "path_to_files": "/content",
    "batch_size": 64,
    "num_workers": 4, 
    #'pin_memory': True,
    "patch_size": 16,
    "im_size": 256, 
    "grayscale": False,
    "category": "age", #gender - 2 classes, age - 8 class
    "num_classes": 8, 
    "fold_numbers": 2,
    #
    "d_model_emb": 1024,
    "d_ff":2048,
    "heads": 4,
    "dropout": 0.05,
    "encoder_number": 6,
    #!
    "local_heads": 2, #more that 0 enable local heads
    "add_sch": False,
    #!
    #
    "local_window_size": 256,
    "attention_type": "selfatt", #performer, selfatt, linear
    "feedforward_type": "glu", # classic, glu
    #
    "learning_rate": 3e-4,
    "epochs": 100, 
    #
    "older_count": False,

}

hyperparams = Namespace(**re_dict)
