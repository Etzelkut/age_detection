from argparse import Namespace

re_dict = {
    "path_to_file": "",
    "batch_size": 2,
    "num_workers": 4, 
    #'pin_memory': True,
    "patch_size": 16,
    "im_size": 256, 
    "grayscale": False,
    "num_classes": 10,


    "d_model_emb": 512,
    "d_ff":1024,
    "heads": 4,
    "dropout": 0.05,
    "encoder_number": 4,
    #!
    "local_heads": 2, #more that 0 enable local heads
    "add_sch": False,
    #!
    #
    "local_window_size": 256,
    "attention_type": "performer", #performer, selfatt, linear
    "feedforward_type": "glu", # classic, glu
    #
    "learning_rate": 3e-4,
    "epochs": 100, 
    #

}

hyperparams = Namespace(**re_dict)
