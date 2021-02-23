#import tensorflow as tf
#from text.symbols import kor_symbols as symbols

def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = {
        'E': 512,
        'ref_enc_filters': [32, 32, 64, 64, 128, 128],
        'ref_enc_size': [3, 3],
        'ref_enc_strides': [2, 2],
        'ref_enc_pad': [1, 1],
        'ref_enc_gru_size': 512 // 2,

        # Style Token Layer
        'token_num': 56,
        'num_heads': 8,
        'n_mels': 80,
    }

  
    return hparams
