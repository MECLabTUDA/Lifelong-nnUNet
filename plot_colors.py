


colors = {
    ## baselines ##
    'Rehearsal': (0/255, 0/255, 0/255),
    'Sequential': (120/255, 120/255, 120/255),
    'LwF': (255/255, 201/255, 201/255),
    'EWC': (144/255, 28/255, 83/255),
    'MiB': (104/255, 57/255, 184/255),
    'cURL': (85/255, 147/255, 117/255),


    'between encoder and decoder': (255/255, 0/255, 0/255),
    'middle encoder': (0/255, 0/255, 255/255),
    'beginning decoder': (0/255, 255/255, 0/255),

    'between encoder and decoder, 0.1': (150/255, 0/255, 0/255),
    'between encoder and decoder, 0.25': (255/255, 0/255, 0/255),
    'between encoder and decoder, 0.5': (255/255, 180/255, 180/255),

    'middle encoder, 0.1': (0/255, 0/255, 150/255),
    'middle encoder, 0.25': (0/255, 0/255, 255/255),
    'middle encoder, 0.5': (180/255, 180/255, 255/255),

    'middle encoder, ground truth': (0/255, 0/255, 150/255),
    'middle encoder, distilled output': (180/255, 180/255, 255/255),
    #'ground truth': (152/255, 3/255, 252/255),
    #'distilled output': (255/255, 0/255, 208/255),


    # yellow, and purple, and green
    ## ablation ##
    'Feature rehearsal, distilled output, 2D, w/o skips, w/ freezing': (237/255, 36/255, 224/255),
    'upper bound': (237/255, 36/255, 224/255),
    'Feature Rehearsal': (156/255, 165/255, 23/255),

    'cVAEr': (255/255, 222/255, 60/255),
    'ccVAEr': (255/255, 170/255, 60/255),

    'VAE': (0,0,0),

    'Sequential, Seg. Dist.': (0/255, 0/255, 150/255),
    'Model Pool + Seg. Dist.': (180/255, 180/255, 255/255),
    'MiB, Softmax': (0/255, 0/255, 150/255)
}