


colors = {
    ## baselines ##
    'Rehearsal': (0/255, 0/255, 0/255),
    'Sequential': (120/255, 120/255, 120/255),
    'LwF': (255/255, 201/255, 201/255),

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
    'CVAEr': (255/255, 222/255, 60/255),
    'CCVAEr': (255/255, 170/255, 60/255),
}