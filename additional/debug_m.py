import matplotlib.pyplot as plt
import numpy as np
import cv2, torch
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.augmentations.utils import pad_nd_image
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch
from additional.model_restore import restore_model

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def load_model():
    checkpoint = '/gris/gris-f/homestud/aranem/Lifelong-nnUNet-storage/nnUNet_trained_models/nnUNet_ext/2d/Task099_HarP_Task098_Dryad_Task097_DecathHip/Task099_HarP_Task098_Dryad_Task097_DecathHip/nnUNetTrainerRehearsal__nnUNetPlansv2.1/Generic_ViT_UNetV2/base/MH/fold_0/model_final_checkpoint.model'
    pkl_file = checkpoint + ".pkl"
    trainer = restore_model(pkl_file, checkpoint, train=False, fp16=True, use_extension=True, extension_type='rehearsal', del_log=True)
    return trainer

def postprocess_activations(activations):
    activations = activations.numpy()


    activations = activations / activations.max()

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(activations.shape).numpy()
    joint_attentions[0] = activations[0]

    for n in range(1, activations.shape[0]):
        joint_attentions[n] = torch.matmul(activations[n], joint_attentions[n-1])
    v = joint_attentions[-1]

    mask = v[0]
    mask = cv2.resize(mask / mask.max(), (48, 64))#[..., np.newaxis]
    return mask


    # #using the approach in https://arxiv.org/abs/1612.03928
    # output = np.abs(activations)#.squeeze()[-1, :]
    # output = np.sum(output, axis = -1).squeeze()

    # # #resize and convert to image 
    # output = cv2.resize(output, (48, 64))
    # output /= output.max()
    # output *= 255
    # return 255 - output.astype('uint8')

def apply_heatmap(weights, img):
    # return cv2.addWeighted(weights, 0.9, img, 0.1, 0)
    
    #generate heat maps 
    # weights = (255-0)/(51-0)*(weights-51)+255
    # heatmap = cv2.applyColorMap(weights.astype(np.uint8), cv2.COLORMAP_JET)

    heatmapshow = None
    heatmapshow = cv2.normalize(weights, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    img = np.stack((img,)*3, axis=-1).astype(np.uint8)
    print(heatmap.shape, img.shape)
    heatmap = cv2.addWeighted(heatmap, 0.7, img, 0.3, 0)
    return heatmap

def plot_heatmaps(rng, img):
    level_maps = None
    
    #given a range of indices generate the heat maps 
    for i in rng:
      activations = activation['ViT.head'+'_'+str(i)]
      weights = postprocess_activations(activations.cpu())
      heatmap = apply_heatmap(weights, img)
      if level_maps is None:
        level_maps = heatmap
      else:
        level_maps = np.concatenate([level_maps, heatmap], axis = 1)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    ax = plt.imshow(level_maps)

    plt.savefig('/gris/gris-f/homestud/aranem/result.svg', bbox_inches = 'tight')  

def main():
    model = load_model()
    for i in range(12):
        getattr(model.network.ViT.blocks, str(i)).mlp.fc2.register_forward_hook(get_activation('ViT.head'+'_'+str(i)))
    model.network.to('cuda:5')

    # -- Validate -- #
    model.network.do_ds = False
    model.load_dataset()
    model.do_split()


    img = 's25_L'
    print(model.dataset.keys())
    data = np.load(model.dataset[img]['data_file'])['data']
    print(img, data.shape)
    data[-1][data[-1] == -1] = 0
    model.network.eval()
    data, _ = pad_nd_image(data[:, 53], model.patch_size, "constant", None, True, model.network.input_shape_must_be_divisible_by)
    data, img = data[0], data[0]
    data = maybe_to_torch(data[None])
    data = to_cuda(data, gpu_id=5)
    model.network(data.unsqueeze(0))
    
    plot_heatmaps([11], img)


if __name__ == "__main__":
    main()