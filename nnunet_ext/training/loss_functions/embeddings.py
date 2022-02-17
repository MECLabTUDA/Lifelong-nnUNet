import torch

def pod_embed(embedding_tensor):
        # -- Calculate the POD embedding -- #
        w_p = torch.mean(embedding_tensor, -1)  # Over W: H × C width-pooled slices of embedding_tensor using mean
        h_p = torch.mean(embedding_tensor, -2)  # Over H: W × C height-pooled slices of embedding_tensor using mean
        return torch.cat((w_p, h_p), dim=1)     # Concat over C axis

def local_POD(h_, h_old, scales):
        # -- Calculate the local POD embedding using intermediate convolutional outputs -- #
        assert h_.size() == h_old.size(), "The embedding tensors of the current and old model should have the same shape.."
        
        # -- Initialize the embedding lists/tensors that are filled in the double for loop -- #
        POD_ = None
        POD_old = None
        # -- Extract the height and width of the current embeddings -- #
        W = h_.size(-1)
        H = h_.size(-2)
        # -- Calculate embeddings for every scale in scales -- #
        for scale in range(0, scales, 1):  # step size = 1
            # -- Calculate step sizes -- #
            w = int(W/(2**scale))
            h = int(H/(2**scale))

            # -- Throw an error if scale is too big resulting in a step size of 0 -- #
            assert w > 0 and h > 0,\
                "The number of scales ({}) are too big in such a way that during scale {} either the step size for H ({}) or W ({}) is 0..".format(scales, scale, h, w)

            # -- Loop through W and H in h and w steps -- #
            for i in range(0, W-w, w):
                for j in range(0, H-h, h):
                    # -- Calculate the POD embeddings for the extracted slice based on i and j -- #
                    pod_ = pod_embed(h_[..., i:i+w, j:j+h])
                    pod_old = pod_embed(h_old[..., i:i+w, j:j+h])
                    # -- In-Place concatenation of the POD embeddings along channels axis --> use last one sine they are different -- #
                    POD_ = pod_ if POD_ is None else torch.cat((POD_, pod_), dim=-1)                  # concat over last dim since those might be different
                    POD_old = pod_old if POD_old is None else torch.cat((POD_old, pod_old), dim=-1)   # concat over last dim since those might be different

        # -- Return the L2 distance between the POD embeddings based on their original implementation from here: -- #
        # -- https://github.com/arthurdouillard/CVPR2021_PLOP/blob/0fb13774735961a6cb50ccfee6ca99d0d30b27bc/train.py#L934 -- #
        layer_loss = torch.stack([torch.linalg.norm(p_ - p_o, dim=-1) for p_, p_o in zip(POD_, POD_old)])
        return torch.mean(layer_loss)