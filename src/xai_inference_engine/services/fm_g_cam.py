import gc
import torch
from torch.nn import functional as F
from ..utils import GradUtils
import queue


class FMGCam:

    
    def __init__(self, model, layer_name, device=None):
        self.model = model
        self.layer_name = layer_name

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

        self.my_queue = queue.Queue()
    
    # def storeInQueue(f, self=self):
    #     def wrapper(*args):
    #         self.my_queue.put(f(*args))
    #     return wrapper

    # @storeInQueue
    def __call__(self, img_tensor, class_count=3, enhance=True, class_rank_index=None, act_mode="relu"):

        img_tensor = img_tensor.to(self.device)
        preds, sorted_pred_indices, grad_list, act_list = GradUtils.get_model_pred_with_grads(
            model=self.model, 
            img_tensor=img_tensor, 
            last_conv_layer=self.layer_name,
            class_count=class_count)

        saliency_maps = GradUtils.weight_activations(act_list, grad_list)

        if class_rank_index is None:

            # Concatenation of activation maps based on top n classes
            saliency_maps = torch.cat(saliency_maps)

            # Filter the saliency_map based on the maximum weighted activation along the channel axis
            hm_mask_indices = saliency_maps.argmax(dim=0).unsqueeze(0)

            hm_3d_mask = torch.cat([hm_mask_indices for _ in range(saliency_maps.size()[0])])

            hm_3d_mask = torch.cat(
                [(hm_3d_mask[index] == (torch.ones_like(hm_3d_mask[index])*index)).unsqueeze(0) for index in range(saliency_maps.size()[0])]
            ).long()

            saliency_maps *= hm_3d_mask

        else:

            saliency_maps = saliency_maps[class_rank_index]

        
        # L2 Normalisation of the saliency_map soften the differences
        if enhance:
            saliency_maps = F.normalize(saliency_maps, p=2, dim=1)

        # Activation on top of the saliency_map
        if act_mode == "relu":
            saliency_maps = F.relu(saliency_maps)
        elif act_mode == "gelu":
            saliency_maps = F.gelu(saliency_maps)
        elif act_mode == "elu":
            saliency_maps = F.elu(saliency_maps)
        
        # Min-max normalization of the saliency_map
        saliency_maps = (saliency_maps - torch.min(saliency_maps))/(torch.max(saliency_maps) - torch.min(saliency_maps))
        
        # Process the generated saliency_maps
        saliency_maps = saliency_maps.detach().cpu().numpy()

        gc.collect()
        self.my_queue.put((preds, sorted_pred_indices, saliency_maps))

        return preds, sorted_pred_indices, saliency_maps