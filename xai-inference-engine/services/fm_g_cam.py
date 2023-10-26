import torch
from torch.nn import functional as F
from utils.grad_utils import GradUtils

class FMGCam:
    def __init__(self, model, layer_name, device):
        self.model = model
        self.layer_name = layer_name
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    def __call__(self, img_tensor, class_count=3, enhance=True, class_rank_index=None):

        img_tensor = img_tensor.to(self.device).unsqueeze(0)
        preds, sorted_pred_indices, grad_list, act_list = GradUtils.get_model_pred_with_grads(
            model=self.model, 
            img_tensor=img_tensor, 
            last_conv_layer=self.layer_name,
            class_count=class_count)

        heatmaps = GradUtils.weight_activations(act_list, grad_list)

        if class_rank_index is None:

            # Concatenation of activation maps based on top n classes
            heatmaps = torch.cat(heatmaps)

            # Filter the heatmap based on the maximum weighted activation along the channel axis
            hm_mask_indices = heatmaps.argmax(dim=0).unsqueeze(0)

            hm_3d_mask = torch.cat([hm_mask_indices for _ in range(heatmaps.size()[0])])

            hm_3d_mask = torch.cat(
                [(hm_3d_mask[index] == (torch.ones_like(hm_3d_mask[index])*index)).unsqueeze(0) for index in range(heatmaps.size()[0])]
            ).long()

            heatmaps *= hm_3d_mask

        else:

            heatmaps = heatmaps[class_rank_index]

        
        # L2 Normalisation of the heatmap soften the differences
        if enhance:
            heatmaps = F.normalize(heatmaps, p=2, dim=1)

        # relu on top of the heatmap
        heatmaps = F.relu(heatmaps)
        
        # Min-max normalization of the heatmap
        heatmaps = (heatmaps - torch.min(heatmaps))/(torch.max(heatmaps) - torch.min(heatmaps))
        
        # Process the generated heatmaps

        return preds, sorted_pred_indices, heatmaps.detach().cpu().numpy()