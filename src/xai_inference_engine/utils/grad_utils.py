import torch
from torch.nn import functional as F

class GradUtils:

    @staticmethod
    def get_model_pred_with_grads(model, img_tensor, last_conv_layer, class_count):
        # TODO: Search for the last convolutional layer
        
        grad_list = []
        act_list = []
        
        for train_param in model.parameters():
            train_param.requires_grad = True
            
        gradients = None
        activations = None

        def hook_backward(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output

        def hook_forward(module, args, output):
            nonlocal activations
            activations = output
            
            
        hook_backward = last_conv_layer.register_full_backward_hook(hook_backward, prepend=False)
        hook_forward = last_conv_layer.register_forward_hook(hook_forward, prepend=False)
        
        model.eval()
        
        preds =  model(img_tensor.unsqueeze(0))
        
        # Sort prediction indices
        sorted_pred_indices = torch.argsort(preds, dim=1, descending=True).squeeze(0)
        
        # Iterate through the top prediction indices
        for rank in range(class_count):
            preds[:, sorted_pred_indices[rank]].backward(retain_graph=True)
            grad_list.append(gradients)
            act_list.append(activations)
        
        hook_backward.remove()
        hook_forward.remove()
        
        for train_param in model.parameters():
            train_param.requires_grad = False
        
        return preds.squeeze().detach().cpu().numpy(), sorted_pred_indices, grad_list, act_list
    

    @staticmethod
    def weight_activations(activation_list, gradient_list):
        saliency_maps = []

        for index, activations in enumerate(activation_list):
            gradients = gradient_list[index]
            
            avg_pooled_gradients = torch.mean(
                gradients[0], # Size [1, 1024, 7, 7] TODO - make this support other models
                dim=[0, 2, 3]
            )

            # Weighting acitvation features (channels) using its related calculated Gradient
            for i in range(activations.size()[1]):
                activations[:, i, :, :] *= avg_pooled_gradients[i]

            # average the channels of the activations
            saliency_map = torch.mean(activations, dim=1).squeeze()
                    
            saliency_maps.append(saliency_map.unsqueeze(0).detach().cpu())

        avg_pooled_gradients = torch.mean(
            gradients[0], # Size [1, 1024, 7, 7] TODO - make this support other models
            dim=[0, 2, 3]
        )

        return saliency_maps