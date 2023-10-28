from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class ImageUtils:

    @staticmethod
    def colourise_heatmaps(heatmaps, image_width=224, image_height=224):

        hm_colours = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1)]
        start_colour = (0, 0, 0)
        
        fused_heatmap_np = None
        
        for i, heatmap in enumerate(heatmaps):
            
            map_colours = [start_colour, hm_colours[i]]

            cmap_tp = LinearSegmentedColormap.from_list("Custom", map_colours, N=256)
            
            if len(heatmaps) == 1:
                cmap_tp = plt.get_cmap('jet')
            
            heatmap_image = Image.fromarray(np.uint8(heatmap*255), 'L').resize((image_width,image_height), resample=Image.BICUBIC)
            heatmap_np = cmap_tp(np.array(heatmap_image))[:, :, :3]
            
            if i == 0:
                fused_heatmap_np = heatmap_np
            else:
                fused_heatmap_np += heatmap_np

        fused_heatmap_np/=np.max(fused_heatmap_np)
                
        fused_heatmap_image = Image.fromarray(np.uint8((fused_heatmap_np * 255)), "RGB")
        
        return fused_heatmap_image
    
    @staticmethod
    def super_imposed_image(heatmaps, original_image, image_width=224, image_height=224, alpha=0.8):
        
        original_image = original_image.resize((image_width, image_height))
        original_image = Image.blend(original_image, heatmaps.convert("RGB"), alpha=alpha)

        return original_image