from services.fm_g_cam import FMGCam

class XAIInferenceEngine:
    def __init__(self, model, num_workers=1, device='cpu'):
        self.model = model
        self.num_workers = num_workers
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    def predict(self, input):
        raise NotImplementedError