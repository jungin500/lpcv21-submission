import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .model import Net

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        use_quantized_model = True

        self.device = torch.device("cuda") if torch.cuda.is_available() and use_cuda else torch.device("cpu")
        logger = logging.getLogger("root.tracker")

        if use_quantized_model and self.device.type == 'cpu':  # use_cuda might always be True!
            model_path = 'solution/deep_sort/deep_sort/deep/checkpoint/deep_lighterweight_qat_int8_model.torchscript'
            self.net = torch.jit.load(model_path)
            self.net.eval()
            logger.info("Deep: using torchscript with Quantization and Fusing")
        else:
            self.net = Net(reid=True)
            self.net.eval()
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
            self.net.load_state_dict(state_dict)
            logger.info("Loading weights from {}... Done!".format(model_path))
            logger.info("WARNING: Not using quantize/fused deep model!")
            self.net.to(self.device)

        self.size = (64, 128)
        self.means = np.expand_dims(np.expand_dims(np.array([0.485, 0.456, 0.406]), 0), 0)
        self.stds = np.expand_dims(np.expand_dims(np.array([0.229, 0.224, 0.225]), 0), 0)
        

    def __call__(self, im_crops):
        with torch.no_grad():
            im_batch = []
            for i, cropped_image in enumerate(im_crops):
                # cv2.imshow("Image %d" % i, cropped_image)
                # cv2.waitKey(1)

                image = cv2.resize(cropped_image, self.size)
                image = (image / 255.0).astype(np.float32)
                image -= self.means
                image /= self.stds
                image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
                im_batch.append(image)
            
            im_batch = torch.cat(im_batch, dim=0)

            if self.device != 'cpu':
                im_batch = im_batch.to(self.device)
            features = self.net(im_batch)

            if self.device.type != 'cpu':
                features = features.cpu()
            return features.numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:,:,(2,1,0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

