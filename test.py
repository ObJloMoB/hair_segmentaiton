import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image

class Predicter:
    def __init__(self, model_path, img_size):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = torch.load(model_path).to(self.device).eval()
        self.preproc = transforms.Compose([transforms.Resize((img_size, img_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std= [0.229, 0.224, 0.225])])

    def predict(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.preproc(Image.fromarray(img))
        tensor = torch.unsqueeze(tensor, 0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor).permute(0, 2, 3, 1).cpu().numpy().squeeze()
        mask = (output.argmax(axis=2) * 255)


def main():
    pred = Predicter('/home/revotailserver/Documents/hair_segmentaiton/checkpoint_e25of30_lr1.000000E-04.pth', 256)
    pred.predict('/disk_b/LfwDataset/raw/images/Aaron_Eckhart_0001.jpg')


if __name__ == '__main__':
    main()




        
