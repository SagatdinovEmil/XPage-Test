from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from .models import Images
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# Модель классификации
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,9),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1),
            nn.AvgPool2d(2,2)
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128,7),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1),
            nn.AvgPool2d(2,2)
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128,256,5),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1),
            nn.AvgPool2d(2,2)
            )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256,512,3),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1),
            nn.AvgPool2d(2,2)
            )
        
            
        self.fc1 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512*4*4,256),
        nn.Dropout(0.2),
        nn.LeakyReLU(0.1),
        nn.Linear(256,128),
        nn.Dropout(0.2),
        nn.LeakyReLU(0.1),
        nn.Linear(128,1),
        )
        
        self.act = nn.Sigmoid()

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc1(x)
        x = self.act(x)
        return x

    def predict(self, x):
        return torch.round(self.forward(x))

model = Model()
# Загрузим модель для дальнейшей работы
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("Model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
transform = transforms.Compose([
        transforms.Resize((120, 120)),
        transforms.ToTensor(),
])


def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        try:
            image = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            image_path = fs.url(filename)
            img_for_model = Image.open(image)
            img_for_model = transform(img_for_model)
            img_for_model = img_for_model.unsqueeze(0)
            with torch.no_grad():
                predicted_class = model.predict(img_for_model)
            if predicted_class == 1:
                animal = "собака"
            else:
                animal = "кот"
            Images.objects.create(image=filename, predicted_class=predicted_class)
            return render(request, 'upload_image.html', {'animal': animal})
        except Exception as e:
            print(e)
    
    return render(request, 'upload_image.html')


def view_results(request):
    results = Images.objects.all()
    return render(request, 'view_results.html', {'results': results})
