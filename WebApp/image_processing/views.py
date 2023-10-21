from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponseNotFound
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
            nn.Conv2d(3,32,3),
            nn.Dropout(0.2), # Добавляем Dropout, чтобы избежать переобучения
            nn.ReLU(),
            # Испоьзуем AvgPool, чтобы уменьшить размер изображения и 
            # уменьшить требуемое количество вычислительных ресурсов
            nn.AvgPool2d(2,2)
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,3),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.AvgPool2d(2,2)
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128,3),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.AvgPool2d(2,2)
            )
            
        self.fc1 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128*13*13, 256),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(128,1),
        )
        
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x)
        x = self.act(x)
        return x

    def predict(self, x):
        return torch.round(self.forward(x))

model = Model()
# Загрузим модель для дальнейшей работы
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("image_processing/Model.pth", map_location=device)
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
            print(filename)
            image_path = fs.url(filename)
            print(image_path)
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
