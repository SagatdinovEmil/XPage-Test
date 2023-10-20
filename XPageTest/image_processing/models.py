from django.db import models

class Images(models.Model):
    image = models.ImageField()
    predicted_class = models.BooleanField()
    timestamp = models.DateTimeField(auto_now_add=True)
