# Generated by Django 4.2.6 on 2023-10-20 16:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('image_processing', '0005_alter_images_image'),
    ]

    operations = [
        migrations.AlterField(
            model_name='images',
            name='image',
            field=models.ImageField(upload_to=''),
        ),
    ]
