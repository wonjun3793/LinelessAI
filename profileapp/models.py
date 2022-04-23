from django.db import models

# Create your models here.
from django.contrib.auth.models import User 


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    
    image = models.ImageField(upload_to='profile/', null=True)
    
    nickname = models.CharField(max_length=20, unique=True, null=False)