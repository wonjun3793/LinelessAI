from django.db import models
from django.views.generic import ListView

# Create your models here.

class Inquiry(models.Model):
    name = models.CharField('이름', max_length = 20, blank = False)
    location = models.CharField('위치', max_length = 100, blank = False)
    phone = models.CharField('헨드폰 번호',max_length = 20, blank = False)
    email_address = models.EmailField('이메일 주소', blank=False)
    agreement = models.BooleanField('동의서', default=False, blank = False)
    
    def __str__(self):
        return self.name 