from django.db import models

# Create your models here.
class Forecasting(models.Model):
    Date_Choices = (('1일','1일'), 
               ('2일','2일'), 
               ('3일','3일'), 
               ('4일','4일'), 
               ('5일','5일'), 
               ('6일','6일'), 
               ('7일','7일'))
    item = models.CharField(max_length = 50)
    forecastinglen = models.CharField(max_length=10, choices=Date_Choices)
    
    def __str__(self):
        return f'{self.item}[예측일: {self.forecastinglen}]'
    


