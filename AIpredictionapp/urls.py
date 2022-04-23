from django.urls import path 
from django.views.generic import TemplateView
import AIpredictionapp.views


app_name = "AIpredictionapp"

urlpatterns = [
    path('AIservice/', AIpredictionapp.views.AIservice, name='AIservice'),
    path('Dataresults/', AIpredictionapp.views.AIservice, name='Dataresults'),
] 
