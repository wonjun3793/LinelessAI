from django.forms import ModelForm
from django import forms
from AIpredictionapp.models import Forecasting

class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField
    