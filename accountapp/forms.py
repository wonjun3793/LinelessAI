from django.contrib.auth.forms import UserCreationForm
from django import forms 
from django.contrib.auth.models import User

class AccountUpdateForm(UserCreationForm):
    class Meta:
        model = User
        fields = ['username']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.fields['username'].disabled = True