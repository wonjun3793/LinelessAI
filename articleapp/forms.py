from django import forms
from django.forms import ModelForm, ValidationError
from .models import Inquiry

class InquiryForm(ModelForm):
    agreement = forms.BooleanField()
    class Meta:
        model = Inquiry
        fields = "__all__"
        labels = {
			'name': '',
			'phone': '',
			'email_address': '',
			'location': '',		
		}
        widgets = {
			'name': forms.TextInput(attrs={'class':'form-control', 'placeholder':'이름'}),
			'location': forms.TextInput(attrs={'class':'form-control', 'placeholder':'매장 위치'}),
			'phone': forms.TextInput(attrs={'class':'form-control', 'placeholder':'핸드폰 번호 (- 없이 작성해주시기 바랍니다)'}),
			'email_address': forms.EmailInput(attrs={'class':'form-control', 'placeholder':'이메일 주소'}),
		}
