from django.contrib import admin
from .models import Inquiry
# Register your models here.

@admin.register(Inquiry)
class InquiryAdmin(admin.ModelAdmin):
	list_display = ('name', 'location', 'phone', 'email_address', 'agreement')
	ordering = ('name',)
	search_fields = ('name', 'location')