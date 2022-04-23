from django.urls import path 
from django.views.generic import TemplateView
import articleapp.views 

app_name = 'articleapp'

urlpatterns = [
    path('list/', TemplateView.as_view(template_name='articleapp/list.html'), name='list'),
    path('product/', articleapp.views.product, name='product'),
    path('pricing/', articleapp.views.pricing, name='pricing'),
    path('inquiry/', articleapp.views.inquiry, name='inquiry'),
] 