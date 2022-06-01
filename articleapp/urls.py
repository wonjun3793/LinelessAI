from django.urls import path 
from django.views.generic import TemplateView
import articleapp.views 

app_name = 'articleapp'

urlpatterns = [
    path('list/', TemplateView.as_view(template_name='articleapp/list.html'), name='list'),
    path('basketproduct/', articleapp.views.basketproduct, name='basketproduct'),
    path('AIproduct/', articleapp.views.AIproduct, name='AIproduct'),
    path('pricing/', articleapp.views.pricing, name='pricing'),
    path('inquiry/', articleapp.views.inquiry, name='inquiry'),
    path('otherinfo/', articleapp.views.otherinfo, name='otherinfo'),
    path('terms/', articleapp.views.terms, name='terms'),
    path('companyinfo/', articleapp.views.companyinfo, name='companyinfo'),
    path('teaminfo/', articleapp.views.teaminfo, name='teaminfo'),
    path('checkout/', articleapp.views.check, name='checkout'),
] 