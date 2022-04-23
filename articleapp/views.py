from django.shortcuts import render


# Create your views here.
def product(request):
    return render(request, 'articleapp/product.html')

def pricing(request):
    return render(request, 'articleapp/pricing.html')

def inquiry(request):
    return render(request, 'articleapp/inquiry.html')