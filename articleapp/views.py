from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import InquiryForm

# Create your views here.
def basketproduct(request):
    return render(request, 'articleapp/basketproduct.html')

def AIproduct(request):
    return render(request, 'articleapp/AIproduct.html')

def check(request):
    return render(request, 'articleapp/check.html')

def pricing(request):
    return render(request, 'articleapp/pricing.html')

def inquiry(request):
    submitted = False
    if request.method == "POST":
        form = InquiryForm(request.POST)
        if form.is_valid():
            form.save()
            return HttpResponseRedirect('/articles/inquiry?submitted=True')
    else:
        form = InquiryForm
        if 'submitted' in request.GET:
            submitted = True
    form = InquiryForm
    return render(request, 'articleapp/inquiry.html', {'form':form, 'submitted':submitted})



def companyinfo(request):
    return render(request, 'articleapp/companyinfo.html')

def otherinfo(request):
    return render(request, 'articleapp/otherinfo.html')

def terms(request):
    return render(request, 'articleapp/terms.html')

def teaminfo(request):
    return render(request, 'articleapp/teaminfo.html')

