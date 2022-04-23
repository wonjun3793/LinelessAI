from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.utils.decorators import method_decorator
from django.contrib.auth.decorators import login_required

from django.urls import reverse 
from django.views.generic import CreateView, DeleteView, DetailView, UpdateView
from django.contrib.auth.models import User 
from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse, reverse_lazy
from accountapp.forms import AccountUpdateForm
from accountapp.decorators import account_ownership_required

from accountapp.models import HelloWorld


has_ownership = [account_ownership_required, login_required]

@login_required
def hello_world(request):
    if request.method == "POST":
        
        temp = request.POST.get('hello_world_input')
        
        new_hello_world = HelloWorld()
        new_hello_world.text = temp 
        new_hello_world.save()
        
        return HttpResponseRedirect(reverse('accountapp:hello_world'))
    else:
        hello_world_list = HelloWorld.objects.all()
        return render(request, 'accountapp/hello_world.html', context={'hello_world_list': hello_world_list})
    


class AccountCreateView(CreateView):
    model = User
    form_class = UserCreationForm
    success_url = reverse_lazy('articleapp:list')
    template_name = 'accountapp/create.html'
    
    
class AccountDetailView(DetailView):
    model = User 
    context_object_name = 'target_user'
    template_name = 'accountapp/detail.html'

    
@method_decorator(has_ownership, 'get')
@method_decorator(has_ownership, 'post')
class AccountUpdateView(UpdateView):
    model = User
    form_class = AccountUpdateForm
    success_url = reverse_lazy('articleapp:list')
    template_name = 'accountapp/update.html'
    
    
@method_decorator(has_ownership, 'get')
@method_decorator(has_ownership, 'post')
class AccountDeleteView(DeleteView):
    model = User 
    success_url  = reverse_lazy('accountapp:login')
    template_name = 'accountapp/delete.html'