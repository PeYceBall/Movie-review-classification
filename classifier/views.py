from django.shortcuts import render
from django.http import HttpResponse
# from zetakappa.classifier.forms import CommentForm
from .forms import CommentForm
from django.contrib import messages
from .model.run_inference import classify_comment

# Create your views here.

def index(request):
    form = CommentForm()
    if request.method == 'POST':
        form = CommentForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['body']
            sentiment, score = classify_comment(text)
            msg = f'{"Позитивный" if sentiment else "Негативный"} отзыв. '
            msg += f'Оценка: {score}'
            messages.success(request, msg)

    context = {
        "form": form,
    }
    return render(request, "index.html", context)