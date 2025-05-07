from django.urls import path
from . import views

urlpatterns = [
    path('', views.landingpage, name = 'landingpage'),
    path('STTpage/', views.STTpage, name = 'STTpage'),
    path('speech-recognition/', views.speech_recognition, name='speech_recognition'),
]