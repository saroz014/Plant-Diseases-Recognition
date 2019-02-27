from django.urls import path
from rest_framework.documentation import include_docs_urls
from .views import Predict

urlpatterns = [
    path('', include_docs_urls(title='Plant Diseases API')),
    path('predict/', Predict.as_view(), name='predict'),
]