from rest_framework import serializers

class ImageSerializer(serializers.Serializer):
    photo = serializers.ImageField(help_text='Image of a leaf')