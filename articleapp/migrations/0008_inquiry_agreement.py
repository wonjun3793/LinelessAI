# Generated by Django 4.0.3 on 2022-05-03 00:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('articleapp', '0007_alter_inquiry_phone'),
    ]

    operations = [
        migrations.AddField(
            model_name='inquiry',
            name='agreement',
            field=models.BooleanField(default=False, verbose_name='동의서'),
        ),
    ]
