# Generated by Django 4.0.3 on 2022-05-02 12:38

from django.db import migrations
import phonenumber_field.modelfields


class Migration(migrations.Migration):

    dependencies = [
        ('articleapp', '0004_alter_inquiry_phone'),
    ]

    operations = [
        migrations.AlterField(
            model_name='inquiry',
            name='phone',
            field=phonenumber_field.modelfields.PhoneNumberField(max_length=128, region=None, unique=True, verbose_name='헨드폰 번호'),
        ),
    ]
