# Generated by Django 4.0.3 on 2022-05-02 12:03

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('articleapp', '0002_delete_inquiry'),
    ]

    operations = [
        migrations.CreateModel(
            name='Inquiry',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=20, verbose_name='이름')),
                ('location', models.CharField(max_length=100, verbose_name='위치')),
                ('phone', models.IntegerField(max_length=20, verbose_name='헨드폰 번호')),
                ('email_address', models.EmailField(max_length=254, verbose_name='이메일 주소')),
            ],
        ),
    ]
