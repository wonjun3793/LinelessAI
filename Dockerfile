FROM python:3.9.0

RUN echo "testing123"

WORKDIR /home/

RUN git clone https://github.com/wonjun3793/LinelessAI.git

WORKDIR /home/mysite

COPY requirements.txt /tmp/

RUN pip3 install --requirement /tmp/requirements.txt

RUN echo "SECRET_KEY=django-insecure-77i)m0xyyjke!l2-t#&^c%w=vkqp5x*(sj@ht)t#)ksh&z&i9%" > .env

RUN python3 manage.py migrate

EXPOSE 8000

CMD ["python", "manage.py", "sunserver", "0.0.0.0:8000"]

