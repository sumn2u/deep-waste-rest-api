# This is my own base image from the Docker Hub
# See https://github.com/sumn2u/ml-model-base-docker
# FROM iamsuman/ml-model-base:latest

# Parent image to run under Nginx+uWSGI
# Python 3.10 on Debian
# FROM tiangolo/uwsgi-nginx-flask:python3.10

# Parent image to run under Meinheld+Gunicorn
# Python 3.9 on Debian
FROM tiangolo/meinheld-gunicorn-flask:python3.9

LABEL maintainer="info@sumankunwar.com.np"

COPY requirements.txt /app/

RUN pip install --upgrade pip \
 && pip install -r /app/requirements.txt

COPY . /app
WORKDIR /app

# Uncomment to set listen port for Nginx+uWSGI
# ENV LISTEN_PORT 8888

# Uncomment to set listen port for Meinheld+Gunicorn
ENV PORT 8888
ENV BIND 0.0.0.0:8888

EXPOSE 8888
ENV PYTHONPATH "${PYTHONPATH}:/app/ml_rest_api"


# Start the Gunicorn server
CMD ["gunicorn", "--worker-class=gthread", "-b", "0.0.0.0:8888", "ml_rest_api.app:APP"]


# Comment the two lines below to run under Nginx+uWSGI
# ENTRYPOINT ["python3"]
# CMD ["ml_rest_api/app.py"]
