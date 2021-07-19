FROM continuumio/anaconda3
copy . /usr/app/
EXPOSE 8000
WORKDIR /usr/app/
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN pip install -r requirements.txt
CMD python flaskApp.py
