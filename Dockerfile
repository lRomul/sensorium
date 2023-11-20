FROM osaiai/dokai:23.10-pytorch

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
