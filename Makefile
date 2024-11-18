install:
	pip install -r requirements.txt

run:
ifeq ($(OS),Windows_NT)
	set FLASK_APP=app.py && flask run --port=3000
else
	FLASK_APP=app.py flask run --port=3000
endif