install:
	pip install -r requirements.txt

db:
	python run.py database create

clean:
	python run.py database clean

cli:
	python run.py query

run:
	streamlit run streamlit_app.py
