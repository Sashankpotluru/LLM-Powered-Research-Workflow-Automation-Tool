.PHONY: run test test-cov lint migrate setup clean

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	cp -n .env.example .env || true
	mkdir -p data/uploads data/faiss_index data/exports

run:
	streamlit run app.py

test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ -v --cov=core --cov=db --cov=utils --cov-report=html --cov-report=term

lint:
	python -m py_compile app.py
	python -m py_compile core/pdf_processor.py
	python -m py_compile core/rag_pipeline.py
	python -m py_compile core/summarizer.py
	python -m py_compile core/citation_extractor.py
	python -m py_compile core/cross_compare.py
	python -m py_compile db/models.py
	python -m py_compile db/database.py
	python -m py_compile utils/config.py

migrate:
	alembic upgrade head

migrate-create:
	alembic revision --autogenerate -m "$(msg)"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
