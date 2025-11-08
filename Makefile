

unit-tests:
	coverage run -m pytest -s tests/unit
	coverage report
	coverage html


integration-tests:
	docker compose up -d
	AWS_ENDPOINT_URL=http://localhost:4566 AWS_S3_ENDPOINT_URL=http://localhost:4566 coverage run -m pytest -s tests/integration
	coverage report
	coverage html
