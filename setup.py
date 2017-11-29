from setuptools import setup

setup(
    name="submission_criteria",
    author="Numerai",
    install_requires=[
        "boto3",
        "botocore",
        "pandas==0.20.3",
        "tqdm",
        "scipy",
        "sklearn",
        "statsmodels",
        "python-dotenv",
        "bottle",
        "numpy",
        "pqueue",
        "randomstate",
        "psycopg2",
        "sqlalchemy",
        "mysqlclient",
        "pylint",
        "flake8"
    ],
)
