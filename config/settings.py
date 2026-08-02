"""
Minimal Django settings for the Dhaka rent estimator.

Kept deliberately small: no user accounts, no database tables — the app
only loads a saved model and returns a prediction. Secrets and host
config read from environment variables so the same code runs locally
and on a host without edits.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# For a real deployment set DJANGO_SECRET_KEY in the host's env vars.
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY", "dev-only-insecure-key-change-me-in-production"
)

# DEBUG defaults to True locally; set DJANGO_DEBUG=0 on the host.
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"

# On the host, set DJANGO_ALLOWED_HOSTS to your domain,
# e.g. "rakibulhasan17.pythonanywhere.com"
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "*").split(",")
CSRF_TRUSTED_ORIGINS = [
    h for h in os.environ.get("DJANGO_CSRF_TRUSTED", "").split(",") if h
]

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "predictor",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": []},
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# The app never touches a database, but Django wants this key defined.
DATABASES = {}

STATIC_URL = "static/"

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Dhaka"
USE_I18N = True
USE_TZ = True

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
