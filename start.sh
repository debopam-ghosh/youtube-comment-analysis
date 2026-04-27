export TF_ENABLE_ONEDNN_OPTS=0
gunicorn - w 5 -b 0.0.0.0 server:app
