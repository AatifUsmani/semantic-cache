import app


def test_import_does_not_load_embedding_model_or_connect_to_redis():
    assert app.model is None
