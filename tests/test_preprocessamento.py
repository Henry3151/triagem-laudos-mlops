from triagem.preprocessamento import normalizar_texto


def test_remove_acentos_e_caixa():
    assert normalizar_texto("Hemorragia INTRACRANIANA à Direita") == (
        "hemorragia intracraniana a direita"
    )


def test_colapsa_espacos_e_quebras_de_linha():
    assert normalizar_texto("  derrame\n\tpleural   leve  ") == "derrame pleural leve"


def test_cedilha_e_til():
    assert normalizar_texto("Coração e pulmões") == "coracao e pulmoes"


def test_string_vazia():
    assert normalizar_texto("") == ""
