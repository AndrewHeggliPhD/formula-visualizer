"""End-to-end headless Streamlit AppTest smoke tests."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_boots_theme_toggle_and_distributions():
    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()
    assert not at.exception

    # Toggle Dark Mode ON and back to Light Mode
    at.radio(key="theme_toggle").set_value("dark").run()
    assert not at.exception
    assert at.session_state["theme"] == "dark"

    at.radio(key="theme_toggle").set_value("light").run()
    assert not at.exception

    # Toggle language to English
    at.radio(key="lang_toggle").set_value("en").run()
    assert not at.exception

    # Switch through Gamma and Binomial distributions
    at.sidebar.selectbox(key="dist_selector").set_value("gamma").run()
    assert not at.exception

    at.sidebar.selectbox(key="dist_selector").set_value("binomial").run()
    assert not at.exception


def test_app_custom_formula_mode_tfidf_and_inverse():
    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()
    assert not at.exception

    # Switch to Custom Formula mode (starts on TF-IDF)
    at.sidebar.radio(key="mode_radio").set_value("custom").run()
    assert not at.exception

    # Switch to Dark Mode in Custom Formula mode
    at.radio(key="theme_toggle").set_value("dark").run()
    assert not at.exception

    # Enter b = 1/a manually
    at.sidebar.text_input(key="custom_formula_text").set_value("b = 1/a").run()
    assert not at.exception

    # Save formula into session
    at.sidebar.button(key="btn_save_formula").click().run()
    assert not at.exception
    assert len(at.session_state["saved_formulas"]) == 1
