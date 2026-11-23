"""End-to-end headless Streamlit AppTest smoke tests."""

from streamlit.testing.v1 import AppTest


def test_app_boots_and_interacts_distributions_mode():
    at = AppTest.from_file("app.py", default_timeout=20)
    at.run()
    assert not at.exception

    # Toggle language to English
    at.radio(key="lang_toggle").set_value("en").run()
    assert not at.exception

    # Switch to Gamma distribution
    at.sidebar.selectbox(key="dist_selector").set_value("gamma").run()
    assert not at.exception

    # Switch to Binomial distribution and CDF view
    at.sidebar.selectbox(key="dist_selector").set_value("binomial").run()
    assert not at.exception
    at.sidebar.radio(key="dist_view_radio").set_value("cdf").run()
    assert not at.exception

    # Save and clear Ghost reference curve
    at.sidebar.button(key="save_ghost_dist").click().run()
    assert not at.exception
    at.sidebar.button(key="clear_ghost_dist").click().run()
    assert not at.exception


def test_app_custom_formula_mode_tfidf_and_inverse():
    at = AppTest.from_file("app.py", default_timeout=20)
    at.run()
    assert not at.exception

    # Switch to Custom Formula mode (starts on TF-IDF)
    at.sidebar.radio(key="mode_radio").set_value("custom").run()
    assert not at.exception

    # Switch scaling mode to Value-based
    at.sidebar.radio(key="scaling_mode_radio").set_value("value").run()
    assert not at.exception

    # Enter b = 1/a manually
    at.sidebar.text_input(key="custom_formula_text").set_value("b = 1/a").run()
    assert not at.exception

    # Save formula into session
    at.sidebar.button(key="btn_save_formula").click().run()
    assert not at.exception
    assert len(at.session_state["saved_formulas"]) == 1
