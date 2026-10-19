import math

from ninecat.theme import TOKENS, app_css, heat, heat_alpha, rgba, tokens


def test_heat_alpha_saturates_at_2_5():
    assert heat_alpha(0) == 0
    assert heat_alpha(1.25) == 0.21
    assert heat_alpha(2.5) == heat_alpha(-9) == 0.42


def test_heat_uses_good_and_bad_colors_per_mode():
    good, bad = rgba(TOKENS["light"]["good"], 0.42), rgba(TOKENS["dark"]["bad"], 0.42)
    assert heat(3.0, "light") == f"background-color: {good}"
    assert heat(-3.0, "dark") == f"background-color: {bad}"
    assert heat(math.nan) == heat(None) == heat("x") == ""


def test_rgba():
    assert rgba("#c2410c", 0.5) == "rgba(194,65,12,0.50)"


def test_tokens_default_to_light_and_css_carries_them():
    assert tokens(None) == tokens("light") != tokens("dark")
    css = app_css("dark")
    assert TOKENS["dark"]["accent"] in css and "Barlow Condensed" in css
