import pytest

from app.moderation import ModerationError, check_prompt


def test_check_prompt_blocks_default_blocklist_terms():
    with pytest.raises(ModerationError):
        check_prompt("a minor character eating a donut")


def test_check_prompt_allows_normal_text():
    check_prompt("Homer Simpson eating a donut")  # ne doit pas lever


def test_check_prompt_is_case_insensitive():
    with pytest.raises(ModerationError):
        check_prompt("A CHILD character")


def test_check_prompt_matches_whole_words_only():
    check_prompt("Hololive mascot and Childish Gambino")  # "loli" / "child" seulement à l'intérieur d'un mot


def test_check_prompt_blocks_plural_and_children():
    with pytest.raises(ModerationError):
        check_prompt("two children playing")
    with pytest.raises(ModerationError):
        check_prompt("some minors")
