from urllib.parse import parse_qs

from ninecat.urlstate import PARAMS, PLAYER, link_target, player_href


def test_player_href_keeps_every_setting_and_round_trips():
    params = {"team": "BB Legends", "source": "season", "punt": ["AST", "FT%"], "disc": "false",
              "tab": "My team", PLAYER: "999"}
    href = player_href(4432166, params)
    assert href.startswith("?")
    back = parse_qs(href[1:])
    assert back == {"team": ["BB Legends"], "source": ["season"], "punt": ["AST", "FT%"], "disc": ["false"],
                    "tab": ["My team"], PLAYER: ["4432166"]}


def test_player_href_without_settings():
    assert player_href(7, {}) == "?player=7"


def test_link_target_maps_name_cells_to_player_ids():
    links = {"Player": [11, 12, 13], "Best drop": [21, 22, 23]}
    assert link_target([(1, "Player")], links) == 12
    assert link_target([(2, "Best drop")], links) == 23
    assert link_target([(0, "Value")], links) is None       # not a name column
    assert link_target([(5, "Player")], links) is None      # out of range
    assert link_target([], links) is None


def test_param_names_are_unique_and_never_player():
    names = list(PARAMS.values())
    assert len(names) == len(set(names)) and PLAYER not in names
