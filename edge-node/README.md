ALLOWED_NODES=market_mall,centre_mall,place_reil,downtown_terminal,confederation_mall

uv run src/sender.py --node-id market_mall --folders test_058 test_059
uv run src/sender.py --node-id downtown_terminal --folders test_112 test_113
uv run src/sender.py --node-id confederation_mall --folders test_066
uv run src/sender.py --node-id place_reil --folders place_reil
uv run src/sender.py --node-id centre_mall --folders test_122 test_115




