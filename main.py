from __future__ import annotations

from bot.commands import create_bot
from services.startup_service import initialize_app


def main() -> None:
    context = initialize_app()
    bot = create_bot(context)

    try:
        bot.run(context.settings.discord_bot_token)
    finally:
        context.close()


if __name__ == "__main__":
    main()
