# ------------------------------------------------------------------------------
# Telegram bot to pass messages about the training, or inform when experiments 
# are done. Follow these links to get a token and chat-id
# - https://www.christian-luetgens.de/homematic/telegram/botfather/Chat-Bot.htm
# - https://stackoverflow.com/questions/32423837/telegram-bot-how-to-get-a-group-chat-id
# ------------------------------------------------------------------------------

import telegram as tel

class TelegramBot():
    r"""Initialize a telegram bot.
    Args:
        login_data (dict[str -> str]): dictionary with the entries 'chat_id'
            and 'token'

    """
    def __init__(self, login_data = None):
        self.chat_id = login_data['chat_id']
        self.bot = tel.Bot(token=login_data['token'])

    def send_msg(self, msg):
        r"""Send a message in string form"""
        self.bot.send_message(chat_id=self.chat_id, text=msg)
