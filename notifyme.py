from pync import Notifier
import os

def notifyme(title="Training Complete", message="Your model finished training.", activate="com.microsoft.VSCode"):
    Notifier.notify(message, title=title, activate=activate)
    os.system(f'say {title}')