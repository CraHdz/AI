import logging
import os
import datetime

#获取日期
today = datetime.date.today()
strtoday = today.strftime("%y%m%d")

#配置logging
logging.basicConfig(filename="tools/nlog/" + strtoday + ".log", level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logging.info("logger starting")

def info(message_info):
    logging.info(message_info)

def warning(message_warning):
    logging.warning(message_warning)

def debug(message_debug):
    logging.debug(message_debug)

def error(message_error):
    logging.error(message_error)