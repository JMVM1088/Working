import smtplib
import os
import sys
import glob
from email.message import EmailMessage
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, time 

# --- CONFIGURATION ---
SENDER_EMAIL = "joekwong.stock@gmail.com"
SENDER_PASSWORD = "zzax iylg ykrb iimc"  # Your 16-character App Password
RECEIVER_EMAIL = "joeykwong128@gmail.com"
DEFAULT_LOG_DIR = r"C:\Trading\logs"

def send_email_old(attachment_path=None, body_text=None, subject_text=None):
    msg = EmailMessage()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    
    # Logic to determine which file to attach
    file_to_send = None
    
    if attachment_path and os.path.exists(attachment_path):
        # Use the specific file provided via command line
        file_to_send = attachment_path
        msg['Subject'] = f"Manual Attachment: {os.path.basename(file_to_send)}"
        #msg.set_content(f"Sending requested file: {os.path.basename(file_to_send)}")
        msg.set_content(body_text if body_text else "Please find the attached log file.")
    else:
        # Default: Find the latest log file
        list_of_files = glob.glob(os.path.join(DEFAULT_LOG_DIR, "*.txt"))
        if list_of_files:
            file_to_send = max(list_of_files, key=os.path.getctime)
            msg['Subject'] = f"Daily Stock Run: {os.path.basename(file_to_send)}"
            msg.set_content(f"The daily backtest is complete. Log attached.")
        else:
            msg['Subject'] = "Stock Run Complete (No Log Found)"
            msg.set_content("The script finished, but no log file was found to attach.")

    # Attach the file if one was found/provided
    if file_to_send:
        try:
            with open(file_to_send, 'rb') as f:
                file_data = f.read()
                file_name = os.path.basename(file_to_send)
                msg.add_attachment(
                    file_data,
                    maintype='application',
                    subtype='octet-stream',
                    filename=file_name
                )
        except Exception as e:
            print(f"Failed to read attachment: {e}")

    # Send the email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error: {e}")

def send_email(subject_text=None, body_text=None,attachment_path=None):
    msg = EmailMessage()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    
    # Logic to determine which file to attach
    file_to_send = None
    
    if attachment_path and os.path.exists(attachment_path):
        # Use the specific file provided via command line
        file_to_send = attachment_path
        msg['Subject'] = f"{subject_text}"
        #msg.set_content(f"Sending requested file: {os.path.basename(file_to_send)}")
        msg.set_content(body_text if body_text else "Please find the attached log file.")
    else:
        msg['Subject'] = f"{subject_text}"
        msg.set_content(body_text if body_text else "The script finished.")

    # Attach the file if one was found/provided
    if file_to_send:
        try:
            with open(file_to_send, 'rb') as f:
                file_data = f.read()
                file_name = os.path.basename(file_to_send)
                msg.add_attachment(
                    file_data,
                    maintype='application',
                    subtype='octet-stream',
                    filename=file_name
                )
        except Exception as e:
            print(f"Failed to read attachment: {e}")

    # Send the email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error: {e}")

def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
        )

        handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024, backupCount=5)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

