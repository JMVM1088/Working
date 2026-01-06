import smtplib
import os
import sys
import glob
from email.message import EmailMessage

# --- CONFIGURATION ---
SENDER_EMAIL = "joekwong.stock@gmail.com"
SENDER_PASSWORD = "zzax iylg ykrb iimc"  # Your 16-character App Password
RECEIVER_EMAIL = "your-email@gmail.com"
DEFAULT_LOG_DIR = r"C:\Trading\logs"

def send_email(attachment_path=None):
    msg = EmailMessage()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    
    # Logic to determine which file to attach
    file_to_send = None
    
    if attachment_path and os.path.exists(attachment_path):
        # Use the specific file provided via command line
        file_to_send = attachment_path
        msg['Subject'] = f"Manual Attachment: {os.path.basename(file_to_send)}"
        msg.set_content(f"Sending requested file: {os.path.basename(file_to_send)}")
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

if __name__ == "__main__":
    # Check if a filename was passed as an argument
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    send_email(path_arg)