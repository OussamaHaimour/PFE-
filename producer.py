import imaplib
import email
from email.header import decode_header
import time
import json
import os
from kafka import KafkaProducer
from dotenv import load_dotenv

load_dotenv()

EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def decode_mime_words(s):
    decoded = decode_header(s)
    return ''.join(
        str(t[0], t[1] or 'utf-8') if isinstance(t[0], bytes) else t[0]
        for t in decoded
    )

def fetch_unseen_emails():
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(EMAIL, PASSWORD)
    mail.select("inbox")

    status, messages = mail.search(None, '(UNSEEN)')
    email_ids = messages[0].split()

    results = []

    for e_id in email_ids:
        _, msg_data = mail.fetch(e_id, "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1])

        subject = decode_mime_words(msg["Subject"]) if msg["Subject"] else ""
        body = ""

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode(errors="ignore")
                    break
        else:
            body = msg.get_payload(decode=True).decode(errors="ignore")

        results.append({
            "email": body,
            "subject": subject
        })

    mail.logout()
    return results

while True:
    emails = fetch_unseen_emails()

    for e in emails:
        producer.send("helpdesk_emails", e)

    producer.flush()
    time.sleep(10)