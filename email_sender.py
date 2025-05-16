import os
import httpx
import json
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class ResendEmailSender:
    """Helper class to send emails via Resend API"""
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key from env or passed directly"""
        self.api_key = api_key or os.environ.get("RESEND_API_KEY")
        if not self.api_key:
            raise ValueError("Resend API key is required. Set RESEND_API_KEY environment variable or pass it to the constructor.")
        
        self.base_url = "https://api.resend.com"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def send_email(self, 
                        to: str, 
                        subject: str, 
                        html_content: str,
                        from_email: str = "no-reply@stephenhib.com"):
        """
        Send an email via Resend API
        
        Args:
            to: Recipient email
            subject: Email subject
            html_content: HTML content of the email
            from_email: Sender email (default: no-reply@stephenhib.com)
            
        Returns:
            dict: Response from Resend API
        """
        payload = {
            "from": from_email,
            "to": to,
            "subject": subject,
            "html": html_content
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/emails",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code >= 400:
                error_msg = f"Failed to send email: {response.text}"
                raise Exception(error_msg)
                
            return response.json()