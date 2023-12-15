import os
import sys
from datetime import date
from typing import Any, Dict, Optional, List

import aiohttp
from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.utils import get_from_dict_or_env
from requests import Request
from superhote_client import Client, AuthenticatedClient
from superhote_client.api.default import (post_api_v2_user_login, get_api_v2_get_all_statuses,
                                          get_api_v2_default_templates, get_api_v2_get_calendars,
                                          get_api_v2_get_current_bookings, get_api_v2_tasks,
                                          get_api_v2_get_statuses, get_api_v2_rentals, get_api_v2_custom_short_codes,
                                          get_api_v2_channels, get_api_v2_messages_list,
                                          post_api_v2_get_not_available_dates, get_api_v2_booking_templates_booking_id,
                                          get_api_v2_payments, post_api_v2_payments, get_api_v2_get_airbnb_listings,
                                          post_api_v2_get_availabilities, post_api_v2_bookings, post_api_v2_messages,
                                          post_api_v2_payments_send_payment, post_api_v2_send_invoice,
                                          get_api_v2_bookings_booking_id, get_api_v2_bookings_get_by_selected_day_date,
                                          get_api_v2_rentals_rental_id
                                          )
from superhote_client.models import (
    UserLoginBody, V2GetnotavailabledatesBody, GetApiV2PaymentsFilterByStatus,
    V2PaymentsBody, V2GetavailabilitiesBody, V2BookingsBody, V2MessagesBody, V2SendinvoiceBody, PaymentsSendpaymentBody
)


class HiddenPrints:
    """Context manager to hide prints."""

    def __enter__(self) -> None:
        """Open file to pipe stdout to."""
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, *_: Any) -> None:
        """Close file that stdout was piped to."""
        sys.stdout.close()
        sys.stdout = self._original_stdout


class SuperhoteAPIWrapper(BaseModel):
    client: Any
    client_authenticated: Any
    superhote_api_username: str = None
    superhote_api_password: str = None
    superhote_api_token: str = None
    aiosession: Optional[aiohttp.ClientSession] = None

    @staticmethod
    def add_apikey_header(request: Request):
        if "SUPERHOTE_API_USERNAME" not in os.environ:
            raise ValueError('SUPERHOTE_API_USERNAME is not defined')
        if "SUPERHOTE_API_PASSWORD" not in os.environ:
            raise ValueError('SUPERHOTE_API_PASSWORD is not defined')

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api creds and python package exists in environment."""
        superhote_api_username = get_from_dict_or_env(
            values, "superhote_api_username", "SUPERHOTE_API_USERNAME"
        )
        superhote_api_password = get_from_dict_or_env(
            values, "superhote_api_password", "SUPERHOTE_API_PASSWORD"
        )
        try:
            client = Client(base_url="https://app.superhote.com")
            values["client"] = client
            login_body = UserLoginBody.from_dict(
                {
                    "email": superhote_api_username,
                    "password": superhote_api_password,
                    "remember_me": 1
                }
            )
            response = post_api_v2_user_login.sync(client=client, json_body=login_body)
            token = response.token
            values["token"] = token
            client_authenticated = AuthenticatedClient(base_url="https://app.superhote.com", token=token,
                                                       raise_on_unexpected_status=True)
            values["client_authenticated"] = client_authenticated

        except ImportError:
            raise ValueError(
                "Could not import superhote_client python package. "
                "Please install it with `pip install superhote-python-client --index-url "
                "https://__token__:<your_personal_token>@gitlab.com/api/v4/projects/53067372/packages/pypi/simple`."
            )
        return values

    def run_get_all_statuses(self):
        return get_api_v2_get_all_statuses.sync(client=self.client_authenticated)

    def run_get_default_templates(self):
        """
            Useful to retrieve message template for sending invoice, payment, scheduled payment, deposit in the user
            language
        """
        return get_api_v2_default_templates.sync(client=self.client_authenticated)

    def run_get_calendars(self, rental_id: str = None, from_: date = None, to: date = None):
        return get_api_v2_get_calendars.sync(client=self.client_authenticated, rental_ids=rental_id, from_=from_, to=to)

    def run_get_current_bookings(self, rental_ids: List = None):
        if rental_ids is not None:
            rental_ids = ','.join(rental_ids)
        return get_api_v2_get_current_bookings.sync(client=self.client_authenticated, rental_ids=rental_ids)

    def run_get_task_statuses(self):
        return get_api_v2_get_statuses.sync(client=self.client_authenticated)

    def run_get_task(self, start: date, end: date, rental_ids: List = None):
        if rental_ids is not None:
            rental_ids = ','.join(rental_ids)
        return get_api_v2_tasks.sync(client=self.client_authenticated, start=start, end=end, rentals=rental_ids)

    def run_get_rentals(self, permission: bool = True):
        return get_api_v2_rentals.sync(client=self.client_authenticated, permission=permission)

    def run_get_custom_short_codes(self):
        """
            Useful to retrieve Checkout procedure or Instruction Access in the user language
        """
        return get_api_v2_custom_short_codes.sync(client=self.client_authenticated)

    def run_get_channels(self):
        return get_api_v2_channels.sync(client=self.client_authenticated)

    def run_get_messages_list_by_booking_id(self, booking_id: int):
        return get_api_v2_messages_list.sync(client=self.client_authenticated, booking_id=booking_id)

    def run_get_not_available_dates_by_rental_id(self, rental_id: int):
        body = V2GetnotavailabledatesBody.from_dict(
            {
                "rentalId": rental_id
            }
        )
        return post_api_v2_get_not_available_dates.sync(client=self.client_authenticated, json_body=body)

    def run_get_booking_templates(self, booking_id: str):
        return get_api_v2_booking_templates_booking_id.sync(client=self.client_authenticated, booking_id=booking_id)

    def run_get_payments(self, filter_by_status: GetApiV2PaymentsFilterByStatus = None, page: int = None,
                         **kwargs: Any):
        return get_api_v2_payments.sync(client=self.client_authenticated, filter_by_status=filter_by_status, page=page)

    def run_create_payments(self, amount: str, booking_id: int, customer_email: str, title: str, stripe_account_id: int,
                            rental_id: int):
        body = V2PaymentsBody.from_dict(
            {
                "amount": amount,
                "type": "other",
                "booking_id": booking_id,
                "customer_email": customer_email,
                "description": title,
                "captured": 1,
                "auto_renew": 1,
                "stripe_account_id": stripe_account_id,
                "title": title,
                "rental_id": rental_id
            }
        )
        return post_api_v2_payments.sync(client=self.client_authenticated, json_body=body)

    def run_get_airbnb_listing(self, rental_id: int):
        return get_api_v2_get_airbnb_listings.sync(client=self.client_authenticated, rental_id=rental_id)

    def run_get_availabilities(self, api_key: str, property_key: str, start_date: date, end_date: date, nbr_adults: int,
                               nbr_children: int, include_taxes: bool = False):
        body = V2GetavailabilitiesBody.from_dict(
            {
                "api_key": api_key,
                "property_key": property_key,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "nbr_adults": nbr_adults,
                "nbr_children": nbr_children,
                "include_taxes": include_taxes
            }
        )
        return post_api_v2_get_availabilities.sync(client=self.client_authenticated, json_body=body)

    def run_create_booking(
            self,
            price: str,
            deposit: str,
            cleaning: str,
            price_extra: str,
            rental_id: int,
            checking: date,
            checkout: date,
            channel_id: int,
            nbr_children: str,
            nbr_adults: str,
            city_taxes: str,
            service_charges: str,
            other_charges: str,
            payment_charges: str,
            commission: str,
            time_in: str,
            time_out: str,
            first_name: str,
            last_name: str,
            email: str,
            phone: str,
            zip_code: str,
            address: str,
            city: str,
            country: str
    ):
        body = V2BookingsBody.from_dict(
            {
                "price": price,
                "deposit": deposit,
                "cleaning": cleaning,
                "price_extra": price_extra,
                "rental": {},
                "rental_id": rental_id,
                "checking": checking.isoformat(),
                "checkout": checkout.isoformat(),
                "channel_id": channel_id,
                "nbr_children": nbr_children,
                "nbr_adults": nbr_adults,
                "follow_up": 1,
                "city_taxes": city_taxes,
                "service_charges": service_charges,
                "other_charges": other_charges,
                "payment_charges": payment_charges,
                "confirmation_code": "",
                "booking_color": "#D7092E",
                "commission": commission,
                "status": 1,
                "time_in": time_in,
                "time_out": time_out,
                "statusCode": 12,
                "deposit_is_done": 0,
                "payment_is_done": 0,
                "signature": None,
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "alternative_email": "",
                "phone": phone,
                "alternative_phone": "",
                "zip_code": zip_code,
                "address": address,
                "city": city,
                "country": country,
                "confirmed": False
            }
        )
        return post_api_v2_bookings.sync(client=self.client_authenticated, json_body=body)

    def run_post_message(self, message: str, booking_id: int, guest_id: int, token: str):
        body = V2MessagesBody.from_dict(
            {
                "message": {
                    "type": "mail",
                    "text": message,
                    "booking_id": booking_id,
                    "guest_id": guest_id,
                    "from": "user"
                },
                "token": token
            }
        )
        return post_api_v2_messages.sync(client=self.client_authenticated, json_body=body)

    def run_send_invoice(self, message: str, email: str, booking_id: int, language: str, subject: str):
        body = V2SendinvoiceBody.from_dict(
            {
                "message": message,
                "email": email,
                "bookingId": booking_id,
                "language": language,
                "fromUI": True,
                "subject": subject
            }
        )
        return post_api_v2_send_invoice.sync(client=self.client_authenticated, json_body=body)

    def run_send_payment(self, message: str, email: str, rental_id: int, payment_id: int, language: str, subject: str):
        body = PaymentsSendpaymentBody.from_dict(
            {
                "type": "mail",
                "language": language,
                "email": email,
                "subject": subject,
                "message": message,
                "rental_id": rental_id,
                "id": payment_id
            }
        )
        return post_api_v2_payments_send_payment.sync(client=self.client_authenticated, json_body=body)

    def run_get_booking(self, booking_id: str):
        return get_api_v2_bookings_booking_id.sync(client=self.client_authenticated, booking_id=booking_id)

    def run_get_booking_by_date(self, date_: date, rental_ids: str = None):
        if rental_ids is not None:
            rental_ids = ','.join(rental_ids)
        return get_api_v2_bookings_get_by_selected_day_date.sync(client=self.client_authenticated, date=date_, rental_ids=rental_ids)

    def run_get_rental(self, rental_id: str):
        return get_api_v2_rentals_rental_id.sync(client=self.client_authenticated, rental_id=rental_id)


