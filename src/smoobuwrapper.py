import os
import sys
import re
from datetime import date, datetime
from typing import Any, Dict, Optional, Tuple, List

import aiohttp

from langchain.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain.utils import get_from_dict_or_env
from pydantic import validator
from pydantic.typing import Literal
from requests import Request


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


ALLOWED_CHANNELS = {
    1: "Airbnb iCal",
    2: "Mowitania",
    11: "Blocked channel",
    13: "Direct booking",
    14: "Booking.com",
    15: "Ferienwohnung24 Berlin",
    16: "9Flats",
    17: "Only Apartments",
    18: "Expedia",
    19: "Wimdu",
    20: "HouseTrip",
    21: "Oh Berlin",
    22: "Travanto Ical",
    23: "FeWo-direkt / HomeAway",
    24: "Tripadvisor",
    25: "Roomorama",
    26: "Red Apple Apartments",
    27: "Rentxpress",
    28: "VRBO / HomeAway",
    29: "Traum-Ferienwohnungen (ical)",
    30: "Vacation Apartments (ical)",
    31: "Ferienwohnungen.de",
    32: "Waytostay",
    33: "Bedandbreakfast.eu",
    34: "BedyCasa",
    35: "Ebab",
    36: "Alterkeys",
    37: "Bemate",
    38: "VacationRentals.com / HomeAway",
    39: "AlwaysOnVacation",
    40: "Misterbnb",
    41: "RentalHouses",
    42: "HolidayRentals.com",
    43: "France-Voyage.com",
    44: "MorningCroissant",
    45: "Flat4day / Go-fewo",
    46: "Abritel / HomeAway",
    47: "Individueller iCal-Kanal 1",
    48: "Individueller iCal-Kanal 2",
    49: "Individueller iCal-Kanal 3",
    50: "HomeAway",
    51: "e-domizil/atraveo",
    52: "E-domizil Ical",
    53: "Bungalow.net",
    54: "Hundeurlaub.de",
    55: "BAYregio",
    56: "Ferienhausmiete.de",
    57: "Homelidays",
    58: "Feratel Deskline",
    59: "Feries.com",
    60: "Casevacanza.it",
    61: "HBook Booking System",
    62: "Pinpoint Booking System",
    63: "Spain Holiday",
    64: "Holidaylettings",
    65: "Flipkey",
    66: "Intobis",
    67: "HRS Destination / ImWeb",
    68: "Agoda",
    69: "RoomCity",
    70: "Homepage",
    71: "Agriturismo.it",
    72: "Ownersdirect.co.uk",
    73: "MWFerienwohnungen",
    74: "Airbnb",
    75: "Prodess",
    76: "Bergfex",
    77: "Tourist-online.de",
    78: "Bellevue Ferienhaus",
    79: "Contao",
    80: "Traum-Ferienwohnungen",
    81: "Ferienhausmarkt.com",
    82: "Rentahouseboat.com",
    83: "Bed-and-breakfast.it",
    84: "Tomas",
    85: "Landreise.de",
    86: "Micazu.nl",
    87: "apartment.at",
    88: "Pura Vida Travel / ORAmap",
    89: "Canada Stays",
    90: "Blocked channel (HRS)",
    91: "Feriwa.com",
    92: "Bed-And-Breakfast.it",
    93: "Motopress Booking Plugin",
    94: "Individual iCal - 4",
    95: "Individual iCal - 5",
    96: "Individual iCal - 6",
    97: "Individual iCal - 7",
    98: "wpestate/wprentals Plugin",
    99: "Beds 24",
    100: "Bauernhofurlaub.de",
    101: "Landsichten",
    102: "urlaub-in-florida.net",
    103: "HRS Destination / ImWeb 2",
    104: "HRS Destination / ImWeb 3",
    105: "Bodenseeferien",
}

ISO_4217_CODES = [
    "AFN",
    "EUR",
    "ALL",
    "DZD",
    "USD",
    "AOA",
    "XCD",
    "ARS",
    "AMD",
    "AWG",
    "AUD",
    "AZN",
    "BSD",
    "BHD",
    "BDT",
    "BBD",
    "BYN",
    "BZD",
    "XOF",
    "BMD",
    "INR",
    "BTN",
    "BOB",
    "BOV",
    "BAM",
    "BWP",
    "NOK",
    "BRL",
    "BND",
    "BGN",
    "BIF",
    "CVE",
    "KHR",
    "XAF",
    "CAD",
    "KYD",
    "CLP",
    "CLF",
    "CNY",
    "COP",
    "COU",
    "KMF",
    "CDF",
    "NZD",
    "CRC",
    "HRK",
    "CUP",
    "CUC",
    "ANG",
    "CZK",
    "DKK",
    "DJF",
    "DOP",
    "EGP",
    "SVC",
    "ERN",
    "ETB",
    "FKP",
    "FJD",
    "XPF",
    "GMD",
    "GEL",
    "GHS",
    "GIP",
    "GTQ",
    "GBP",
    "GNF",
    "GYD",
    "HTG",
    "HNL",
    "HKD",
    "HUF",
    "ISK",
    "IDR",
    "XDR",
    "IRR",
    "IQD",
    "ILS",
    "JMD",
    "JPY",
    "JOD",
    "KZT",
    "KES",
    "KPW",
    "KRW",
    "KWD",
    "KGS",
    "LAK",
    "LBP",
    "LSL",
    "ZAR",
    "LRD",
    "LYD",
    "CHF",
    "MOP",
    "MKD",
    "MGA",
    "MWK",
    "MYR",
    "MVR",
    "MRU",
    "MUR",
    "XUA",
    "MXN",
    "MXV",
    "MDL",
    "MNT",
    "MAD",
    "MZN",
    "MMK",
    "NAD",
    "NPR",
    "NIO",
    "NGN",
    "OMR",
    "PKR"
]


class Address:
    street: str
    postalCode: str
    location: str


class PriceElement:
    type: Optional[str] = None
    name: str
    amount: float
    quantity: Optional[int] = None
    tax: Optional[float] = None
    currencyCode: str
    sortOrder: Optional[int] = None
    priceIncludedInId: Optional[int] = None

    @validator('currencyCode')
    def validate_currency_code(cls, v):
        if v not in ISO_4217_CODES:
            raise ValueError(f'Invalid currency code: {v}. Must be an ISO 4217 code.')
        return v


class SmoobuAvailabityBodyInput(BaseModel):
    arrivalDate: date = Field(description="Parameter that defines the desired Start date of the booking. Format: "
                                          "yyyy-mm-dd.")
    departureDate: date = Field(description="Parameter that defines the desired End date of the booking. Format: "
                                            "yyyy-mm-dd.")
    apartments: List[int] = Field(default=[],
                                  description="Check availability in the given apartments. If apartments is empty it "
                                              "check all apartments.")
    customerId: int = Field(description="Id of the customer/user with the given apartment.")
    guests: Optional[int] = Field(default=None, description="number of guests.")
    discountCode: Optional[str] = Field(default=None, description="Discount code from bookingTool settings.")

    @validator('arrivalDate', 'departureDate')
    def validate_dates(cls, v, field):
        if field.name == 'departureDate' and 'arrivalDate' in cls.__fields_set__:
            assert v >= cls.__fields__['arrivalDate'].default, 'arrivalDate must be after or equal to departureDate'
        return v


class CreateBookingBodyInput(BaseModel):
    arrivalDate: date = Field(description="Parameter that defines the Start date of the booking. Format: "
                                          "yyyy-mm-dd.")
    departureDate: date = Field(description="Parameter that defines the End date of the booking. Format: "
                                            "yyyy-mm-dd.")
    channelId: Optional[int] = Field(default=None, description="Id of the Channel. Allowed values are: "
                                                               "1 for Airbnb iCal, "
                                                               "2 for Mowitania, "
                                                               "11 for Blocked channel, "
                                                               "13 for Direct booking, "
                                                               "14 for Booking.com, "
                                                               "15 for Ferienwohnung24 Berlin, "
                                                               "16 for 9Flats, "
                                                               "17 for Only Apartments, "
                                                               "18 for Expedia, "
                                                               "19 for Wimdu, "
                                                               "20 for HouseTrip, "
                                                               "21 for Oh Berlin, "
                                                               "22 for Travanto Ical, "
                                                               "23 for FeWo-direkt / HomeAway, "
                                                               "24 for Tripadvisor, "
                                                               "25 for Roomorama, "
                                                               "26 for Red Apple Apartments, "
                                                               "27 for Rentxpress, "
                                                               "28 for VRBO / HomeAway, "
                                                               "29 for Traum-Ferienwohnungen (ical), "
                                                               "30 for Vacation Apartments (ical), "
                                                               "31 for Ferienwohnungen.de, "
                                                               "32 for Waytostay, "
                                                               "33 for Bedandbreakfast.eu, "
                                                               "34 for BedyCasa, "
                                                               "35 for Ebab, "
                                                               "36 for Alterkeys, "
                                                               "37 for Bemate, "
                                                               "38 for VacationRentals.com / HomeAway, "
                                                               "39 for AlwaysOnVacation, "
                                                               "40 for Misterbnb, "
                                                               "41 for RentalHouses, "
                                                               "42 for HolidayRentals.com, "
                                                               "43 for France-Voyage.com, "
                                                               "44 for MorningCroissant, "
                                                               "45 for Flat4day / Go-fewo, "
                                                               "46 for Abritel / HomeAway, "
                                                               "47 for Individueller iCal-Kanal 1, "
                                                               "48 for Individueller iCal-Kanal 2, "
                                                               "49 for Individueller iCal-Kanal 3, "
                                                               "50 for HomeAway, "
                                                               "51 for e-domizil/atraveo, "
                                                               "52 for E-domizil Ical, "
                                                               "53 for Bungalow.net, "
                                                               "54 for Hundeurlaub.de, "
                                                               "55 for BAYregio, "
                                                               "56 for Ferienhausmiete.de, "
                                                               "57 for Homelidays, "
                                                               "58 for Feratel Deskline, "
                                                               "59 for Feries.com, "
                                                               "60 for Casevacanza.it, "
                                                               "61 for HBook Booking System, "
                                                               "62 for Pinpoint Booking System, "
                                                               "63 for Spain Holiday, "
                                                               "64 for Holidaylettings, "
                                                               "65 for Flipkey, "
                                                               "66 for Intobis, "
                                                               "67 for HRS Destination / ImWeb, "
                                                               "68 for Agoda, "
                                                               "69 for RoomCity, "
                                                               "70 for Homepage, "
                                                               "71 for Agriturismo.it, "
                                                               "72 for Ownersdirect.co.uk, "
                                                               "73 for MWFerienwohnungen, "
                                                               "74 for Airbnb, "
                                                               "75 for Prodess, "
                                                               "76 for Bergfex, "
                                                               "77 for Tourist-online.de, "
                                                               "78 for Bellevue Ferienhaus, "
                                                               "79 for Contao, "
                                                               "80 for Traum-Ferienwohnungen, "
                                                               "81 for Ferienhausmarkt.com, "
                                                               "82 for Rentahouseboat.com, "
                                                               "83 for Bed-and-breakfast.it, "
                                                               "84 for Tomas, "
                                                               "85 for Landreise.de, "
                                                               "86 for Micazu.nl, "
                                                               "87 for apartment.at, "
                                                               "88 for Pura Vida Travel / ORAmap, "
                                                               "89 for Canada Stays, "
                                                               "90 for Blocked channel (HRS), "
                                                               "91 for Feriwa.com, "
                                                               "92 for Bed-And-Breakfast.it, "
                                                               "93 for Motopress Booking Plugin, "
                                                               "94 for Individual iCal - 4, "
                                                               "95 for Individual iCal - 5, "
                                                               "96 for Individual iCal - 6, "
                                                               "97 for Individual iCal - 7, "
                                                               "98 for wpestate/wprentals Plugin, "
                                                               "99 for Beds 24, "
                                                               "100 for Bauernhofurlaub.de, "
                                                               "101 for Landsichten, "
                                                               "102 for urlaub-in-florida.net, "
                                                               "103 for HRS Destination / ImWeb 2, "
                                                               "104 for HRS Destination / ImWeb 3, "
                                                               "105 for Bodenseeferien.")
    apartmentId: int = Field(description="Id of the apartment.")
    arrivalTime: Optional[str] = Field(default=None, description="Time when the guest will arrive. In the Format "
                                                                 "'HH:ii'.")
    departureTime: Optional[str] = Field(default=None, description="Time when the guest will departs. In the Format "
                                                                   "'HH:ii'.")
    firstName: Optional[str] = Field(default=None, description="First name of the guest.")
    lastName: Optional[str] = Field(default=None, description="Last name of the guest.")
    notice: Optional[str] = Field(default=None, description="Notes. Free text field for you information.")
    adults: Optional[int] = Field(default=None, description="Number of adults.")
    children: Optional[int] = Field(default=None, description="Number of children.")
    price: Optional[float] = Field(default=None, description="Total price of the booking.")
    priceStatus: Optional[Literal[0, 1]] = Field(default=None, description="Payment status, Allowed values are: "
                                                                           "0 for open/not payed, "
                                                                           "1 for complete payment.")
    prepayment: Optional[float] = Field(default=None, description="Prepayment amount of the booking.")
    prepaymentStatus: Optional[Literal[0, 1]] = Field(default=None,
                                                      description="Prepayment status, Allowed values are: "
                                                                  "0 for open/not payed, "
                                                                  "1 for complete payment.")
    deposit: Optional[float] = Field(default=None, description="Deposit of the booking.")
    depositStatus: Optional[Literal[0, 1]] = Field(default=None, description="Deposit status, Allowed values are: "
                                                                             "0 for open/not payed, "
                                                                             "1 for complete payment.")
    address: Optional[Address] = Field(default=None, description="Address of the guest.")
    country: Optional[str] = Field(default=None, description="Country of the address.")
    email: Optional[str] = Field(default=None, description="Email of the guest.")
    phone: Optional[str] = Field(default=None, description="Phone Number of the guest.")
    language: Optional[str] = Field(default=None, description="Guest language i.e en, de, es.")
    priceElements: Optional[List[PriceElement]] = Field(default=None,
                                                        description="Array of price elements for the booking.")

    @validator('arrivalDate', 'departureDate')
    def validate_dates(cls, v, field):
        if field.name == 'departureDate' and 'arrivalDate' in cls.__fields_set__:
            assert v >= cls.__fields__['arrivalDate'].default, 'arrivalDate must be after or equal to departureDate'
        return v

    @validator('channelId')
    def validate_channel_id(cls, v):
        if v is not None and v not in ALLOWED_CHANNELS:
            raise ValueError(f'Invalid channelId: {v}')
        return v


class UpdateBookingBodyInput(BaseModel):
    arrivalTime: Optional[str] = Field(default=None, description="Time when the guest will arrive. In the Format "
                                                                 "'HH:ii'.")
    departureTime: Optional[str] = Field(default=None, description="Time when the guest will departs. In the Format "
                                                                   "'HH:ii'.")
    guestName: Optional[str] = Field(default=None, description="Full name of the guest.")
    notice: Optional[str] = Field(default=None, description="Notes. Free text field for you information.")
    assistantNotice: Optional[str] = Field(default=None, description="Assistant instructions.")
    adults: Optional[int] = Field(default=None, description="Number of adults.")
    children: Optional[int] = Field(default=None, description="Number of children.")
    price: Optional[float] = Field(default=None, description="Total price of the booking.")
    priceStatus: Optional[Literal[0, 1]] = Field(default=None, description="Payment status, Allowed values are: "
                                                                           "0 for open/not payed, "
                                                                           "1 for complete payment.")
    prepayment: Optional[float] = Field(default=None, description="Prepayment amount of the booking.")
    prepaymentStatus: Optional[Literal[0, 1]] = Field(default=None,
                                                      description="Prepayment status, Allowed values are: "
                                                                  "0 for open/not payed, "
                                                                  "1 for complete payment.")
    deposit: Optional[float] = Field(default=None, description="Deposit of the booking.")
    depositStatus: Optional[Literal[0, 1]] = Field(default=None, description="Deposit status, Allowed values are: "
                                                                             "0 for open/not payed, "
                                                                             "1 for complete payment.")
    address: Optional[Address] = Field(default=None, description="Address of the guest.")
    country: Optional[str] = Field(default=None, description="Country of the address.")
    guestEmail: Optional[str] = Field(default=None, description="Email of the guest.")
    guestPhone: Optional[str] = Field(default=None, description="Phone Number of the guest.")
    language: Optional[str] = Field(default=None, description="Guest language i.e en, de, es.")


class GetBookingsQueryParams(BaseModel):
    created_from: Optional[date] = Field(default=None,
                                         description="Parameter that defines the start date to show all bookings with "
                                                     "the created at in the range. Format: yyyy-mm-dd.")
    created_to: Optional[date] = Field(default=None, description="Parameter that defines the end date to show all "
                                                                 "bookings with"
                                                                 "the created at in the range. Format: yyyy-mm-dd.")
    fromDate: Optional[date] = Field(default=None,
                                     description="Parameter that defines the start date to show all bookings in the "
                                                 "range. Format: yyyy-mm-dd.", alias='from')
    to: Optional[date] = Field(default=None,
                               description="Parameter that defines the end date to show all bookings in the range. "
                                           "Format: yyyy-mm-dd.")
    modifiedFrom: Optional[date] = Field(default=None,
                                         description="Parameter that defines the start date to show all bookings with "
                                                     "the modified at in the range. Format: yyyy-mm-dd.")
    modifiedTo: Optional[date] = Field(default=None,
                                       description="Parameter that defines the end date to show all bookings with the "
                                                   "modified at in the range. Format: yyyy-mm-dd.")
    arrivalFrom: Optional[date] = Field(default=None,
                                        description="Parameter that defines the start date to show all bookings with "
                                                    "the arrival date in the range. Format: yyyy-mm-dd.")
    arrivalTo: Optional[date] = Field(default=None,
                                      description="Parameter that defines the end date to show all bookings with the "
                                                  "arrival date in the range. Format: yyyy-mm-dd.")
    departureFrom: Optional[date] = Field(default=None,
                                          description="Parameter that defines the start date to show all bookings with "
                                                      "the departure date in the range. Format: yyyy-mm-dd.")
    departureTo: Optional[date] = Field(default=None,
                                        description="Parameter that defines the end date to show all bookings with the "
                                                    "departure date in the range. Format: yyyy-mm-dd.")
    showCancellation: Optional[bool] = Field(default=None, description="Include cancelled bookings.")
    excludeBlocked: Optional[bool] = Field(default=None, description="Hide blocked bookings.")
    page: Optional[int] = Field(default=None, description="Current page")
    pageSize: Optional[int] = Field(default=None, description="Bookings per page (max 100).")
    apartmentId: Optional[int] = Field(default=None, description="Id of the apartment.")
    includeRelated: Optional[bool] = Field(default=None,
                                           description="When used together with apartmentId, returns bookings from "
                                                       "grouped apartments.")
    includePriceElements: Optional[bool] = Field(default=None,
                                                 description="Set true if want to show price elements in booking "
                                                             "response.")

    @validator('created_from', 'created_to, fromDate', 'to', 'modifiedFrom', 'modifiedTo', 'arrivalFrom', 'arrivalTo',
               'departureFrom', 'departureTo')
    def validate_dates(cls, v, field):
        if field.name == 'created_to' and 'created_from' in cls.__fields_set__:
            assert v >= cls.__fields__['created_from'].default, 'created_to must be after or equal to created_from'
        elif field.name == 'to' and 'fromDate' in cls.__fields_set__:
            assert v >= cls.__fields__['fromDate'].default, 'to must be after or equal to fromDate'
        elif field.name == 'modifiedTo' and 'modifiedFrom' in cls.__fields_set__:
            assert v >= cls.__fields__['modifiedFrom'].default, 'modifiedTo must be after or equal to modifiedFrom'
        elif field.name == 'arrivalTo' and 'arrivalFrom' in cls.__fields_set__:
            assert v >= cls.__fields__['arrivalFrom'].default, 'arrivalTo must be after or equal to arrivalFrom'
        elif field.name == 'departureTo' and 'departureFrom' in cls.__fields_set__:
            assert v >= cls.__fields__['departureFrom'].default, 'departureTo must be after or equal to departureFrom'
        return v

    @validator('pageSize')
    def validate_page_size(cls, v):
        if v is not None and (v < 1 or v > 100):
            raise ValueError('pageSize must be between 1 and 100')
        return v


class CreatePriceElementBodyInput(BaseModel):
    type: Optional[Literal["basePrice", "cleaningFee", "addon", "longStayDiscount", "coupon"]] = Field(default=None,
                                                                                                       description="Type of price element to create.")
    name: str = Field(description="Name of price element.")
    amount: float = Field(description="Amount.")
    quantity: Optional[int] = Field(default=None, description="Quantity.")
    tax: Optional[float] = Field(default=None, description="Tax if applicable")
    currencyCode: str = Field(description="ISO 4217 currency code.")
    sortOrder: Optional[int] = Field(default=None,
                                     description="sorting order to display price elements in reservation detail page. "
                                                 "The direction of sorting is ascending.")
    priceIncludedInId: Optional[int] = Field(default=None,
                                             description="Id of price element which cover the price of this element.")


class UpdatePriceElementBodyInput(BaseModel):
    type: Optional[Literal["basePrice", "cleaningFee", "addon", "longStayDiscount", "coupon"]] = Field(
        description="Type of price element to create.")
    name: str = Field(description="Name of price element.")
    amount: float = Field(description="Amount.")
    quantity: Optional[int] = Field(default=None, description="Quantity.")
    tax: Optional[float] = Field(default=None, description="Tax if applicable")
    currencyCode: str = Field(description="ISO 4217 currency code.")
    sortOrder: Optional[int] = Field(default=None, description="sorting order to display price elements in "
                                                               "reservation detail page."
                                                               "The direction of sorting is ascending.")
    priceIncludedInId: Optional[int] = Field(default=None,
                                             description="Id of price element which cover the price of this element.")


class GetRatesQueryParams(BaseModel):
    start_date: date = Field(..., description="Start date to show all rates in the range. Format: yyyy-mm-dd.")
    end_date: date = Field(..., description="End date to show all rates in the range. Format: yyyy-mm-dd.")
    apartments: List[int] = Field(..., description="array of apartment ids")

    @validator('start_date', 'end_date')
    def validate_dates(cls, v, field):
        if field.name == 'end_date' and 'start_date' in cls.__fields_set__:
            assert v >= cls.__fields__['start_date'].default, 'end_date must be after or equal to start_date'
        return v


class OperationDates(BaseModel):
    # Define a custom validator to ensure the format is either a single date or a date range
    date: str

    @validator('date')
    def validate_date_format(cls, v):
        if not any([cls.is_valid_single_date(v), cls.is_valid_date_range(v)]):
            raise ValueError('Invalid date format')
        return v

    @staticmethod
    def is_valid_single_date(date_str):
        # Regular expression for date format 'yyyy-mm-dd'
        date_pattern = r'^\d{4}-\d{2}-\d{2}$'
        if re.match(date_pattern, date_str):
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
                return True
            except ValueError:
                return False
        return False

    @staticmethod
    def is_valid_date_range(date_range_str):
        # Regular expression for date range format 'yyyy-mm-dd:yyyy-mm-dd'
        date_range_pattern = r'^\d{4}-\d{2}-\d{2}:\d{4}-\d{2}-\d{2}$'
        if re.match(date_range_pattern, date_range_str):
            start_date_str, end_date_str = date_range_str.split(':')
            try:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                return start_date < end_date
            except ValueError:
                return False
        return False


class Operation(BaseModel):
    dates: List[OperationDates]
    daily_price: Optional[float]
    min_length_of_stay: Optional[int]

    @validator('daily_price', 'min_length_of_stay', pre=True, always=True)
    def check_at_least_one(cls, v, values, **kwargs):
        if 'daily_price' not in values and 'min_length_of_stay' not in values:
            raise ValueError('At least one of daily_price or min_length_of_stay must be set')
        return v


class PostRatesBodyInput(BaseModel):
    apartments: List[int]
    operations: List[Operation]

    @validator('apartments', 'operations')
    def check_mandatory_fields(cls, v):
        if not v:
            raise ValueError('This field is mandatory')
        return v


class GetMessageQueryParams(BaseModel):
    page: Optional[int] = Field(default=None, description="current page")
    onlyRelatedToGuest: Optional[bool] = Field(default=None, description="get messages only related to guest")

    class Config:
        schema_extra = {
            "example": {
                "page": 1,
                "onlyRelatedToGuest": True
            }
        }


class SendMessageToGuestBodyInput(BaseModel):
    subject: Optional[str] = Field(default=None, description="Message subject")
    messageBody: str = Field(..., description="Message content in HTML or plain-text")


class SendMessageToHostBodyInput(BaseModel):
    subject: Optional[str] = Field(default=None, description="Message subject")
    messageBody: str = Field(..., description="Message content in HTML or plain-text")
    internal: Optional[bool] = Field(default=None, description="If true, message will only be visible to host. ("
                                                               "hidden from guest)")


class CreateCustomPlaceholderBodyInput(BaseModel):
    key: str = Field(..., description="Placeholder key.")
    defaultValue: Optional[str] = Field(default=None, description="Default value if no translation found.")
    type: Optional[Literal[1, 2]] = Field(default=None, description="Type of the placeholder (1 for Booking, 2 for "
                                                                    "Apartment).")
    foreignId: Optional[int] = Field(default=None, description="Depends on type. i.e., if type is booking, then it is "
                                                               "the booking id.")


class UpdateCustomPlaceholderBodyInput(BaseModel):
    key: str = Field(..., description="Placeholder key.")
    defaultValue: Optional[str] = Field(default=None, description="Default value if no translation found.")
    type: Optional[Literal[1, 2]] = Field(default=None, description="Type of the placeholder (1 for Booking, 2 for "
                                                                    "Apartment).")
    foreignId: Optional[int] = Field(default=None, description="Depends on type. i.e., if type is booking, then it is "
                                                               "the booking id.")


SupportedLocales = Literal[
    "ar", "az", "bg", "ca", "cs", "da", "de", "el", "en", "es", "et", "fr", "fi", "he", "hi",
    "hr", "hu", "id", "is", "it", "ja", "km", "ko", "lo", "lt", "lv", "ms", "nl", "no", "pl",
    "pt", "ro", "ru", "sk", "sl", "sr", "sv", "tl", "th", "tr", "uk", "vi", "zh"
]


class CreateCustomPlaceholderTranslationBodyInput(BaseModel):
    locale: SupportedLocales = Field(..., description="Locale of the language.")
    value: str = Field(..., description="Translation in the given locale.")


class UpdateCustomPlaceholderTranslationBodyInput(BaseModel):
    locale: SupportedLocales = Field(..., description="Locale of the language.")
    value: str = Field(..., description="Translation in the given locale.")


class GetGuestsQueryParams(BaseModel):
    page: Optional[int] = Field(default=None, description="Current page")
    pageSize: Optional[int] = Field(default=None, description="Bookings per page (maximum 100)")
    query: Optional[int] = Field(default=None, description="Search term")

    @validator('pageSize')
    def validate_page_size(cls, v):
        if v is not None and (v < 1 or v > 100):
            raise ValueError('pageSize must be between 1 and 100')
        return v


class SmoobuAPIWrapper(BaseModel):
    client: Any
    smoobu_api_key: str = None
    aiosession: Optional[aiohttp.ClientSession] = None

    @staticmethod
    def add_apikey_header(request: Request):
        if "SMOOBU_API_KEY" not in os.environ:
            raise ValueError('SMOOBU_API_KEY is not defined')
        request.headers['API-key:'] = os.getenv("SMOOBU_API_KEY")

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        smoobu_api_key = get_from_dict_or_env(
            values, "smoobu_api_key", "SMOOBU_API_KEY"
        )
        values["smoobu_api_key"] = smoobu_api_key
        try:
            from smoobu_python_client import ClientWithResponses
            client = ClientWithResponses(server="https://login.smoobu.com/api/",
                                         request_editors=[cls.add_apikey_header])

            values["client"] = client
        except ImportError:
            raise ValueError(
                "Could not import smoobu_python_client python package. "
                "Please install it with `pip install smoobu-python-client --index-url "
                "https://__token__:<your_personal_token>@gitlab.com/api/v4/projects/52825812/packages/pypi/simple`."
            )
        return values

    def run_get_user(self, **kwargs: Any):
        response = self.client.get_user_with_response(**kwargs)
        return response.data if response.error is None else response.error

    def run_get_smoobu_availability(self, body: SmoobuAvailabityBodyInput, **kwargs: Any):
        response = self.client.get_smoobu_availability_with_response(body=body.dict(), **kwargs)
        return response.data if response.error is None else response.error

    def run_create_booking(self, body: CreateBookingBodyInput, **kwargs: Any):
        response = self.client.create_booking_with_response(body=body.dict(), **kwargs)
        return response.data if response.error is None else response.error

    def run_update_booking(self, params: Dict, body: UpdateBookingBodyInput, **kwargs: Any):
        response = self.client.update_booking_with_response(params=params, body=body.dict(), **kwargs)
        return response.data if response.error is None else response.error

    def run_get_bookings(self, params: GetBookingsQueryParams, **kwargs: Any):
        response = self.client.get_bookings_with_response(params=params.dict(), **kwargs)
        return response.data if response.error is None else response.error

    def run_cancel_reservation(self, params: Dict, **kwargs: Any):
        response = self.client.cancel_reservation(params=params, **kwargs)
        return response.data if response.error is None else response.error

    def run_get_booking(self, params: Dict, **kwargs: Any):
        response = self.client.get_booking_with_response(params=params, **kwargs)
        return response.data if response.error is None else response.error

    def run_get_price_elements(self, params: Dict, **kwargs: Any):
        response = self.client.get_price_elements_with_response(params=params, **kwargs)
        return response.data if response.error is None else response.error

    def run_get_price_element(self, params: Dict, **kwargs: Any):
        response = self.client.get_price_element_with_response(params=params, **kwargs)
        return response.data if response.error is None else response.error

    def run_create_price_element(self, params: Dict, body: CreatePriceElementBodyInput, **kwargs: Any):
        response = self.client.create_price_element_with_response(params=params, body=body.dict(), **kwargs)
        return response.data if response.error is None else response.error

    def run_update_price_element(self, params: Dict, body: UpdatePriceElementBodyInput, **kwargs: Any):
        response = self.client.update_price_element_with_response(params=params, body=body.dict(), **kwargs)
        return response.data if response.error is None else response.error

    def run_delete_price_element(self, params: Dict, **kwargs: Any):
        response = self.client.delete_price_element_with_response(params=params, **kwargs)
        return response.data if response.error is None else response.error

    def run_get_placeholders(self, params: Dict, **kwargs: Any):
        response = self.client.get_placeholders_with_response(params=params, **kwargs)
        return response.data if response.error is None else response.error

    def run_get_rates(self, params: GetRatesQueryParams, **kwargs: Any):
        response = self.client.get_rates_with_response(params=params.dict(), **kwargs)
        return response.data if response.error is None else response.error

    def run_create_rates(self, body: PostRatesBodyInput, **kwargs: Any):
        response = self.client.create_rates_with_response(body=body.dict(), **kwargs)
        return response.data if response.error is None else response.error

    def run_get_apartment_ids(self, **kwargs: Any):
        response = self.client.get_apartment_ids_with_response(**kwargs)
        return response.data if response.error is None else response.error

    def run_get_apartment(self, params: Dict, **kwargs: Any):
        response = self.client.get_apartment_ids_with_response(params=params, **kwargs)
        return response.data if response.error is None else response.error

    def run_get_message(self, params: GetMessageQueryParams, **kwargs: Any):
        response = self.client.get_message_with_response(params=params.dict(), **kwargs)
        return response.data if response.error is None else response.error

    def run_get_addons(self, params: Dict, **kwargs: Any):
        response = self.client.get_addons_with_response(params=params, **kwargs)
        return response.data if response.error is None else response.error

    def run_send_message_to_guest(self, params: Dict, body: SendMessageToGuestBodyInput, **kwargs: Any):
        response = self.client.send_message_to_guest_with_response(params=params, body=body.dict(), **kwargs)
        return response.data if response.error is None else response.error

    def run_send_message_to_host(self, params: Dict, body: SendMessageToHostBodyInput, **kwargs: Any):
        response = self.client.send_message_to_host_with_response(params=params, body=body.dict(), **kwargs)
        return response.data if response.error is None else response.error

    def run_get_custom_placeholders(self, **kwargs: Any):
        response = self.client.get_custom_placeholders_with_response(**kwargs)
        return response.data if response.error is None else response.error

    def run_get_custom_placeholder(self, params: Dict, **kwargs: Any):
        response = self.client.get_custom_placeholder_with_response(params=params, **kwargs)
        return response.data if response.error is None else response.error

    def run_create_custom_placeholder(self, body: CreateCustomPlaceholderBodyInput, **kwargs: Any):
        response = self.client.create_custom_placeholder_with_response(body=body.dict(), **kwargs)
        return response.data if response.error is None else response.error

    def run_update_custom_placeholder(self, params: Dict, body: UpdateCustomPlaceholderBodyInput, **kwargs: Any):
        response = self.client.update_custom_placeholder_with_response(params=params, body=body.dict(), **kwargs)
        return response.data if response.error is None else response.error

    def run_delete_custom_placeholder(self, params: Dict, **kwargs: Any):
        response = self.client.delete_custom_placeholder_with_response(params=params, **kwargs)
        return response.data if response.error is None else response.error

    def run_get_custom_placeholder_translations(self, params: Dict, **kwargs: Any):
        response = self.client.get_custom_placeholder_translations_with_response(params=params, **kwargs)
        return response.data if response.error is None else response.error

    def run_get_custom_placeholder_translation(self, params: Dict, **kwargs: Any):
        response = self.client.get_custom_placeholder_translation_with_response(params=params, **kwargs)
        return response.data if response.error is None else response.error

    def run_create_custom_placeholder_translation(self, params: Dict,
                                                  body: CreateCustomPlaceholderTranslationBodyInput, **kwargs: Any):
        response = self.client.create_custom_placeholder_translation_with_response(params=params, body=body.dict(),
                                                                                   **kwargs)
        return response.data if response.error is None else response.error

    def run_update_custom_placeholder_translation(self, params: Dict,
                                                  body: UpdateCustomPlaceholderTranslationBodyInput, **kwargs: Any):
        response = self.client.update_custom_placeholder_translation_with_response(params=params, body=body.dict(),
                                                                                   **kwargs)
        return response.data if response.error is None else response.error

    def run_delete_custom_placeholder_translation(self, params: Dict, **kwargs: Any):
        response = self.client.delete_custom_placeholder_translation_with_response(params=params, **kwargs)
        return response.data if response.error is None else response.error

    def run_get_guests(self, params: GetGuestsQueryParams, **kwargs: Any):
        response = self.client.get_guests_with_response(params=params.dict(), **kwargs)
        return response.data if response.error is None else response.error

    def run_get_guest(self, params: Dict, **kwargs: Any):
        response = self.client.get_guest_with_response(params=params, **kwargs)
        return response.data if response.error is None else response.error
