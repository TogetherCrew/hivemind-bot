import logging
from datetime import timedelta

from dateutil import parser


def process_dates(dates: list[str], d: int) -> list[str]:
    """
    process the dates to be from `date - d` to `date + d`

    Parameters
    ------------
    dates : list[str]
        the list of dates given
    d : int
        to update the `dates` list to have `-d` and `+d` days


    Returns
    ----------
    dates_modified : list[str]
        days added to it
    """
    dates_modified: list[str] = []
    if dates != []:
        lowest_date = min(parser.parse(date) for date in dates)
        greatest_date = max(parser.parse(date) for date in dates)

        delta_days = timedelta(days=d)

        # the date condition
        dt = lowest_date - delta_days
        while dt <= greatest_date + delta_days:
            dates_modified.append(dt.strftime("%Y-%m-%d"))
            dt += timedelta(days=1)
    else:
        logging.warning("No dates given!")

    return dates_modified
