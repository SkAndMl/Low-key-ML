def weekend_or_weekday(year,month,date):
    """
    This function returns 1 if its a weekday and returns a 0 if its a weekend.
    This is done to encode the feature at this stage itself.
    """
    import datetime
    d = datetime(year,month,date)
    if d.weekday()>4:
        return 0
    else:
        return 1