import pandas as pd
def import_data():
    fred_md = pd.read_csv("current.csv", header = 0, parse_dates = True).drop(index = 0)
    fred_md['sasdate'] = pd.to_datetime(fred_md['sasdate'])
    return fred_md
