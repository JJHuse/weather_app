from datetime import datetime, timedelta
import json
import time
import logging
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load NOAA API token
logger.info("Loading NOAA API token from secrets.json")
with open("secrets.json", "r") as f:
    secrets = json.load(f)
TOKEN = secrets.get("NOAA_API_TOKEN")
headers = {"token": TOKEN}

# Load datatypes from file
logger.info("Loading data files")
datatypes_df = pd.read_csv("datatypes.csv")
DATATYPES = dict(zip(datatypes_df["code"], datatypes_df["description"]))

cities_df = pd.read_csv("cities.csv")
LOCATIONS = dict(zip(cities_df['name'], cities_df['id']))  # Maps city name to location ID

# NOAA API endpoint
URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"

# Seconds to wait between queries
THROTTLE_TIME = 1

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout: Modernized
app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row([
            html.H1("Could you really fry an egg?", className="text-center mt-4 mb-1"),
            html.H6("Weather fact-checker (NOAA data)", className="text-center mt-2 mb-4"),
            dcc.Store(id='stored-data')
        ]),
        dbc.Row([
            dbc.Col(
                children=[
                    html.H4("Past Searches", className="text-center mt-2"),
                    dcc.Tabs(id="sidebar-tabs", value=None, children=[], vertical=True),
                    # html.Div(id="sidebar-content", className="mt-3")  # Content for the selected tab
                ],
                style={"width": "auto", "background-color": "#f8f9fa", "padding": "1rem",
                       "border-right": "1px solid #ddd"},
                width=2
            ),
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(
                                id='city-dropdown',
                                options=[{'label': name, 'value': name} for name in LOCATIONS.keys()],
                                value='Fargo, ND US',
                                placeholder='Search for a city',
                                searchable=True,
                                className='mb-2'
                            ), width=6),
                            dbc.Col([
                                dcc.DatePickerRange(
                                    id='date-picker',
                                    min_date_allowed='1982-01-01',
                                    max_date_allowed='2025-03-04',
                                    start_date='2000-01-01',
                                    end_date='2000-12-01',
                                    style={'display': 'none'}
                                ),
                                html.P(id='min-date-display', className='text-muted')
                            ], width=3),
                            # dbc.Col(html.P(id='min-date-display', className='text-muted'), width=3)
                        ]),
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(
                                id='weather-category',
                                options=[{'label': DATATYPES[dt], 'value': dt} for dt in DATATYPES],
                                value='TMAX',
                                placeholder='Select weather category',
                            ), width=6),
                            dbc.Col(dcc.Input(
                                id='guess-input',
                                type='number',
                                placeholder='Your guess',
                                className='form-control'
                            ), width=3),
                            dbc.Col(html.Button('Submit', id='submit-btn', n_clicks=0, className='btn btn-primary'),
                                    width=3)
                        ], className='mt-3'),
                    ]),
                    className='mb-4 shadow-sm'
                ),
                dbc.Card(
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(
                                children=[
                                    dcc.Graph(id='distribution-graph', config={'displayModeBar': False}),
                                    html.Div(id='stats-output', className='text-center text-muted mt-2')
                                ],
                                width=10
                            ),
                            dbc.Col([
                                dbc.Card(
                                    dbc.CardBody([
                                        html.H6("Days Represented", className="card-title"),
                                        html.P(id='days-represented', className="card-text")
                                    ]),
                                    style={"width": "12rem"}
                                ),
                                dbc.Card(
                                    dbc.CardBody([
                                        html.H6("Minimum value", className="card-title"),
                                        html.P(id='stats-min', className="card-text")
                                    ]),
                                    style={"width": "12rem"}
                                ),
                                dbc.Card(
                                    dbc.CardBody([
                                        html.H6("Maximum value", className="card-title"),
                                        html.P(id='stats-max', className="card-text")
                                    ]),
                                    style={"width": "12rem"}
                                )],
                                width=2
                            )
                            ]),
                        ]
                    ),
                    className='shadow-sm'
                )
            ])
        ])
        ]
)


# Helper function to split date range into one-year blocks
def split_into_year_blocks(start_date, end_date):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    blocks = []
    
    while start <= end:
        block_end = min(start + timedelta(days=364), end)  # 364 ensures ≤ 1 year
        blocks.append((start.strftime('%Y-%m-%d'), block_end.strftime('%Y-%m-%d')))
        start = block_end + timedelta(days=1)
    
    return blocks


# Helper function to fetch data for a station within a date range
def fetch_station_data(station_id, datatype, start_date, end_date):
    all_data = []
    current_start = start_date
    break_while = False
    
    while current_start <= end_date:
        if break_while:
            break
        params = {
            "datasetid": "GHCND",
            "stationid": station_id,
            "datatypeid": datatype,
            "startdate": current_start,
            "enddate": end_date,
            "limit": 1000,
            "units": "standard"
        }
        
        for attempt in range(2):  # Try twice
            start_time = time.time()
            try:
                logger.info(f"Attempt {attempt + 1}: Fetching data for station {station_id} from {current_start} to\
                             {end_date}")
                response = requests.get(URL, headers=headers, params=params, timeout=30)
                time.sleep(THROTTLE_TIME)
                elapsed = time.time() - start_time
                logger.info(f"API response status: {response.status_code}, took {elapsed:.2f}s")
                
                if response.status_code != 200 or 'results' not in response.json():
                    logger.warning(f"Fetch failed: Status {response.status_code}, Response: {response.text}")
                    break_while = True
                    break
                
                data = pd.DataFrame(response.json()["results"])
                all_data.append(data)
                
                if len(data) < 1000:
                    logger.info(f"Retrieved {len(data)} records; full range covered")
                    break_while = True
                    break
                
                last_date = pd.to_datetime(data['date']).max().strftime('%Y-%m-%d')
                logger.info(f"Last date in batch: {last_date}")
                if last_date >= end_date:
                    break_while = True
                    break
                
                current_start = (pd.to_datetime(last_date) + timedelta(days=1)).strftime('%Y-%m-%d')
                break  # Success, exit retry loop
            
            except requests.Timeout:
                logger.error(f"Query timed out after {time.time() - start_time:.2f}s")
                if attempt == 1:
                    logger.error("Retry failed, abandoning this block")
                    return pd.DataFrame()  # Give up after second timeout
                time.sleep(THROTTLE_TIME)  # Wait before retry
            except Exception as e:
                logger.error(f"Fetch failed: {str(e)}")
                break
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def get_station_id(params, get_all=False):
    for attempt in range(2):
        start_time = time.time()
        try:
            # log query attempt
            if attempt == 0:
                logger.info(f"Initial query to get station ID with params: {params}")
            else:
                logger.info(f"Retry query to get station ID with params: {params}")

            # execute query
            response = requests.get(URL, headers=headers, params=params, timeout=30)
            time.sleep(THROTTLE_TIME)

            # check response
            if response.status_code != 200 or 'results' not in response.json():
                logger.warning(f"Initial fetch failed: Status {response.status_code}, Response: {response.text}")
                raise Exception("No data available in this date range")
            data = pd.DataFrame(response.json()["results"])

            # decide what to return
            if data.empty:
                logger.warning("No data returned for initial query")
                return None
            if get_all:
                return set(data['station'])
            station_id = data['station'].iloc[0]
            break

        except requests.Timeout:
            logger.error(f"Query timed out after {time.time() - start_time:.2f}s")
            time.sleep(THROTTLE_TIME)
        except Exception as e:
            if attempt == 1:
                raise e
            continue

    logger.info(f"Selected station ID: {station_id}")
    return station_id


def get_days_in_range(start, end):
    date_format = '%Y-%m-%d'
    return (datetime.strptime(end, date_format) - datetime.strptime(start, date_format)).days + 1


@app.callback(
        [Output('date-picker', 'style'),
         Output('date-picker', 'min_date_allowed'),
         Output('date-picker', 'max_date_allowed'),
         Output('min-date-display', 'children')],
        [Input('city-dropdown', 'value')]
)
def get_date_picker_range(city):
    if city is None:
        return {'display': 'none'}, '01-01-2000', '01-01-2000', 'select location'
    url = "https://www.ncei.noaa.gov/cdo-web/api/v2/locations/"
    response = requests.get(url + LOCATIONS[city], headers=headers)
    if response.status_code != 200:
        return {'display': 'none'}, '01-01-2000', '01-01-2000', 'error with location'
    data = response.json()
    start_date = data['mindate']
    end_date = data['maxdate']
    return {'display': 'block'}, start_date, end_date, f"Earliest date: {start_date}"


# @app.callback(
#     Output("sidebar", "is_open"),
#     [Input("toggle-sidebar-btn", "n_clicks")],
#     [State("sidebar", "is_open")]
# )
# def toggle_sidebar(n_clicks, is_open):
#     if n_clicks:
#         return not is_open
#     return is_open
# @app.callback(
#     Output("sidebar", "style"),
#     [Input("toggle-sidebar-btn", "n_clicks")],
#     [State("sidebar", "style")]
# )
# def toggle_sidebar(n_clicks, current_style):
#     if n_clicks:
#         if current_style and current_style.get("display") == "none":
#             return {"display": "block", "background-color": "#f8f9fa", "padding": "1rem",
#                     "border-right": "1px solid #ddd"}
#         else:
#             return {"display": "none"}
#     return current_style


@app.callback(
    Output("sidebar-tabs", "children"),
    [Input("stored-data", "data")]
)
def update_sidebar_tabs(stored_data):
    if not stored_data:
        return [dcc.Tab(label="No Data Available")]
    return [
        dcc.Tab(label=(
            key.split(',')[0] + ' ' + key.split('_')[1] + ' ' +
            str(pd.to_datetime(min(stored_data[key], key=pd.to_datetime)).year) + '-' +
            str(pd.to_datetime(max(stored_data[key], key=pd.to_datetime)).year)
            )) for key in reversed(stored_data.keys())
    ]


# @app.callback(
#     Output("sidebar-content", "children"),
#     [Input("sidebar-tabs", "value")],
#     [State("stored-data", "data")]
# )
# def update_sidebar_content(selected_tab, stored_data):
#     if not selected_tab or not stored_data:
#         return "Select a tab to view data."
#     data = stored_data[selected_tab]
#     return html.Pre(json.dumps(data, indent=2))  # Display the data in JSON format


# Callback to fetch NOAA data and update output
@app.callback(
    [Output('distribution-graph', 'figure'),
     Output('stats-min', 'children'),
     Output('stats-max', 'children'),
     Output('stored-data', 'data'),
     Output('days-represented', 'children')],
    [Input('submit-btn', 'n_clicks')],
    [State('city-dropdown', 'value'),
     State('date-picker', 'start_date'),
     State('date-picker', 'end_date'),
     State('weather-category', 'value'),
     State('guess-input', 'value'),
     State('stored-data', 'data')]
)
def update_output(n_clicks, city, start_date, end_date, category, guess, stored_data):
    empty_fig = go.Figure().update_layout(
        plot_bgcolor='white', 
        paper_bgcolor='white', 
        font={'color': '#333'},
        xaxis={'visible': False}, 
        yaxis={'visible': False}
    )
    if stored_data is None:
        stored_data = {}

    if n_clicks == 0 or guess is None or city is None:
        logger.info("Callback skipped: Missing input (n_clicks, guess, or city)")
        return empty_fig, "None yet", "None yet", stored_data, "0 days"

    location_id = LOCATIONS[city]
    logger.info(f"Fetching data for {city}, {category} from {start_date} to {end_date} using location {location_id}")

    # Step 1: Initial query to get a station ID (limit 1)
    params = {
        "datasetid": "GHCND",
        "locationid": location_id,
        "datatypeid": category,
        "startdate": start_date,
        "enddate": end_date,
        "limit": 1,
        "units": "standard"
    }
    station_id = None

    # Step 2: Break into one-year blocks and fetch data
    year_blocks = split_into_year_blocks(start_date, end_date)
    logger.info(f"Date range split into {len(year_blocks)} year blocks: {year_blocks}")
    all_data = []

    for block_start, block_end in year_blocks:
        if station_id is None:
            try:
                params.update({"startdate": block_start, "enddate": block_end})
                station_id = get_station_id(params)
            except Exception:
                logger.info("No data for block %s to %s", block_start, block_end)
                continue
        missing_days_tolerance = 75
        target_days = get_days_in_range(block_start, block_end) - missing_days_tolerance
        block_data = fetch_station_data(station_id, category, block_start, block_end)
        if block_data.empty or block_data.shape[0] < target_days:
            params.update({"startdate": block_start, "enddate": block_end})
            params['limit'] = 5
            station_ids = get_station_id(params, get_all=True)
            for s in station_ids:
                if s == station_id:
                    continue
                logger.info("Trying new stationid %s", s)
                block_data = fetch_station_data(s, category, block_start, block_end)
                if block_data.shape[0] >= target_days:
                    station_id = s
                    break
        all_data.append(block_data)

    if not all_data:
        logger.warning(f"No data retrieved for station {station_id} across all blocks")
        stat_error = f"No data available for {city} from station {station_id}"
        return empty_fig, stat_error, stat_error, stored_data, "0 days"

    full_data = pd.concat(all_data, ignore_index=True)
    values = full_data['value'].astype(float)
    logger.info(f"Total data fetched for station {station_id}: {len(values)} records")

    unique_days = len(full_data['date'].unique())
    days = get_days_in_range(start_date, end_date)
    days_text = f"{unique_days} days of {days}"

    below, above = split_data(full_data, guess)

    # Graph: Histogram with guess line
    fig = px.histogram(values, nbins=20, color_discrete_sequence=['#4682B4'])
    fig.add_vline(x=guess, line_dash="dash", line_color="red", annotation_text=f"Below guess: {below}",
                  annotation_position="top left")
    fig.add_vline(x=guess, line_dash="dash", line_color="red", annotation_text=f"Above guess: {above}",
                  annotation_position="top right")
    fig.update_layout(
        showlegend=False, 
        plot_bgcolor='white', 
        paper_bgcolor='white', 
        font={'color': '#333'}, 
        bargap=0.1,
        xaxis_title=category,
        yaxis_title="Count"
    )

    # Stats
    # min_row = df.iloc[df["value"].idxmin()]['date'].split('T')[0]
    min_row = full_data.iloc[values.idxmin()]
    max_row = full_data.iloc[values.idxmax()]
    stat_min = f"{min_row['value']:.1f}, {min_row['date'].split('T')[0]}"
    stat_max = f"{max_row['value']:.1f}, {max_row['date'].split('T')[0]}"
    logger.info(f"Stats calculated for station {station_id}: {stat_min} {stat_max}")

    key = f"{city}_{category}"
    value = full_data.set_index('date')['value'].to_dict()
    if key in stored_data:
        stored_data[key].update(value)
    else:
        stored_data.update({key: value})  # Convert DataFrame to JSON-serializable format
    return fig, stat_min, stat_max, stored_data, days_text


def split_data(df, fulcrum):
    values = df['value'].astype(float)
    below = sum(1 for v in values if v < fulcrum)
    above = sum(1 for v in values if v > fulcrum)
    return below, above


if __name__ == '__main__':
    logger.info("Starting Dash app")
    app.run(debug=True)
