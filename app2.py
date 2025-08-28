import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, AgGridTheme

st.set_page_config(page_title="Manager Dashboard", layout="wide", initial_sidebar_state="collapsed")


# --- Session State ---
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = True
if "username" not in st.session_state:
    st.session_state.username = ""
if "comments" not in st.session_state:
    st.session_state.comments = {}

# -----------------------------
# --- Users / Credentials ---
# -----------------------------
users = {
    "Leon": {"password": "LF1"},
}

# -----------------------------
# --- Login ---
# -----------------------------
if not st.session_state.logged_in:
    st.title("🔒 Manager Dashboard Login")
    username_input = st.text_input("Username")
    password_input = st.text_input("Password", type="password")
    login = st.button("Login")

    if login:
        if username_input in users and password_input == users[username_input]["password"]:
            st.session_state.logged_in = True
            st.session_state.username = username_input
            # st.experimental_rerun()
        else:
            st.error("Incorrect username or password")
    st.stop()  # 🚨 Nothing else renders (including sidebar!) until logged in



# -----------------------
# -----------------------------
# --- Load Data ---
# -----------------------------
employees = pd.read_csv("Employees.csv", parse_dates=["Hire Date"])
LAE_Metrics = pd.read_csv("NewData.csv", parse_dates=["Date"])
office_info = pd.read_csv("OfficeInfo.csv")
underwriting = pd.read_csv("UW.csv", parse_dates=["Date"])

# -----------------------------
# --- Helper Function ---
# -----------------------------
# --- Compute projections ---
today = pd.Timestamp.today() - pd.Timedelta(days=1)
current_month = today.to_period("M")

all_days = pd.date_range(start=current_month.start_time, end=current_month.end_time)
working_days = all_days[all_days.dayofweek != 6]  # exclude Sundays
total_working_days = len(working_days)
working_days_so_far = len(working_days[working_days <= today])
days_left = total_working_days - working_days_so_far


employees = employees[employees['Office'] != 'Irvine']
# --- Compute summaries for all offices ---
all_offices = LAE_Metrics['Office'].unique()

def compute_office_summary_simple(office):
    LAE_filtered = LAE_Metrics[LAE_Metrics['Office'] == office].copy()
    if LAE_filtered.empty:
        return None

    # Dates and periods
    LAE_filtered['Date'] = pd.to_datetime(LAE_filtered['Date'], errors='coerce')
    LAE_filtered = LAE_filtered.dropna(subset=['Date'])
    LAE_filtered['MonthD'] = LAE_filtered['Date'].dt.to_period('M')

    # Define current month and last month inside the function
    today = pd.Timestamp.today() - pd.Timedelta(days=1)
    current_month = today.to_period("M")
    last_month = (today.replace(day=1) - pd.DateOffset(days=1)).to_period("M")

    # Current month totals
    current_data = LAE_filtered[
    LAE_filtered['MonthD'] == current_month]    

    current_GI = current_data['GI'].sum()
    current_NB = current_data['NB'].sum()


    # Compute working days for projections
    all_days = pd.date_range(start=current_month.start_time, end=current_month.end_time)
    working_days = all_days[all_days.dayofweek != 6]  # exclude Sundays
    total_working_days = len(working_days)
    working_days_so_far = len(working_days[working_days <= today])


    # Projection
    if working_days_so_far > 0:
        GI_projection = (current_GI / working_days_so_far * total_working_days)
        NB_projection = (current_NB / working_days_so_far * total_working_days)
    else:
        GI_projection = 0
        NB_projection = 0

    # Last month totals
    last_month_data = LAE_filtered[LAE_filtered['MonthD'] == last_month]
    last_month_GI = last_month_data['GI'].sum()
    last_month_NB = last_month_data['NB'].sum()

    # Last 3 months avg (excluding current month)

    last3_months = pd.period_range(end=last_month, periods=3, freq="M")
    last3_data = LAE_filtered[LAE_filtered['MonthD'].isin(last3_months)]

    # --- Monthly Average ---
    last3_avg_GI = last3_data.groupby('MonthD')['GI'].sum().mean()
    last3_avg_NB = last3_data.groupby('MonthD')['NB'].sum().mean()

    # --- Weekly Average ---
    last3_data['Week'] = last3_data['Date'].dt.to_period('W')
    last3_avg_GI_weekly = last3_data.groupby('Week')['GI'].sum().mean()
    last3_avg_NB_weekly = last3_data.groupby('Week')['NB'].sum().mean()

    # --- Daily Average ---
    last3_avg_GI_daily = last3_data.groupby('Date')['GI'].sum().mean()
    last3_avg_NB_daily = last3_data.groupby('Date')['NB'].sum().mean()

    LMDIff_GI = GI_projection-last_month_GI
    LMDIff_NB = NB_projection-last_month_NB

    DIff_GI_100K = current_GI- 100_000
    #3 Month AVg Diff
    DIff_GI_100K_Avg = last3_avg_GI- 100_000
    DIff_GI_100K_AvgWeekly = DIff_GI_100K_Avg/4
    DIff_GI_100K_AvgDaily = DIff_GI_100K_Avg/(total_working_days)


    return {
        "Office": office,
        "GI": current_GI,
        "NB": current_NB,
        "GI Projection": GI_projection,
        "NB Projection": NB_projection,
        "GI LM": last_month_GI,
        "LM GI Diff": LMDIff_GI,
        "NB LM": last_month_NB,
        "LM NB Diff": LMDIff_NB, "Diff $100K": DIff_GI_100K,"Per Day - $100K": DIff_GI_100K,
        "GI Monthly Avg": last3_avg_GI,
        "NB Monthly Avg": last3_avg_NB, "GI Weekly Avg": last3_avg_GI_weekly,
        "NB Weekly Avg": last3_avg_NB_weekly, "GI Daily Avg": last3_avg_GI_daily,
        "NB Daily Avg": last3_avg_NB_daily, 
        "Diff $100K - Month": DIff_GI_100K_Avg,"Diff $100K - Week": DIff_GI_100K_AvgWeekly,"Diff $100K - Day": DIff_GI_100K_AvgDaily,
    }


# Build the summary dataframe
summaries = [compute_office_summary_simple(office) for office in all_offices]
summary_df = pd.DataFrame([s for s in summaries if s is not None])

# --- Add Goals from office_info ---
goals = office_info[['Regional','Office','Manager', 'GI Goal', 'NB Goal']].copy()
summary_df = summary_df.merge(goals, on='Office', how='left')


# Define positions to count
count_agents = ['Floor Assistant', 'Agent']
# Count agents per office
agents_count_df = (
    employees[employees['Position'].isin(count_agents)]
    .groupby('Office')['Employee']  # or 'Employee' if that’s the column name
    .nunique()
    .reset_index()
    .rename(columns={'Employee': 'Agents'}))

# If you want to merge this with your summary_df
summary_df = summary_df.merge(agents_count_df, on='Office', how='left')



summary_df['Per Agent - $100K'] = summary_df['Per Day - $100K'] / summary_df['Agents']
summary_df['Diff Goal'] = summary_df['GI']- summary_df['GI Goal']
summary_df['Per Day - Goal'] = summary_df['Diff Goal']/(days_left)
summary_df['Per Agent - Goal'] = summary_df['Per Day - Goal'] / summary_df['Agents']


#3 Month AVg Diff
summary_df['Diff Goal - Month'] = summary_df['GI Monthly Avg']- summary_df['GI Goal']
summary_df['Diff Goal - Week'] = summary_df['Diff Goal - Month']/4
summary_df['Diff Goal - Day'] = summary_df['Diff Goal - Month']/(total_working_days)



UW_Office = (underwriting.groupby("Office")[["UW BF", "UW NB"]].sum())
summary_df = summary_df.merge(UW_Office, on='Office', how='left')


desired_order = [
   'Regional',"Office",'Manager','Agents', "GI Goal", "NB Goal", "GI Projection", "NB Projection","GI","NB",'UW BF','UW NB',
    "GI LM","LM GI Diff", "NB LM","LM NB Diff",'Diff $100K',"Per Day - $100K","Per Agent - $100K",'Diff Goal',"Per Day - Goal","Per Agent - Goal", "GI Monthly Avg",
    "GI Weekly Avg", "GI Daily Avg","Diff $100K - Month","Diff $100K - Week","Diff $100K - Day",'Diff Goal - Month','Diff Goal - Week','Diff Goal - Day',
    "NB Monthly Avg","NB Weekly Avg", "NB Daily Avg"]#"NB Weekly Avg", "NB Daily Avg"
formatted_df = summary_df[desired_order]





def Offices_view():
    # # -----------------------------
    # # --- Display as AgGrid ---
    # # -----------------------------
    st.title("📊 Office Auto Sales - AI")
    st.markdown(f"""
    <div style="font-size:14px; color: #555;">
        {current_month.start_time.strftime("%m/%d/%y")} – {today.strftime("%m/%d/%y")} | &nbsp;
        <strong>Worked:</strong> {working_days_so_far} &nbsp;&nbsp;
        <strong>Left:</strong> {days_left} &nbsp;&nbsp;
        <strong>Total:</strong> {total_working_days}
    </div>
    """, unsafe_allow_html=True)
    st.markdown("***")
    # All available regions (excluding the ones you want to skip)
    exclude_regions = ["DMV", "Karina Cano", "Immigration", " "]
    available_regions = [r for r in formatted_df['Regional'].dropna().unique() if r not in exclude_regions]

    # # Multiselect filter
    # selected_regions = st.multiselect(
    #     "Select Region(s) to display",
    #     options=available_regions,
    #     default=available_regions  # show all by default
    # )



    # office_managers_df = employees[employees['Position'] == "Office Manager"][['Employee Name', 'Office']]
    # office_managers_df['Label'] = office_managers_df['Employee Name'] + ' - ' + office_managers_df['Office']

    with st.sidebar.expander("Select Regionals", expanded=False):
        selected_regions = st.multiselect("Select Regional", sorted(available_regions) , default=available_regions)
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            # st.experimental_rerun()








    # Get unique regions, drop NaNs
    unique_regions = formatted_df['Regional'].dropna().unique()

    for region in selected_regions:
        st.subheader(f"Regional: {region}")
        region_df = formatted_df[formatted_df['Regional'] == region].copy()
        if region in exclude_regions:
            continue

        # Filter for this region
        region_df = region_df.drop(columns=['Regional'])

        # Ensure numeric columns
        numeric_cols = ['GI Projection', 'GI Goal', 'NB Projection', 'NB Goal', 'Agents']
        for col in numeric_cols:
            region_df[col] = pd.to_numeric(region_df[col], errors='coerce')

        # Replace remaining NaN with blank string for display
        region_df = region_df.fillna('-')

        # --- Apply styling vs goal ---
        def style_negative(row, metric):
            """Color the cell red if value < 0, orange if between 0 and 5, else green"""
            val = row[metric]

            # Try to convert to float, otherwise skip coloring
            try:
                val = float(val)
            except (ValueError, TypeError):
                return ['' if col == metric else '' for col in row.index]

            color = ''
            if val < 0:
                color = 'red'
            else:
                color = 'green'

            return [f'color: {color}; font-weight: bold' if col == metric else '' for col in row.index]


        def style_vs_goal(row, metric, goal_col, threshold):
            val = row[metric]
            goal = row[goal_col]
            color = ''
            if pd.notna(val) and pd.notna(goal) and goal > 0:
                if val > goal:
                    color = 'green'
                elif val >= goal - threshold:
                    color = 'orange'
                else:
                    color = 'red'
            return [f'color: {color}; font-weight: bold' if col == metric else '' for col in row.index]

        styled_region_df = region_df.style.apply(
            style_vs_goal, axis=1, metric='GI Projection', goal_col='GI Goal', threshold=2500
        ).apply(
            style_vs_goal, axis=1, metric='NB Projection', goal_col='NB Goal', threshold=10
        )

        negative_cols = ['LM GI Diff', 'LM NB Diff','Diff $100K','Per Day - $100K','Per Agent - $100K','Diff Goal','Per Day - Goal',
                        'Per Agent - Goal','Diff $100K - Month','Diff $100K - Week','Diff $100K - Day','Diff Goal - Month',
                        'Diff Goal - Week','Diff Goal - Day']  # add more if needed

        for col in negative_cols:
            styled_region_df = styled_region_df.apply(style_negative, axis=1, metric=col)

        def format_currency(x):
            return f"${x:,.0f}" if isinstance(x, (int, float)) else x

        def format_number(x):
            return f"{x:,.0f}" if isinstance(x, (int, float)) else x

        # --- General styling ---
        currency_cols = [
        'GI Projection', 'GI', 'GI LM', 'GI Monthly Avg', 'GI Weekly Avg', 'GI Daily Avg',
        'GI Goal', 'LM GI Diff', 'Diff $100K', 'Diff Goal', 'Per Day - Goal', 
        'Per Agent - Goal', 'Per Day - $100K', 'Per Agent - $100K', 'UW BF','Diff $100K - Month','Diff $100K - Week',
        'Diff $100K - Day','Diff Goal - Month','Diff Goal - Week','Diff Goal - Day'
        ]

        number_cols = [
            'UW NB', 'LM NB Diff', 'NB Monthly Avg', 'NB Daily Avg', 'NB Weekly Avg',
            'NB Projection', 'NB Goal', 'Agents'
        ]
        styled_region_df = styled_region_df.format(
        {col: format_currency for col in currency_cols} |
        {col: format_number for col in number_cols}
        ).set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', "#3583D2"),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('font-size', '14px'),
                ('text-align', 'center'),
                ('padding', '8px'),
                ('border-bottom', '2px solid #1565C0'),
                ('width', '120px')
            ]},
            {'selector': 'td', 'props': [
                ('text-align', 'center'),
                ('padding', '6px'),
                ('font-size', '13px'),
                ('border-bottom', '1px solid #e0e0e0')
            ]},
            {'selector': 'tr', 'props': [('background-color', 'white')]}
        ]).hide(axis="index")


        def grey_background(row, columns):
            return [
                'background-color: #f0f0f0' if col in columns else '' 
                for col in row.index
            ]

        # Suppose you want to shade 'DIff_GI_100K_DayAgent' and 'LM GI Diff'
        styled_region_df = styled_region_df.apply(
            grey_background, axis=1, columns=['Per Day - $100K', 'Per Agent - $100K',"Diff $100K"]
        )
        def blue_background(row, columns):
                return [
                    'background-color: #DEEDFC' if col in columns else '' 
                    for col in row.index
                ]

        # Suppose you want to shade 'DIff_GI_100K_DayAgent' and 'LM GI Diff'
        styled_region_df = styled_region_df.apply(
            blue_background, axis=1, columns=['Office', 'Manager']
        )

            
        
        table_height = max(200, 30 + 30 * len(region_df))  # 30px per row + small header offset

        html_table = styled_region_df.to_html()

        # Inject custom header row *inside* thead, not replacing it
        extra_header = """
    <thead>
    <tr>
        <!-- empty placeholders for first 2 frozen columns -->
        <th rowspan="2" style="background-color:transparent; border:none;"></th>
        <th rowspan="2" style="background-color:transparent; border:none;"></th>
        <th rowspan="2" style="background-color:transparent; border:none;"></th>

        
        <!-- Grouped headers -->
        <th colspan="2" style="background-color:#00438A;
        color:white; font-weight:bold; border:1px solid #1565C0;">Goals</th>
        <th colspan="2" style="background-color:#00438A;
        color:white; font-weight:bold; border:1px solid #1565C0;">Projections</th>
        <th colspan="2" style="background-color:#00438A;
        color:white; font-weight:bold; border:1px solid #1565C0;">Totals</th>
            <th colspan="2" style="background-color:#00438A;
        color:white; font-weight:bold; border:1px solid #1565C0;"> Month Underwriting</th>
        <th colspan="4" style="background-color:#00438A;
        color:white; font-weight:bold; border:1px solid #1565C0;">Last Month</th>
        <th colspan="6" style="background-color:#00438A;
        color:white; font-weight:bold; border:1px solid #1565C0;">$ Need per Day Left to Reach Goals</th>
        <th colspan="3" style="background-color:#00438A;
        color:white; font-weight:bold; border:1px solid #1565C0;">3 Month GI Average</th>
        <th colspan="6" style="background-color:#00438A;
        color:white; font-weight:bold; border:1px solid #1565C0;">Reaching GI Goal based on Average |  $100K & Office</th>
    <th colspan="3" style="background-color:#00438A;
        color:white; font-weight:bold; border:1px solid #1565C0;">3 Month NB Average</th>
    
    </tr>
    </thead>
    """

        # Place right after <thead>
        html_table = html_table.replace("<thead>", f"<thead>{extra_header}")

        # Wrap with CSS for sticky first column + sticky headers
        # --- CSS for sticky first two columns ---
        # Estimate height dynamically: 30px per row + ~60px for headers
        table_height = 60 + 30 * len(region_df)

        st.components.v1.html(
        f"""
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            table {{
                border-collapse: collapse;
                width: 100%;
            }}
            th, td {{
                padding: 6px;
                text-align: center;
                white-space: nowrap;
                min-width: 80px; /* smaller for mobile */
            }}

            /* Freeze first column */
            th:nth-child(1), td:nth-child(1) {{
                position: sticky;
                left: 0;
                background-color: white;
                z-index: 4;
            }}
            # /* Freeze second column */
            # th:nth-child(2), td:nth-child(2) {{
            #     position: sticky;
            #     left: 80px;
            #     background-color: white;
            #     z-index: 4;
            # }}
            /* Freeze headers */
            thead th {{
                position: sticky;
                top: 0;
                z-index: 5;
                background-color: #1976D2;
                color: white;
            }}

            /* Responsive tweaks */
            @media (max-width: 768px) {{
                table {{
                    font-size: 11px;
                }}
                th, td {{
                    min-width: 60px;
                    padding: 4px; 
                }}
                thead tr:first-child {{font-size: 8px;
            padding: 1px 2px;
    
        }} /* hide big grouped headers on small screens */
            }}
        </style>
        <div class="responsive-table" style="overflow-x:auto; max-height:{table_height}px;">
            {html_table}
        </div>
        """,
        height=table_height + 50,
        scrolling=True
    )
    















def data_frame_demo():



    st.markdown(f"# {list(page_names_to_funcs.keys())[1]}")
    st.write(
        """
    This is where I would add agents information.
"""
    )


page_names_to_funcs = {
"Offices": Offices_view,
"Agents in Offices": data_frame_demo
}
    

page_names_to_funcs = {
    "Offices": Offices_view,
    "Agents in Offices": data_frame_demo
}

demo_name = st.sidebar.selectbox("View", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
