import streamlit as s
import pandas as pd
from datetime import datetime, timedelta
import numpy as np 
from io import BytesIO

# --- Data Loading Functions ---

def find_column(df, keywords):
    """Fuzzy finds a column in the DataFrame based on keywords."""
    for col in df.columns:
        if any(kw in col.lower().replace('.', '').replace('_', ' ') for kw in keywords):
            return col
    return None

@st.cache_data(show_spinner=True)
def load_main_data(uploaded_files):
    """Loads and preprocesses the main leads data."""
    # ... (No major change needed here, keeping it clean)
    all_data = []
    date_columns = [
        'Date', 'Time', 'Field Visit Date', 'Next Call', 'PTP Date', 
        'Claim Paid Date', 'I.C Issue Date', 'Due Date', 'Last Pay Date'
    ]
    numeric_columns = [
        'S.No', 'DPD', 'PTP Amount', 'Claim Paid Amount', 'Days Past Write Off', 
        'Balance', 'Over Limit Amount', 'Min Payment', 'Monthly Installment', 
        '30 Days', 'MIA', 'Call Duration', 'Talk Time Duration' 
    ]
    
    for file in uploaded_files:
        try:
            df = pd.read_excel(file)
            df.columns = [col.strip() for col in df.columns]

            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

            if 'Old IC' in df.columns:
                 df['Old IC'] = df['Old IC'].astype(str)
            
            all_data.append(df)
            
        except Exception as e:
            st.error(f"Error loading main file **{file.name}**: {e}")
            return None 
            
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame() 

@st.cache_data(show_spinner=True)
def load_filter_data(uploaded_file):
    """Loads the Client/Placement filter file with robust column matching."""
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            df.columns = [col.strip() for col in df.columns]
            
            # --- Robust Column Finding ---
            ic_col = find_column(df, ['old ic', 'ic'])
            client_col = find_column(df, ['client', 'bank'])
            placement_col = find_column(df, ['placement', 'loc', 'location'])
            
            if not all([ic_col, client_col, placement_col]):
                missing = [name for name, col in zip(['Old IC', 'Client', 'Placement'], [ic_col, client_col, placement_col]) if not col]
                st.error(f"Filter file missing required column(s): **{', '.join(missing)}**. Please check spelling/header.")
                return None
            
            # Rename columns to standard names for merging
            df.rename(columns={
                ic_col: 'Old IC', 
                client_col: 'Client', 
                placement_col: 'Placement'
            }, inplace=True)
            
            df['Old IC'] = df['Old IC'].astype(str)
            return df[['Old IC', 'Client', 'Placement']].drop_duplicates(subset='Old IC')
        except Exception as e:
            st.error(f"Error loading filter file: {e}")
            return None
    return None

@st.cache_data(show_spinner=True)
def load_exclude_data(uploaded_file):
    """Loads the final exclusion list."""
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            df.columns = [col.strip() for col in df.columns]

            # Find the Old IC column robustly
            ic_col = find_column(df, ['old ic', 'ic'])
            
            if ic_col is None:
                 ic_col = df.columns[0] # Fallback to first column
            
            return set(df[ic_col].astype(str).dropna().unique())
        except Exception as e:
            st.error(f"Error loading exclusion file: {e}")
            return set()
    return set()

# --- Calculation Function (No change needed here, logic is sound) ---

def calculate_summary(df):
    """
    Calculates the counts and the sets of Old ICs for each Leads Management category.
    """
    # --- Setup Current Dates for Filtering ---
    now = datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0) 
    current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # Initialize all sets
    excluded_ics = set()
    kept_ics, ptp_ics, bp_ics, rpc_ics, rna_ics, busy_ics, btc_ics = set(), set(), set(), set(), set(), set(), set()
    summary_counts = {}
    
    # --- 0. Pre-Calculation: Identify all Future PTPs ---
    future_ptp_ics = set()
    if 'PTP Date' in df.columns:
        all_ptp_df = df[
            (df['PTP Amount'].fillna(0) > 0) & 
            (df['Status'].astype(str).str.contains(r'\bPTP\b', case=False, na=False)) & 
            ~(df['Status'].astype(str).str.contains('FF|FOLLOW UP|PTP_FFUP|PTP REMINDER', case=False, na=False))
        ].copy()
        
        if not all_ptp_df.empty:
            all_latest_ptp_dates = all_ptp_df.groupby('Old IC')['PTP Date'].max().reset_index()
            future_ptp_ics = set(
                all_latest_ptp_dates[all_latest_ptp_dates['PTP Date'].dt.normalize() >= today]['Old IC'].dropna().unique()
            )

    # --- 1. KEPT (Highest Priority) ---
    if 'Claim Paid Date' in df.columns:
        kept_df = df[
            (df['Claim Paid Amount'].fillna(0) > 0) & 
            (df['Claim Paid Date'].dt.normalize() >= current_month_start) & 
            (df['Status'].astype(str).str.contains('CONFIRMED', case=False, na=False))
        ].copy()
        kept_ics = set(kept_df['Old IC'].dropna().unique())
        excluded_ics.update(kept_ics)
    
    # --- 2. PTP (Base for BP) ---
    ptp_df = pd.DataFrame()
    if 'PTP Date' in df.columns:
        base_ptp_df = df[
            (df['PTP Amount'].fillna(0) > 0) & 
            (df['PTP Date'].dt.normalize() >= current_month_start) & 
            (df['Status'].astype(str).str.contains(r'\bPTP\b', case=False, na=False)) & 
            ~(df['Status'].astype(str).str.contains('FF|FOLLOW UP|PTP_FFUP|PTP REMINDER', case=False, na=False))
        ].copy()

        if not base_ptp_df.empty:
            latest_ptp_dates = base_ptp_df.groupby('Old IC')['PTP Date'].max().reset_index()
            latest_ptp_dates.rename(columns={'PTP Date': 'Latest PTP Date'}, inplace=True)
            ptp_df = pd.merge(base_ptp_df, latest_ptp_dates, on='Old IC', how='inner')
            ptp_df = ptp_df[
                ptp_df['PTP Date'].dt.normalize() == ptp_df['Latest PTP Date'].dt.normalize()
            ].drop(columns=['Latest PTP Date']).copy()
            ptp_ics = set(ptp_df['Old IC'].dropna().unique())

    # --- 3. BP (Broken Promise) ---
    yesterday = today - timedelta(days=1)
    if 'PTP Date' in df.columns and not ptp_df.empty:
        bp_df = ptp_df[
            (ptp_df['PTP Date'].dt.normalize() <= yesterday) 
        ].copy()
        bp_ics = set(bp_df['Old IC'].dropna().unique()) - excluded_ics
        excluded_ics.update(bp_ics)

    # --- 4. RPC (Right Party Contact) ---
    rpc_df = df[
        (df['Status'].astype(str).str.contains('POS CLIENT|RPC|POSITIVE CLIENT', case=False, na=False))
    ].copy()
    rpc_exclusions = excluded_ics.union(future_ptp_ics)
    rpc_ics = set(rpc_df['Old IC'].dropna().unique()) - rpc_exclusions
    excluded_ics.update(rpc_ics)

    # --- 5. RNA (Ring No Answer) ---
    rna_df = df[
        (df.get('Remark Type', pd.Series()).astype(str).str.contains('Follow Up', case=False, na=False)) &
        (df.get('Remark', pd.Series()).astype(str).str.contains('Predictive', case=False, na=False)) &
        (df['Status'].astype(str).str.contains('RNA', case=False, na=False))
    ].copy()
    rna_exclusions = excluded_ics.union(future_ptp_ics)
    rna_ics = set(rna_df['Old IC'].dropna().unique()) - rna_exclusions
    excluded_ics.update(rna_ics)

    # --- 6. BUSY ---
    busy_df = df[
        (df.get('Remark Type', pd.Series()).astype(str).str.contains('Follow Up', case=False, na=False)) &
        (df.get('Remark', pd.Series()).astype(str).str.contains('Predictive', case=False, na=False)) &
        (df['Status'].astype(str).str.contains('BUSY', case=False, na=False))
    ].copy()
    busy_exclusions = excluded_ics.union(future_ptp_ics)
    busy_ics = set(busy_df['Old IC'].dropna().unique()) - busy_exclusions
    excluded_ics.update(busy_ics)

    # --- 7. BTC ACTIVE (Broadcast Active) ---
    btc_df = df[
        (df['Status'].astype(str).str.contains('PM|PU', case=False, na=False)) & 
        (df.get('Remark', pd.Series()).astype(str).str.contains('Broadcast', case=False, na=False))
    ].copy()
    btc_ics_raw = set(btc_df['Old IC'].dropna().unique())
    btc_exclusions = kept_ics.union(future_ptp_ics)
    btc_ics = btc_ics_raw - btc_exclusions
    
    # Consolidate results
    results = {
        'KEPT': kept_ics,
        'PTP': ptp_ics,
        'BP': bp_ics,
        'RPC': rpc_ics,
        'RNA': rna_ics,
        'BUSY': busy_ics,
        'BTC ACTIVE': btc_ics
    }
    
    # Convert sets to counts
    summary_counts = {k: len(v) for k, v in results.items()}
    
    return summary_counts, results, len(future_ptp_ics)

# Function to convert the DataFrame to an Excel file in memory
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Old_IC_List')
    processed_data = output.getvalue()
    return processed_data

# --- Streamlit App Layout ---
def app():
    st.set_page_config(
        page_title="Leads Management Summary",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📊 Leads Management Summary Dashboard")

    # --- Sidebar Uploaders ---
    with st.sidebar:
        st.header("1. Main Data Files")
        main_files = st.file_uploader(
            "Upload Leads Data (.xlsx)", 
            type=['xlsx'], 
            accept_multiple_files=True
        )

        st.header("2. Inclusion Filter (Client/Placement)")
        filter_file = st.file_uploader(
            "Upload Filter File (Old IC, Client, Placement)", 
            type=['xlsx'], 
            key="filter_file"
        )
        
        st.header("3. Final Exclusion List")
        exclude_file = st.file_uploader(
            "Upload Final Exclusions (Old IC list)", 
            type=['xlsx'], 
            key="exclude_file"
        )

    st.markdown("---")
    
    if not main_files:
        st.info("👈 Please upload your **Main Data Files** in the sidebar to begin the analysis.")
        return

    # --- Data Loading and Filtering ---
    with st.spinner("Processing data..."):
        combined_df = load_main_data(main_files)
        filter_df = load_filter_data(filter_file)
        final_exclude_ics = load_exclude_data(exclude_file)

    if combined_df is None or combined_df.empty:
        st.error("No valid data loaded from Main Data Files.")
        return

    # --- PREPARE: Remove existing Client/Placement columns from main data (Fix B) ---
    for col in ['Client', 'Placement']:
        if col in combined_df.columns:
            combined_df.drop(columns=[col], inplace=True)

    # --- Apply Inclusion Filter (Client/Placement) ---
    if filter_df is not None and not filter_df.empty:
        # Merge main data with filter data, keeping only matching 'Old IC's
        filtered_df = pd.merge(
            combined_df, 
            filter_df, 
            on='Old IC', 
            how='inner'
        )
        st.success(f"Filter applied: **{len(filtered_df.drop_duplicates(subset='Old IC')):,}** unique Old ICs remaining from filter list.")
    else:
        st.warning("No filter file uploaded. Processing all records in the main data under 'N/A' grouping.")
        filtered_df = combined_df.copy()
    
    # GUARANTEE 'Client' and 'Placement' columns for groupby (Fix A safeguard)
    if 'Client' not in filtered_df.columns:
        filtered_df['Client'] = 'N/A (No Filter File)'
    if 'Placement' not in filtered_df.columns:
        filtered_df['Placement'] = 'N/A (No Filter File)'
        
    
    # --- Group and Calculate ---
    if filtered_df.empty:
        st.warning("No records remain after applying the inclusion filter.")
        return

    # Group by Client and Placement
    groups = filtered_df.groupby(['Client', 'Placement'])
    all_aligned_ids = []

    st.header("🎯 Leads Management Summary Results")

    for (client, placement), group_df in groups:
        st.subheader(f"Results for: Client **{client}** | Placement **{placement}**")

        # Calculate summary for the current group
        with st.expander("Show Detailed Calculation for this Group", expanded=False):
            st.markdown(f"Total records in this group: {len(group_df):,}")
            summary_counts, ics_by_category, future_ptp_count = calculate_summary(group_df)
            st.info(f"Active/Future PTPs ($\ge$ Today) excluded from RPC, RNA, BUSY, BTC: **{future_ptp_count:,}**")

        # --- Apply Final Exclusion ---
        if final_exclude_ics:
            st.info(f"Applying **Final Exclusion List** (Contains {len(final_exclude_ics):,} Old ICs) to all categories.")
            
            # Recalculate sets after applying final exclusion
            for category, ids in ics_by_category.items():
                final_ids = ids - final_exclude_ics
                ics_by_category[category] = final_ids
                summary_counts[category] = len(final_ids)

        # --- Display Summary Counts ---
        summary_list = [{'Category': k, 'Count (Old IC)': v} for k, v in summary_counts.items()]
        summary_table_counts = pd.DataFrame(summary_list)
        total_count = sum(summary_counts.values())
        
        st.dataframe(
            summary_table_counts.style.format({'Count (Old IC)': "{:,}"})
                         .set_properties(**{'font-weight': 'bold'}) 
                         .set_table_styles([{'selector': 'th', 'props': [('background-color', '#f0f2f6')]}])
                         .set_caption(f"Total Unique Old ICs accounted for in these categories: {total_count:,}"),
            hide_index=True,
            use_container_width=True
        )
        
        # --- Prepare Aligned Old IC List for this group ---
        
        # 1. Convert sets to sorted lists
        data_for_df = {k: sorted(list(v)) for k, v in ics_by_category.items()}

        # 2. Find the maximum length to align columns
        max_len = max(len(v) for v in data_for_df.values())

        # 3. Pad shorter lists with None for alignment
        aligned_data = {}
        for k, v in data_for_df.items():
            aligned_data[k] = v + [None] * (max_len - len(v))

        # 4. Create the final DataFrame for display and download
        summary_table_ids = pd.DataFrame(aligned_data)
        
        # Add Client/Placement columns for the download file
        summary_table_ids.insert(0, 'Client', client)
        summary_table_ids.insert(1, 'Placement', placement)
        
        all_aligned_ids.append(summary_table_ids)
        
        st.markdown("#### Old IC List by Category (Aligned)")
        st.dataframe(
            summary_table_ids.drop(columns=['Client', 'Placement']).fillna(''),
            use_container_width=True
        )
        
        st.markdown("---")

    # --- Combined Download Button ---
    if all_aligned_ids:
        final_download_df = pd.concat(all_aligned_ids, ignore_index=True).fillna('')
        
        st.header("⬇️ Download Combined Results")
        st.download_button(
            label="Download All Old IC Lists (Grouped & Aligned)",
            data=to_excel(final_download_df),
            file_name="Leads_Management_Filtered_IC_List.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Downloads a single Excel file containing the aligned Old IC lists, grouped by Client and Placement."
        )


if __name__ == "__main__":

    app()
