import streamlit as st
import pandas as pd
from datetime import date, timedelta
import os
import requests
import calendar
from pandas.tseries.offsets import MonthBegin
import plotly.express as px
import plotly.graph_objects as go



# ---------------------------------------------------------
# Ustawienia podstawowe
# ---------------------------------------------------------
st.set_page_config(
    page_title="myAPP",
    page_icon="💰",
    layout="wide"
)

CSV_FORM_PATH = "db.csv"

# ---------------------------------------------------------
# Funkcje pomocnicze
# ---------------------------------------------------------

@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data
def load_form_data():
    """Wczytuje plik CSV używany przez formularz. 
    Jeśli plik nie istnieje, zwraca pustą ramkę."""
    if os.path.exists(CSV_FORM_PATH):
        return pd.read_csv(CSV_FORM_PATH)
    else:
        return pd.DataFrame()


def save_form_row(row_dict):
    """Dodaje jeden wiersz do CSV_FORM_PATH, zachowując kolejność kolumn."""
    new_row_df = pd.DataFrame([row_dict])

    if os.path.exists(CSV_FORM_PATH):
        df_existing = pd.read_csv(CSV_FORM_PATH)
        # upewniamy się, że mamy te same kolumny
        for col in df_existing.columns:
            if col not in new_row_df.columns:
                new_row_df[col] = ""
        new_row_df = new_row_df[df_existing.columns]
        df_combined = pd.concat([df_existing, new_row_df], ignore_index=True)
        df_combined.to_csv(CSV_FORM_PATH, index=False)
    else:
        # pierwszy zapis – użyjemy kolejności kluczy z row_dict
        new_row_df.to_csv(CSV_FORM_PATH, index=False)


def sample_budget_data():
    data = {
        "Kategoria": ["Mieszkanie", "Jedzenie", "Transport", "Rozrywka", "Inne"],
        "Plan": [2500, 1200, 400, 600, 300],
        "Rzeczywiste": [2450, 1350, 500, 550, 280]
    }
    return pd.DataFrame(data)

def sample_usd_data():
    # Przykladowe dane – w realnej aplikacji możesz je pobrać z API NBP/ECB/FX itp.
    dates = pd.date_range(end=date.today(), periods=30)
    usd_rates = [4.10 + 0.05 * (i % 5) for i in range(30)]
    return pd.DataFrame({"Data": dates, "Kurs_USD": usd_rates})

@st.cache_data
def fetch_nbp_rates(code: str, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Pobiera kursy średnie z NBP API (tabela A) dla danej waluty (np. 'USD', 'EUR')
    i zakresu dat. Zwraca DataFrame z kolumnami: Data, Kurs.
    """
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    url = f"http://api.nbp.pl/api/exchangerates/rates/A/{code}/{start_str}/{end_str}/?format=json"

    resp = requests.get(url)
    resp.raise_for_status()

    data = resp.json()
    rates = data.get("rates", [])

    df = pd.DataFrame(rates)
    df.rename(columns={"effectiveDate": "Data", "mid": "Kurs"}, inplace=True)
    df["Data"] = pd.to_datetime(df["Data"])

    return df[["Data", "Kurs"]].sort_values("Data")


# ---------------------------------------------------------
# Nawigacja między stronami
# ---------------------------------------------------------

st.sidebar.title("📊 Menu")
page = st.sidebar.radio(
    "Wybierz stronę:",
    (
        "📝 Formularz (CSV)",
        "📂 Przeglądanie CSV",
        "📈 Wykresy budżetu",
        "💵 Kurs USD",
        "📅 Kalendarz"
    )
)

# ---------------------------------------------------------
# 1. Formularz – użytkownik wybiera plik CSV z kolumnami
# ---------------------------------------------------------
if page == "📝 Formularz (CSV)":
    st.title("📝 Formularz generowany na podstawie wybranego CSV")

    uploaded_file = st.file_uploader(
        "Wybierz plik CSV, który ma definiować pola formularza",
        type=["csv"],
        key="form_csv"
    )

    if uploaded_file is None:
        st.info("Wgraj plik CSV, aby pojawił się formularz.")
        st.stop()

    # Wczytujemy dane
    df_form = pd.read_csv(uploaded_file)
    columns = df_form.columns.tolist()

    if not columns:
        st.error("CSV nie zawiera żadnych kolumn (nagłówków).")
        st.stop()

    st.write("🔍 Kolumny wykryte w pliku:")
    st.code(", ".join(columns))

    # -------------------------------------------
    # FORMULARZ - generowany dynamicznie
    # -------------------------------------------
    with st.form("dynamic_form"):
        inputs = {}

        for col in columns:
            label = col
            lower = col.lower()

            # Typ ID
            if lower == "id" and pd.api.types.is_numeric_dtype(df_form[col]):
                next_id = int(df_form[col].max()) + 1 if not df_form.empty else 1
                inputs[col] = st.number_input(f"{label} (ID)", value=next_id, step=1)

            # Typ data
            elif "date" in lower or "data" in lower:
                inputs[col] = st.date_input(label)

            # Pole numeryczne
            elif pd.api.types.is_numeric_dtype(df_form[col]):
                inputs[col] = st.number_input(label, value=0.0)

            # Domyślnie tekst
            else:
                inputs[col] = st.text_input(label, "")

        submitted = st.form_submit_button("Zapisz do pliku")

    # -------------------------------------------
    # Zapis do pliku UŻYTKOWNIKA (nie stały plik!)
    # -------------------------------------------
    if submitted:
        # Tworzymy nowy wiersz
        new_row = {}
        for col in columns:
            val = inputs[col]

            # Konwersja daty na tekst
            if hasattr(val, "isoformat"):
                val = val.isoformat()

            new_row[col] = val

        df_new = pd.DataFrame([new_row])

        # Łączymy z istniejącą ramką
        df_out = pd.concat([df_form, df_new], ignore_index=True)

        # Nadpisujemy plik użytkownika
        df_out.to_csv("uploaded_form_output.csv", index=False)

        st.success("✅ Zapisano dane do pliku: `uploaded_form_output.csv`")

        st.download_button(
            label="⬇️ Pobierz zaktualizowany plik",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="updated_form.csv",
            mime="text/csv"
        )

        st.info(
            "Dane zostały zapisane lokalnie jako `uploaded_form_output.csv`, "
            "ale możesz również pobrać bezpośrednio z przycisku powyżej."
        )

    # -------------------------------------------
    # Podgląd danych
    # -------------------------------------------
    st.subheader("📄 Aktualna zawartość wgranego pliku")
    st.dataframe(df_form, use_container_width=True)



# ---------------------------------------------------------
# 2. Przeglądanie plików CSV + filtrowanie po kolumnach
# ---------------------------------------------------------
elif page == "📂 Przeglądanie CSV":
    st.title("📂 Przeglądanie plików CSV")

    uploaded_file = st.file_uploader("Wgraj plik CSV", type=["csv"], key="browse_csv")

    if uploaded_file is not None:
        df = load_csv(uploaded_file)

        st.subheader("🔍 Filtrowanie danych")

        # Kopia do filtrowania
        filtered_df = df.copy()

        with st.expander("Pokaż / ukryj opcje filtrowania", expanded=False):
            st.write("Wybierz kolumny, po których chcesz filtrować:")

            cols_to_filter = st.multiselect(
                "Kolumny do filtrowania",
                options=list(df.columns),
                default=[]
            )

            for col in cols_to_filter:
                col_series = df[col]
                col_type = col_series.dtype

                st.markdown(f"**Filtr dla kolumny: `{col}`**")

                # NUMERYCZNE
                if pd.api.types.is_numeric_dtype(col_series):
                    min_val = float(col_series.min())
                    max_val = float(col_series.max())
                    if min_val == max_val:
                        st.info(f"W kolumnie `{col}` wszystkie wartości są równe: {min_val}")
                    f_min, f_max = st.slider(
                        f"Zakres wartości dla `{col}`",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        key=f"slider_{col}"
                    )
                    filtered_df = filtered_df[
                        (filtered_df[col] >= f_min) & (filtered_df[col] <= f_max)
                    ]

                # DATY
                elif pd.api.types.is_datetime64_any_dtype(col_series):
                    min_date = col_series.min().date()
                    max_date = col_series.max().date()
                    start_date, end_date = st.date_input(
                        f"Zakres dat dla `{col}`",
                        value=(min_date, max_date),
                        key=f"date_{col}"
                    )
                    if start_date > end_date:
                        st.warning("Data początkowa jest późniejsza niż końcowa – filtr pominięty.")
                    else:
                        mask = (
                            filtered_df[col].dt.date >= start_date
                        ) & (
                            filtered_df[col].dt.date <= end_date
                        )
                        filtered_df = filtered_df[mask]

                # TEKST / INNE
                else:
                    text = st.text_input(
                        f"Szukaj (fragment) w `{col}`",
                        value="",
                        key=f"text_{col}"
                    )
                    if text:
                        filtered_df = filtered_df[
                            filtered_df[col].astype(str).str.contains(text, case=False, na=False)
                        ]

                st.markdown("---")

        # PODGLĄD PO FILTRACH
        st.subheader("📄 Podgląd danych (po zastosowaniu filtrów)")
        st.dataframe(filtered_df, use_container_width=True)

        st.subheader("📊 Informacje o ramce danych")
        col1, col2 = st.columns(2)

        with col1:
            st.write("🔹 Kształt (rows, cols):", filtered_df.shape)
            st.write("🔹 Kolumny:")
            st.write(filtered_df.columns.tolist())

        with col2:
            st.write("🔹 Typy danych:")
            st.write(filtered_df.dtypes)

        st.subheader("📈 Podstawowe statystyki (numeryczne)")
        if not filtered_df.select_dtypes(include="number").empty:
            st.write(filtered_df.describe())
        else:
            st.info("Brak kolumn numerycznych do pokazania statystyk.")
    else:
        st.info("Wgraj plik CSV, aby zobaczyć dane i opcje filtrowania.")


# ---------------------------------------------------------
# 3. Budżet – bez dat, każdy wiersz to miesięczne kwoty
# ---------------------------------------------------------
elif page == "📈 Wykresy budżetu":
    st.title("📈 Budżet – miesięczny i prognoza na 12 miesięcy")

    st.write(
        """
        Wgraj plik CSV, w którym:
        - kolumna **Salary** zawiera miesięczne wynagrodzenie (może być w kilku wierszach – zostanie zsumowane),
        - wszystkie **pozostałe kolumny są traktowane jako koszty miesięczne**.

        Przykład:

        | Salary | Rent | Food | Fuel | Entertainment |
        |--------|------|------|------|---------------|
        | 6000   | 2500 | 800  | 300  | 200           |
        | 0      | 0    | 200  | 0    | 0             |
        """
    )

    uploaded_budget = st.file_uploader("Wgraj CSV budżetowy", type=["csv"], key="budget_csv")

    if uploaded_budget is None:
        st.info("Wgraj plik CSV, aby kontynuować.")
        st.stop()

    # ---------------------------
    # Wczytanie danych
    # ---------------------------
    df_raw = pd.read_csv(uploaded_budget)

    st.subheader("📄 Surowe dane")
    st.dataframe(df_raw, use_container_width=True)

    if df_raw.empty:
        st.error("Plik CSV jest pusty.")
        st.stop()

    # Szukamy kolumny Salary (case-insensitive)
    columns_lower = {col.lower(): col for col in df_raw.columns}
    if "salary" not in columns_lower:
        st.error("Brak wymaganej kolumny **Salary** w pliku CSV.")
        st.stop()

    salary_col = columns_lower["salary"]
    cost_columns = [c for c in df_raw.columns if c != salary_col]

    if not cost_columns:
        st.error("Musi istnieć co najmniej jedna kolumna kosztowa (poza Salary).")
        st.stop()

    st.write("🔍 Wykryte kolumny kosztowe:", cost_columns)

    # ---------------------------
    # Miesięczny budżet bazowy
    # ---------------------------
    monthly_salary = df_raw[salary_col].sum()
    monthly_costs = df_raw[cost_columns].sum().sum()
    monthly_saldo = monthly_salary - monthly_costs

    st.subheader("📆 Miesięczny budżet bazowy (na podstawie CSV)")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Suma Salary / miesiąc", f"{monthly_salary:,.2f}")
    with c2:
        st.metric("Suma kosztów / miesiąc", f"{monthly_costs:,.2f}")
    with c3:
        st.metric("Saldo / miesiąc", f"{monthly_saldo:,.2f}")

    # ---------------------------
    # Prognoza na 12 miesięcy
    # ---------------------------
    st.subheader("🔮 Prognoza na kolejne 12 miesięcy")

    # Tutaj już nie ma historii ani dat – bierzemy po prostu stałe wartości
    months_labels = [f"Miesiąc {i}" for i in range(1, 13)]

    df_forecast = pd.DataFrame({
        "Miesiąc": months_labels,
        "Koszty_plan": monthly_costs,
        "Salary_plan": monthly_salary
    })
    df_forecast["Saldo_plan"] = df_forecast["Salary_plan"] - df_forecast["Koszty_plan"]

    st.markdown("### 📋 Tabela prognozy (12 miesięcy)")
    st.dataframe(
        df_forecast[["Miesiąc", "Koszty_plan", "Salary_plan", "Saldo_plan"]],
        use_container_width=True
    )

    # ---------------------------
    # Wykres – Plotly (koszty vs salary + saldo)
    # ---------------------------
    st.markdown("### 📊 Wykres budżetu – prognoza")

    x = df_forecast["Miesiąc"]

    fig = go.Figure()

    fig.add_bar(
        name="Koszty (plan)",
        x=x,
        y=df_forecast["Koszty_plan"]
    )
    fig.add_bar(
        name="Salary (plan)",
        x=x,
        y=df_forecast["Salary_plan"]
    )

    fig.add_trace(
        go.Scatter(
            name="Saldo (plan)",
            x=x,
            y=df_forecast["Saldo_plan"],
            mode="lines+markers",
            yaxis="y2"
        )
    )

    fig.update_layout(
        barmode="group",
        xaxis_title="Miesiąc",
        yaxis_title="Kwota",
        yaxis2=dict(
            title="Saldo",
            overlaying="y",
            side="right"
        ),
        height=400,
        margin=dict(l=20, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# 4. Wykres zmian kursów walut (USD / EUR, API NBP + Plotly)
# ---------------------------------------------------------
elif page == "💵 Kurs USD":
    st.title("💵 Zmiany kursów walut (NBP)")

    st.write(
        "Dane dla USD i EUR są pobierane z oficjalnego API NBP (tabela A – kursy średnie)."
    )

    # Wybór waluty (PLN usunięty)
    currency = st.selectbox(
        "Wybierz walutę",
        ["USD", "EUR"]
    )

    # Wybór zakresu dat (domyślnie ostatnie 30 dni)
    today = date.today()
    default_start = today - timedelta(days=30)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Data początkowa", value=default_start)
    with col2:
        end_date = st.date_input("Data końcowa", value=today)

    if start_date > end_date:
        st.error("Data początkowa nie może być późniejsza niż końcowa.")
    else:
        try:
            # Pobranie danych z API NBP
            df_currency = fetch_nbp_rates(currency, start_date, end_date)

            if df_currency.empty:
                st.warning("Brak danych dla wybranego zakresu dat (np. same weekendy/święta).")
            else:
                # Filtr bezpieczeństwa
                mask = (df_currency["Data"].dt.date >= start_date) & (df_currency["Data"].dt.date <= end_date)
                df_filtered = df_currency[mask].copy()

                st.subheader(f"Tabela kursu {currency}")
                st.dataframe(df_filtered, use_container_width=True)

                # ----------------------------
                # Wykres – Plotly
                # ----------------------------
                st.subheader(f"Wykres kursu {currency} (NBP)")

                fig = px.line(
                    df_filtered,
                    x="Data",
                    y="Kurs",
                    title=f"Kurs {currency} w PLN",
                    markers=True
                )

                fig.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20),
                    title=dict(x=0.5),
                )

                fig.update_traces(line=dict(width=2))

                st.plotly_chart(fig, use_container_width=True)

        except requests.HTTPError as e:
            st.error(f"Błąd HTTP podczas pobierania danych z NBP: {e}")
        except Exception as e:
            st.error(f"Wystąpił nieoczekiwany błąd: {e}")



# ---------------------------------------------------------
# 5. Kalendarz – widok miesięczny lub roczny
# ---------------------------------------------------------
elif page == "📅 Kalendarz":
    st.title("📅 Kalendarz")

    st.write(
        "Wybierz, czy chcesz zobaczyć kalendarz dla konkretnego miesiąca, czy dla całego roku."
    )

    today = date.today()

    # Wybór trybu – miesiąc albo rok
    mode = st.radio(
        "Tryb widoku",
        ["Miesiąc", "Rok"],
        horizontal=True
    )

    # Funkcja rysująca kalendarz jednego miesiąca jako tabelę HTML
    def render_month_calendar(year: int, month: int):
        weekday_names = ["Pn", "Wt", "Śr", "Cz", "Pt", "So", "Nd"]
        cal = calendar.monthcalendar(year, month)

        html = "<table style='border-collapse: collapse; width: 100%; text-align: center; margin-bottom: 1rem;'>"
        html += "<tr>" + "".join(
            f"<th style='padding:4px; border-bottom:1px solid #bbb;'>{d}</th>" for d in weekday_names
        ) + "</tr>"

        for week in cal:
            html += "<tr>"
            for day in week:
                if day == 0:
                    html += "<td style='padding:6px; color:#ccc;'> </td>"
                else:
                    style = "padding:6px; border:1px solid #eee;"
                    # weekend – delikatne tło
                    # weekday(): 0=Pn ... 6=Nd
                    day_weekday = calendar.weekday(year, month, day)
                    if day_weekday >= 5:
                        style += " background-color:#fafafa;"
                    html += f"<td style='{style}'>{day}</td>"
            html += "</tr>"

        html += "</table>"
        st.markdown(html, unsafe_allow_html=True)

    # -----------------------------
    # Tryb: MIESIĄC
    # -----------------------------
    if mode == "Miesiąc":
        col1, col2 = st.columns(2)
        with col1:
            selected_year = st.number_input(
                "Rok",
                min_value=1900,
                max_value=2100,
                value=today.year,
                step=1
            )
        with col2:
            selected_month = st.selectbox(
                "Miesiąc",
                options=list(range(1, 13)),
                index=today.month - 1,
                format_func=lambda m: calendar.month_name[m]
            )

        st.subheader(f"{calendar.month_name[selected_month]} {selected_year}")
        render_month_calendar(int(selected_year), int(selected_month))

    # -----------------------------
    # Tryb: ROK
    # -----------------------------
    else:  # mode == "Rok"
        selected_year = st.number_input(
            "Rok",
            min_value=1900,
            max_value=2100,
            value=today.year,
            step=1
        )

        st.subheader(f"Kalendarz na rok {int(selected_year)}")

        # Po kolei każdy miesiąc w roku
        for m in range(1, 13):
            st.markdown(f"### {calendar.month_name[m]} {int(selected_year)}")
            render_month_calendar(int(selected_year), m)
            st.markdown("---")
