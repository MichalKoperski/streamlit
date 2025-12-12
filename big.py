import streamlit as st
import pandas as pd
from datetime import date, timedelta
import os
import requests
import calendar
from pandas.tseries.offsets import MonthBegin
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from io import StringIO
import re
from urllib.parse import urljoin


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
        "📈 Budżet",
        "💵 Kursy",
        "📅 Kalendarz",
        "🧾 Edytor Markdown",
        "🤼 PPV: WCW i WWF/WWE"
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
elif page == "📈 Budżet":
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
elif page == "💵 Kursy":
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

                st.subheader(f"Tabela kursu {currency}")
                st.dataframe(df_filtered, use_container_width=True)

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
# ---------------------------------------------------------
# 6. Edytor Markdown (upload + edycja)
# ---------------------------------------------------------
elif page == "🧾 Edytor Markdown":
    st.title("🧾 Edytor Markdown")

    st.write(
        "Możesz **wgrać plik Markdown (.md)** lub pisać od zera. "
        "Po lewej edycja, po prawej podgląd na żywo."
    )

    # ---------------------------------
    # Upload pliku Markdown
    # ---------------------------------
    uploaded_md = st.file_uploader(
        "Wgraj plik Markdown (.md)",
        type=["md"],
        key="md_uploader"
    )

    # Domyślna treść
    default_md = """# Nowy dokument Markdown

Możesz:
- pisać od zera
- albo wgrać istniejący plik `.md`

**Markdown działa od razu.**
"""

    # Jeśli użytkownik wgrał plik – czytamy jego zawartość
    if uploaded_md is not None:
        try:
            md_text = uploaded_md.read().decode("utf-8")
            file_name = uploaded_md.name
        except Exception:
            st.error("Nie udało się odczytać pliku Markdown.")
            md_text = default_md
            file_name = "dokument.md"
    else:
        md_text = default_md
        file_name = "dokument.md"

    col1, col2 = st.columns(2)

    # ---------------------------------
    # Edytor
    # ---------------------------------
    with col1:
        md_text = st.text_area(
            "Edytor Markdown",
            value=md_text,
            height=450
        )

        st.download_button(
            label="⬇️ Pobierz jako .md",
            data=md_text.encode("utf-8"),
            file_name=file_name,
            mime="text/markdown"
        )

    # ---------------------------------
    # Podgląd
    # ---------------------------------
    with col2:
        st.markdown("### 👀 Podgląd")
        st.markdown(md_text)
# ---------------------------------------------------------
# X. PPV: WCW i WWF/WWE (Wikipedia: lista + szczegóły)
# ---------------------------------------------------------
elif page == "🤼 PPV: WCW i WWF/WWE":
    st.title("🤼 PPV: WCW i WWF/WWE – daty, info i match card")

    WIKI_BASE = "https://en.wikipedia.org"
    WWE_LIST_URL = "https://en.wikipedia.org/wiki/List_of_WWE_pay-per-view_and_livestreaming_supercards"
    WCW_LIST_URL = "https://en.wikipedia.org/wiki/List_of_JCP/WCW_closed-circuit_events_and_pay-per-view_events"

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (StreamlitApp; +https://streamlit.io) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"
    }

    # -------------------------
    # Helpers
    # -------------------------
    def make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Arrow/Streamlit nie toleruje duplikatów nazw kolumn."""
        cols = []
        seen = {}
        for c in df.columns:
            c = str(c)
            if c in seen:
                seen[c] += 1
                cols.append(f"{c}.{seen[c]}")
            else:
                seen[c] = 0
                cols.append(c)
        out = df.copy()
        out.columns = cols
        return out

    def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Spłaszcza MultiIndex kolumn do stringów."""
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = [
                " ".join([str(x) for x in tup if str(x) != "nan"]).strip()
                for tup in out.columns.values
            ]
        else:
            out.columns = [str(c) for c in out.columns]
        return out

    def normalize_event_name(s: str) -> str:
        s = re.sub(r"\[.*?\]", "", str(s)).strip()  # usuń przypisy typu [1]
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def get_html(url: str) -> str:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} podczas pobierania: {url}")
        return r.text

    def find_best_event_table_soup(soup: BeautifulSoup) -> BeautifulSoup | None:
        """
        Znajdź tabelę (wikitable), która wygląda jak lista eventów:
        - ma nagłówek zawierający 'Event' i 'Date'
        """
        tables = soup.select("table.wikitable")
        best = None
        best_score = -1

        for t in tables:
            # nagłówki
            headers = [th.get_text(" ", strip=True).lower() for th in t.select("tr th")]
            # score: czy zawiera event + date
            has_event = any("event" in h for h in headers)
            has_date = any("date" in h for h in headers)
            if not (has_event and has_date):
                continue

            # score po liczbie wierszy
            rows = t.select("tr")
            score = len(rows)
            if score > best_score:
                best_score = score
                best = t

        return best

    def extract_event_links_from_table(table: BeautifulSoup) -> dict[str, str]:
        """
        Z tabeli wyciąga mapowanie: EventClean -> pełny URL Wikipedii.
        Wybiera pierwszego sensownego linka w komórce eventu.
        """
        event_to_url = {}

        # rozpoznaj indeks kolumny "Event"
        header_cells = table.select("tr th")
        header_texts = [h.get_text(" ", strip=True).lower() for h in header_cells]

        # czasem w tabelach nagłówek jest w pierwszym wierszu, ale bywa wielowierszowy
        # więc bierzemy pierwszy wiersz z th w tej tabeli:
        first_header_row = None
        for tr in table.select("tr"):
            ths = tr.find_all("th")
            if ths and tr.find_all("td") == []:
                first_header_row = tr
                break

        if first_header_row:
            header_texts = [th.get_text(" ", strip=True).lower() for th in first_header_row.find_all("th")]

        event_idx = None
        for i, h in enumerate(header_texts):
            if "event" in h:
                event_idx = i
                break

        # iteruj po wierszach danych
        for tr in table.select("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue

            # jeśli nie wykryliśmy event_idx, spróbuj w całym wierszu znaleźć link z tytułem
            if event_idx is None or event_idx >= len(tds):
                a = tr.find("a", href=True)
                if a:
                    name = normalize_event_name(a.get_text(" ", strip=True))
                    href = urljoin(WIKI_BASE, a["href"])
                    if name and "/wiki/" in href:
                        event_to_url.setdefault(name, href)
                continue

            # normalnie: weź komórkę eventu
            cell = tds[event_idx]
            a = cell.find("a", href=True)
            if not a:
                continue
            name = normalize_event_name(cell.get_text(" ", strip=True))
            href = urljoin(WIKI_BASE, a["href"])
            if name and "/wiki/" in href:
                event_to_url.setdefault(name, href)

        return event_to_url

    @st.cache_data(show_spinner=False, ttl=60 * 60)
    def load_events_list(promo: str) -> tuple[pd.DataFrame, dict[str, str]]:
        """
        Zwraca:
        - df: kolumny (Date, Event, Location?, Notes?)
        - event_links: mapowanie Event -> URL
        """
        list_url = WWE_LIST_URL if promo == "WWE" else WCW_LIST_URL
        html = get_html(list_url)
        soup = BeautifulSoup(html, "html.parser")

        table = find_best_event_table_soup(soup)
        if table is None:
            return pd.DataFrame(), {}

        # linki eventów (prawdziwe href)
        event_links = extract_event_links_from_table(table)

        # tabela jako DataFrame
        tables = pd.read_html(StringIO(str(table)))
        if not tables:
            return pd.DataFrame(), event_links

        df = tables[0]
        df = flatten_columns(df)
        df = make_unique_columns(df)

        # normalizacja nazw kolumn
        rename = {}
        for c in df.columns:
            lc = str(c).strip().lower()
            if "date" in lc:
                rename[c] = "Date"
            elif "event" in lc:
                rename[c] = "Event"
            elif "location" in lc or "venue" in lc or "city" in lc:
                rename[c] = "Location"
            elif "notes" in lc:
                rename[c] = "Notes"
        df = df.rename(columns=rename)

        if "Event" not in df.columns or "Date" not in df.columns:
            return pd.DataFrame(), event_links

        df["Event"] = df["Event"].map(normalize_event_name)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # opcjonalne kolumny
        keep = [c for c in ["Date", "Event", "Location", "Notes"] if c in df.columns]
        df = df[keep].copy()

        # dołącz URL jeśli udało się dopasować nazwę
        df["URL"] = df["Event"].map(lambda n: event_links.get(n, ""))

        # usuń puste
        df = df[df["Event"].astype(str).str.len() > 0].copy()
        df = df.sort_values("Date", na_position="last").reset_index(drop=True)

        # ostatecznie unikalne kolumny
        df = make_unique_columns(df)
        return df, event_links

    @st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
    def fetch_event_details(url: str) -> dict:
        """
        Pobiera:
        - infobox (dict)
        - results table (DataFrame or None)
        """
        html = get_html(url)
        soup = BeautifulSoup(html, "html.parser")

        # infobox
        infobox = {}
        ib = soup.select_one("table.infobox")
        if ib:
            for row in ib.select("tr"):
                th = row.find("th")
                td = row.find("td")
                if th and td:
                    k = th.get_text(" ", strip=True)
                    v = td.get_text(" ", strip=True)
                    if k and v:
                        infobox[k] = v

        # results
        results_df = None
        try:
            tables = pd.read_html(StringIO(html))
            for t in tables:
                t = flatten_columns(t)
                t = make_unique_columns(t)
                cols = [str(c).lower() for c in t.columns]

                # heurystyka wyników
                if (
                    any("match" in c for c in cols)
                    or any("winner" in c for c in cols)
                    or any("stipulation" in c for c in cols)
                    or any("result" in c for c in cols)
                ):
                    if len(t) >= 2:
                        results_df = t
                        break
        except Exception:
            results_df = None

        return {"infobox": infobox, "results": results_df}

    # -------------------------
    # UI
    # -------------------------
    promo = st.selectbox("Federacja", ["WWE", "WCW"], index=0)

    try:
        df, links = load_events_list(promo)
    except Exception as e:
        st.error(f"Błąd pobierania listy PPV: {e}")
        st.stop()

    if df.empty:
        st.warning("Nie udało się pobrać listy eventów (brak tabeli lub zmiana struktury strony).")
        st.stop()

    # filtry
    years = sorted([int(y) for y in df["Date"].dropna().dt.year.unique()]) if "Date" in df.columns else []
    c1, c2, c3 = st.columns([1, 2, 1])

    with c1:
        year = st.selectbox("Rok", ["Wszystkie"] + years, index=0)
    with c2:
        q = st.text_input("Szukaj w nazwie eventu", value="")
    with c3:
        limit = st.number_input("Limit wyników", min_value=10, max_value=2000, value=200, step=10)

    view = df.copy()
    if year != "Wszystkie":
        view = view[view["Date"].dt.year == int(year)]
    if q.strip():
        view = view[view["Event"].astype(str).str.contains(q.strip(), case=False, na=False)]
    view = view.head(int(limit)).reset_index(drop=True)

    # Arrow-safe
    view = make_unique_columns(view)

    st.subheader("Lista eventów")
    st.dataframe(view, use_container_width=True, height=380)

    st.markdown("---")
    st.subheader("Szczegóły eventu")

    if view.empty:
        st.info("Brak wyników dla wybranych filtrów.")
        st.stop()

    # wybór eventu
    default_idx = 0
    selected_event = st.selectbox("Wybierz event", view["Event"].tolist(), index=default_idx)

    # URL eventu
    selected_url = view.loc[view["Event"] == selected_event, "URL"].iloc[0] if "URL" in view.columns else ""
    if not selected_url:
        st.warning("Nie udało się dopasować linku do strony eventu (brak href w tabeli).")
        st.stop()

    st.markdown(f"**Wikipedia URL:** {selected_url}")

    if st.button("Pobierz szczegóły", type="primary"):
        with st.spinner("Pobieram dane eventu..."):
            details = fetch_event_details(selected_url)

        # infobox
        if details["infobox"]:
            st.markdown("### Info (Infobox)")
            info_df = pd.DataFrame([{"Pole": k, "Wartość": v} for k, v in details["infobox"].items()])
            info_df = make_unique_columns(info_df)
            st.dataframe(info_df, use_container_width=True, height=280)
        else:
            st.info("Nie znaleziono infobox na stronie eventu.")

        # results
        if details["results"] is not None and not details["results"].empty:
            st.markdown("### Matches / Results")
            res = make_unique_columns(details["results"])
            st.dataframe(res, use_container_width=True, height=450)
        else:
            st.info("Nie udało się automatycznie znaleźć tabeli Results/Match card na tej stronie.")
