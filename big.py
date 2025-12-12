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
        "🧪 Notebook (bezpieczny)",
        "📈 Budżet",
        "💵 Kursy",
        "📅 Kalendarz",
        "🧾 Edytor Markdown",
        "🤼 PPV (CSV: WWE/WCW)"
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
# 3. Mini-notebook (bezpieczny): multi-cells + save/load
# ---------------------------------------------------------
elif page == "🧪 Notebook (bezpieczny)":
    import json
    import os
    import pandas as pd
    import streamlit as st

    st.title("🧪 Mini-Notebook (bezpieczny)")
    st.caption("Wiele komórek, Run/Run all, historia wyników, zapis/odczyt JSON. Bez importów i bez dostępu do systemu.")

    # -------------------------
    # Helpers / safety
    # -------------------------
    SAFE_BUILTINS = {
        "len": len, "min": min, "max": max, "sum": sum, "sorted": sorted,
        "round": round, "range": range, "enumerate": enumerate, "zip": zip,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "str": str, "int": int, "float": float, "bool": bool,
        "abs": abs, "all": all, "any": any,
    }

    banned_tokens = [
        "import ", "__", "open(", "exec(", "eval(", "compile(",
        "os.", "sys.", "subprocess", "socket", "pathlib", "shutil",
        "requests", "urllib", "http", "pip", "conda"
    ]

    def code_is_safe(code: str) -> tuple[bool, str]:
        for t in banned_tokens:
            if t in code:
                return False, f"Niedozwolone użycie: `{t}`"
        return True, ""

    # plotly.express w bezpieczny sposób (bez normalnego importu w kodzie użytkownika)
    px = __import__("plotly.express", fromlist=["express"])

    # -------------------------
    # Data source
    # -------------------------
    st.subheader("Dane (df)")

    data_source = st.radio(
        "Źródło danych dla `df`",
        ["📤 Upload CSV", "📁 Pliki z repo: wwe.csv / wcw.csv"],
        horizontal=True
    )

    df = None

    if data_source == "📤 Upload CSV":
        up = st.file_uploader("Wgraj CSV", type=["csv"], key="nb_upload")
        if up is not None:
            df = pd.read_csv(up)
            st.session_state["nb_df"] = df
        else:
            df = st.session_state.get("nb_df")

    else:
        # repo files option
        wwe_path = "wwe.csv"
        wcw_path = "wcw.csv"

        available = []
        if os.path.exists(wwe_path):
            available.append("wwe.csv")
        if os.path.exists(wcw_path):
            available.append("wcw.csv")

        if not available:
            st.warning("Nie widzę `wwe.csv` ani `wcw.csv` w katalogu aplikacji. Dodaj je do repo (root).")
        else:
            pick = st.selectbox("Wybierz plik", available)
            try:
                df = pd.read_csv(pick)
                st.session_state["nb_df"] = df
            except Exception as e:
                st.error(f"Nie udało się wczytać {pick}: {e}")

    if df is None:
        st.info("Wybierz źródło danych, aby rozpocząć.")
        st.stop()

    st.markdown("**Podgląd `df`**")
    st.dataframe(df.head(30), use_container_width=True)

    # -------------------------
    # Notebook state
    # -------------------------
    if "nb_cells" not in st.session_state:
        st.session_state["nb_cells"] = [
            {
                "id": 1,
                "title": "Komórka 1",
                "code": "out_df = df.head(10)\n# fig = px.histogram(df, x=df.columns[0])",
                "last_text": "",
                "has_table": False,
                "has_fig": False,
            }
        ]
        st.session_state["nb_next_id"] = 2

    def add_cell():
        cid = st.session_state["nb_next_id"]
        st.session_state["nb_next_id"] += 1
        st.session_state["nb_cells"].append({
            "id": cid,
            "title": f"Komórka {cid}",
            "code": "# out_df = ...\n# fig = ...",
            "last_text": "",
            "has_table": False,
            "has_fig": False,
        })

    def delete_cell(cell_id: int):
        st.session_state["nb_cells"] = [c for c in st.session_state["nb_cells"] if c["id"] != cell_id]

    def move_cell(cell_id: int, direction: int):
        cells = st.session_state["nb_cells"]
        idx = next((i for i, c in enumerate(cells) if c["id"] == cell_id), None)
        if idx is None:
            return
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(cells):
            return
        cells[idx], cells[new_idx] = cells[new_idx], cells[idx]
        st.session_state["nb_cells"] = cells

    # -------------------------
    # Save / Load notebook (JSON)
    # -------------------------
    st.subheader("Notatnik: zapis / odczyt")

    colS1, colS2, colS3 = st.columns([1, 1, 2])

    with colS1:
        if st.button("➕ Dodaj komórkę"):
            add_cell()

    with colS2:
        # Export notebook JSON
        nb_export = {
            "version": 1,
            "cells": [
                {"id": c["id"], "title": c["title"], "code": c["code"]}
                for c in st.session_state["nb_cells"]
            ]
        }
        st.download_button(
            "⬇️ Pobierz notebook.json",
            data=json.dumps(nb_export, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="notebook.json",
            mime="application/json",
        )

    with colS3:
        nb_file = st.file_uploader("Wczytaj notebook.json", type=["json"], key="nb_json")
        if nb_file is not None:
            try:
                loaded = json.loads(nb_file.read().decode("utf-8"))
                if not isinstance(loaded, dict) or "cells" not in loaded:
                    raise ValueError("Zły format JSON (brak 'cells').")

                new_cells = []
                max_id = 0
                for c in loaded["cells"]:
                    cid = int(c.get("id", 0))
                    max_id = max(max_id, cid)
                    new_cells.append({
                        "id": cid,
                        "title": str(c.get("title", f"Komórka {cid}")),
                        "code": str(c.get("code", "")),
                        "last_text": "",
                        "has_table": False,
                        "has_fig": False,
                    })

                if not new_cells:
                    raise ValueError("Brak komórek w JSON.")

                st.session_state["nb_cells"] = new_cells
                st.session_state["nb_next_id"] = max_id + 1
                st.success("Wczytano notatnik ✅")
            except Exception as e:
                st.error(f"Nie udało się wczytać JSON: {e}")

    st.markdown("---")

    # -------------------------
    # Execution engine
    # -------------------------
    def run_cell(cell_index: int):
        cell = st.session_state["nb_cells"][cell_index]
        code = cell["code"]

        ok, reason = code_is_safe(code)
        if not ok:
            cell["last_text"] = f"❌ Zablokowano: {reason}"
            cell["has_table"] = False
            cell["has_fig"] = False
            return None, None

        # Shared namespace across cells (like a notebook)
        if "nb_ns" not in st.session_state:
            st.session_state["nb_ns"] = {}

        ns = st.session_state["nb_ns"]

        # Provide df/pd/px each run
        global_env = {
            "__builtins__": SAFE_BUILTINS,
            "df": df,
            "pd": pd,
            "px": px,
        }
        # Keep previous variables in ns
        global_env.update(ns)

        local_env = {}

        try:
            exec(code, global_env, local_env)

            # merge locals into ns
            for k, v in local_env.items():
                global_env[k] = v

            # persist updated ns (excluding builtins)
            ns_new = {k: v for k, v in global_env.items() if k not in ["__builtins__"]}
            st.session_state["nb_ns"] = ns_new

            out_df = ns_new.get("out_df", None)
            fig = ns_new.get("fig", None)

            cell["last_text"] = "✅ Wykonano"
            cell["has_table"] = out_df is not None
            cell["has_fig"] = fig is not None

            return out_df, fig

        except Exception as e:
            cell["last_text"] = f"❌ Błąd: {e}"
            cell["has_table"] = False
            cell["has_fig"] = False
            return None, None

    topA, topB, topC = st.columns([1, 1, 2])

    with topA:
        run_all = st.button("▶️ Run all", type="primary")
    with topB:
        if st.button("🧹 Wyczyść namespace (zmienne)"):
            st.session_state["nb_ns"] = {}
            st.success("Wyczyszczono zmienne notebooka.")
    with topC:
        st.caption("Używaj `out_df = ...` (tabela) oraz/lub `fig = ...` (wykres Plotly). Zmienne zostają między komórkami.")

    if run_all:
        with st.spinner("Wykonuję wszystkie komórki..."):
            for i in range(len(st.session_state["nb_cells"])):
                run_cell(i)

    # -------------------------
    # Render cells
    # -------------------------
    st.subheader("Komórki")

    for i, cell in enumerate(st.session_state["nb_cells"]):
        with st.container(border=True):
            head1, head2, head3, head4, head5 = st.columns([2.2, 0.8, 0.8, 0.8, 0.6])

            with head1:
                cell["title"] = st.text_input(
                    "Tytuł",
                    value=cell["title"],
                    key=f"nb_title_{cell['id']}"
                )
            with head2:
                if st.button("⬆️", key=f"nb_up_{cell['id']}"):
                    move_cell(cell["id"], -1)
                    st.rerun()
            with head3:
                if st.button("⬇️", key=f"nb_down_{cell['id']}"):
                    move_cell(cell["id"], +1)
                    st.rerun()
            with head4:
                if st.button("▶️ Run", key=f"nb_run_{cell['id']}"):
                    out_df, fig = run_cell(i)
                else:
                    out_df = None
                    fig = None
            with head5:
                if st.button("🗑️", key=f"nb_del_{cell['id']}"):
                    delete_cell(cell["id"])
                    st.rerun()

            # code editor
            cell["code"] = st.text_area(
                "Kod",
                value=cell["code"],
                height=170,
                key=f"nb_code_{cell['id']}"
            )

            # status
            if cell.get("last_text"):
                st.write(cell["last_text"])

            # show last outputs using current namespace values (so you see latest)
            ns = st.session_state.get("nb_ns", {})
            out_df_current = ns.get("out_df", None)
            fig_current = ns.get("fig", None)

            # To avoid confusion: only show outputs if this cell says it produced them last time
            if cell.get("has_table") and out_df_current is not None:
                st.markdown("**out_df**")
                if isinstance(out_df_current, pd.DataFrame):
                    st.dataframe(out_df_current, use_container_width=True, height=320)
                else:
                    st.write(out_df_current)

            if cell.get("has_fig") and fig_current is not None:
                st.markdown("**fig**")
                st.plotly_chart(fig_current, use_container_width=True)

    # -------------------------
    # Quick commands (like notebook)
    # -------------------------
    st.markdown("---")
    st.subheader("Szybkie akcje")

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        if st.button("df.shape"):
            st.write(df.shape)
    with q2:
        if st.button("df.columns"):
            st.write(list(df.columns))
    with q3:
        if st.button("df.describe()"):
            st.dataframe(df.describe(include="all"), use_container_width=True)
    with q4:
        if st.button("df.head(20)"):
            st.dataframe(df.head(20), use_container_width=True)


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
# PPV z CSV: wwe.csv + wcw.csv
# ---------------------------------------------------------
elif page == "🤼 PPV (CSV: WWE/WCW)":
    st.title("🤼 PPV – dane z CSV (WWE/WWF/WWWF + WCW)")

    WWE_CSV = "wwe.csv"
    WCW_CSV = "wcw.csv"

    @st.cache_data(show_spinner=False)
    def load_csv_files(wwe_path: str, wcw_path: str) -> pd.DataFrame:
        if not os.path.exists(wwe_path):
            raise FileNotFoundError(f"Brak pliku: {wwe_path}")
        if not os.path.exists(wcw_path):
            raise FileNotFoundError(f"Brak pliku: {wcw_path}")

        df_wwe = pd.read_csv(wwe_path)
        df_wcw = pd.read_csv(wcw_path)

        # Normalizacja: Date -> datetime
        for df in (df_wwe, df_wcw):
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            else:
                raise ValueError("CSV musi mieć kolumnę 'Date'.")

        # Federacja / Promotion
        # WWE: zwykle ma kolumnę Promotion (WWWF/WWF/WWE)
        if "Promotion" not in df_wwe.columns:
            df_wwe["Promotion"] = "WWE"

        # WCW: jeśli brak Promotion -> ustaw WCW
        if "Promotion" not in df_wcw.columns:
            df_wcw["Promotion"] = "WCW"

        # Ujednolicenie kolumn (bezpieczne)
        wanted = [
            "Date", "Promotion", "Event", "Location",
            "Match", "Winner", "Loser", "WinType",
            "Stipulation", "Duration", "CardURL"
        ]
        for df in (df_wwe, df_wcw):
            for c in wanted:
                if c not in df.columns:
                    df[c] = ""

        df = pd.concat([df_wwe[wanted], df_wcw[wanted]], ignore_index=True)

        # Kolumna Year
        df["Year"] = df["Date"].dt.year

        # Tekst do wyszukiwania wrestlerów (Winner/Loser/Match)
        df["__search"] = (
            df["Winner"].astype(str).fillna("") + " | " +
            df["Loser"].astype(str).fillna("") + " | " +
            df["Match"].astype(str).fillna("")
        ).str.lower()

        return df

    # Wczytaj dane
    try:
        df = load_csv_files(WWE_CSV, WCW_CSV)
    except Exception as e:
        st.error(f"Nie udało się wczytać danych: {e}")
        st.info("Upewnij się, że w repo są pliki: `wwe.csv` oraz `wcw.csv` (w katalogu głównym).")
        st.stop()

    # --- Panel filtrów
    st.subheader("Filtry")

    col1, col2, col3 = st.columns([1.2, 1.2, 2.0])

    with col1:
        promotions = sorted(df["Promotion"].dropna().astype(str).unique().tolist())
        promo_sel = st.multiselect(
            "Federacja",
            options=promotions,
            default=promotions
        )

    with col2:
        years = df["Year"].dropna()
        if years.empty:
            st.error("Brak poprawnych dat w danych (kolumna Date).")
            st.stop()

        y_min = int(years.min())
        y_max = int(years.max())
        year_range = st.slider(
            "Zakres lat",
            min_value=y_min,
            max_value=y_max,
            value=(y_min, y_max),
            step=1
        )

    with col3:
        wrestler_query = st.text_input(
            "Wrestler (szukaj po Winner/Loser/Match) – możesz wpisać kilka, oddzielone przecinkiem",
            value=""
        )

    # --- Filtrowanie
    view = df.copy()

    if promo_sel:
        view = view[view["Promotion"].isin(promo_sel)]

    view = view[(view["Year"] >= year_range[0]) & (view["Year"] <= year_range[1])]

    # Wrestler filter (OR dla wielu nazw)
    wq = [x.strip().lower() for x in wrestler_query.split(",") if x.strip()]
    if wq:
        mask = False
        for term in wq:
            mask = mask | view["__search"].str.contains(term, na=False)
        view = view[mask]

    # --- Sortowanie
    view = view.sort_values(["Date", "Promotion", "Event"], ascending=[False, True, True])

    # --- Podsumowania
    st.subheader("Podsumowanie")
    cA, cB, cC = st.columns(3)
    with cA:
        st.metric("Wiersze (walki)", f"{len(view):,}")
    with cB:
        st.metric("Eventy", f"{view['Event'].nunique():,}")
    with cC:
        st.metric("Zakres dat", f"{view['Date'].min().date()} → {view['Date'].max().date()}")

    st.subheader("Wyniki")
    st.dataframe(
        view[[
            "Date", "Promotion", "Event", "Location",
            "Match", "Winner", "Loser", "WinType",
            "Stipulation", "Duration", "CardURL"
        ]],
        use_container_width=True,
        height=520
    )

    st.download_button(
        "⬇️ Pobierz przefiltrowane dane (CSV)",
        data=view.drop(columns=["__search"]).to_csv(index=False).encode("utf-8"),
        file_name="ppv_filtered.csv",
        mime="text/csv"
    )
