import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils_pdf import generate_pdf_report
from utils_estatistica import executar_bootstrap_confirmados

# Estilo global e paleta de cores para os gráficos mais profissionais
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#e1e4e8",
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "grid.color": "#f0f2f6",
    "grid.linestyle": "--",
    "legend.frameon": True,
    "legend.facecolor": "#ffffff",
    "legend.edgecolor": "#e1e4e8"
})
from io import BytesIO
import requests
import zipfile

st.set_page_config(page_title="Dashboard Acadêmico - COVID-19", page_icon="🧪", layout="wide")

# CSS customizado para tornar a interface mais limpa, moderna e profissional
st.markdown("""
<style>
    @keyframes virusDrift {
        0% { transform: scale(1) translate(0px, 0px); }
        50% { transform: scale(1.1) translate(-10px, -10px); }
        100% { transform: scale(1) translate(0px, 0px); }
    }

    /* Elemento HTML de imagem de fundo animada (Efeito Marca D'Água) */
    #custom-bg-image {
        position: fixed;
        top: 0; left: 0; 
        width: 100vw; height: 100vh;
        background-image: url("https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-position: center;
        z-index: 0;
        opacity: 0.12; /* Nível perfeito para marca d'água (não atrapalha a leitura) */
        pointer-events: none;
        animation: virusDrift 20s ease-in-out infinite;
    }

    /* Ocultar fundo apenas do Header para manter o layout limpo */
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Painel Central sobreposto à marca d'água */
    .block-container {
        position: relative;
        z-index: 1; /* Mantém o texto acima da marca d'água */
        background-color: transparent !important;
        padding-top: 2rem !important;
    }

    /* Barra lateral */
    [data-testid="stSidebar"] {
        z-index: 2;
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }
</style>
<div id="custom-bg-image"></div>
<style>
    /* Cards de métricas internos */
    div[data-testid="metric-container"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.15);
    }
    
    /* Títulos de métricas (labels) */
    div[data-testid="metric-container"] > div:nth-child(1) > div > p {
        color: color-mix(in srgb, var(--text-color) 70%, transparent);
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Valores das métricas */
    div[data-testid="metric-container"] > div:nth-child(2) > div > div {
        color: var(--text-color);
        font-weight: 800;
        font-size: 2rem;
    }
    
    /* Estilo de Botões */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid rgba(128, 128, 128, 0.3);
        transition: all 0.3s;
    }
    .stButton > button[kind="primary"] {
        background-color: #3498db;
        color: white;
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #2980b9;
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.3);
    }
    
    /* Abas do Streamlit */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: var(--secondary-background-color);
        border-radius: 8px 8px 0 0;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-bottom: none;
        padding: 10px 20px;
        color: color-mix(in srgb, var(--text-color) 70%, transparent);
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db !important;
        color: white !important;
        font-weight: 700;
    }
    
    /* Títulos gerais */
    h1, h2, h3 {
        color: var(--text-color);
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data_from_kaggle(token):
    url = "https://www.kaggle.com/api/v1/datasets/download/harshadapatil31/covid-19-statistics"
    headers = {"Authorization": f"Bearer {token}"}
    local_file = "COVID19_Statistics_200_Rows-1.csv"
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                for filename in z.namelist():
                    if filename.endswith(".csv"):
                        with z.open(filename) as f:
                            return pd.read_csv(f)
        else:
            st.warning(f"Erro ao baixar do Kaggle (HTTP {response.status_code}). Usando base de dados local (Fallback).")
            if os.path.exists(local_file):
                return pd.read_csv(local_file)
            return None
    except Exception as e:
        st.warning(f"Sem conexão com a internet ou Kaggle fora do ar. Carregando base local segura. (Erro: {e})")
        if os.path.exists(local_file):
            return pd.read_csv(local_file)
        return None

# =========================================================
# 1. Objetivo e contexto do trabalho
# =========================================================
st.title("🧪 Dashboard Acadêmico: Análise Estatística da COVID-19")
st.markdown(
    """
    Este painel apresenta uma análise descritiva, gráfica e inferencial (Bootstrap) 
    baseada nos dados da COVID-19.
    
    **Base de dados:** Estatísticas da COVID-19 — [Kaggle](https://www.kaggle.com/datasets/harshadapatil31/covid-19-statistics)
    """
)

# =========================================================
# Filtros Globais (Barra lateral)
# =========================================================
st.sidebar.header("🔍 Filtros Globais")

# Token oculto para autenticação no Kaggle lido do arquivo de segredos (secrets.toml)
try:
    token_kaggle = st.secrets["KAGGLE_TOKEN"]
except Exception:
    st.error("Token do Kaggle não encontrado! Crie o arquivo .streamlit/secrets.toml com a chave KAGGLE_TOKEN.")
    st.stop()

with st.spinner("Baixando e extraindo base de dados mais recente do Kaggle..."):
    df = load_data_from_kaggle(token_kaggle)

if df is None:
    st.stop()

# Traduzindo as colunas do dataset
df = df.rename(columns={
    "Country": "País",
    "Date": "Data",
    "Confirmed": "Confirmados",
    "Deaths": "Mortes",
    "Recovered": "Recuperados",
    "Active": "Ativos",
    "Tests_Conducted": "Testes Realizados",
    "Vaccination_Rate_%": "Taxa de Vacinação (%)"
})

# Dicionário de tradução dos países
dict_paises = {
    "United States": "Estados Unidos", "Brazil": "Brasil", "India": "Índia", "Russia": "Rússia", 
    "United Kingdom": "Reino Unido", "France": "França", "Germany": "Alemanha", "Spain": "Espanha", 
    "Italy": "Itália", "Argentina": "Argentina", "Colombia": "Colômbia", "Mexico": "México", 
    "Peru": "Peru", "South Africa": "África do Sul", "Iran": "Irã", "Poland": "Polônia", 
    "Turkey": "Turquia", "Ukraine": "Ucrânia", "Netherlands": "Holanda", "Belgium": "Bélgica", 
    "Canada": "Canadá", "Chile": "Chile", "Romania": "Romênia", "Czechia": "Chéquia", 
    "Israel": "Israel", "Portugal": "Portugal", "Sweden": "Suécia", "Switzerland": "Suíça", 
    "Japan": "Japão", "South Korea": "Coreia do Sul", "China": "China", "Australia": "Austrália", 
    "New Zealand": "Nova Zelândia", "Egypt": "Egito", "Saudi Arabia": "Arábia Saudita", 
    "United Arab Emirates": "Emirados Árabes Unidos", "Afghanistan": "Afeganistão", "Albania": "Albânia", 
    "Algeria": "Argélia", "Andorra": "Andorra", "Angola": "Angola", "Antigua and Barbuda": "Antígua e Barbuda", 
    "Armenia": "Armênia", "Austria": "Áustria", "Azerbaijan": "Azerbaijão", "Bahamas": "Bahamas", 
    "Bahrain": "Bahrein", "Bangladesh": "Bangladesh", "Barbados": "Barbados", "Belarus": "Bielorrússia", 
    "Belize": "Belize", "Benin": "Benim", "Bhutan": "Butão", "Bolivia": "Bolívia", 
    "Bosnia and Herzegovina": "Bósnia e Herzegovina", "Botswana": "Botsuana", "Brunei": "Brunei", 
    "Bulgaria": "Bulgária", "Burkina Faso": "Burkina Faso", "Burundi": "Burundi", "Cabo Verde": "Cabo Verde", 
    "Cambodia": "Camboja", "Cameroon": "Camarões", "Central African Republic": "República Centro-Africana", 
    "Chad": "Chade", "Comoros": "Comores", "Congo (Brazzaville)": "Congo", 
    "Congo (Kinshasa)": "República Democrática do Congo", "Costa Rica": "Costa Rica", 
    "Cote d'Ivoire": "Costa do Marfim", "Croatia": "Croácia", "Cuba": "Cuba", "Cyprus": "Chipre", 
    "Denmark": "Dinamarca", "Djibouti": "Djibuti", "Dominica": "Dominica", "Dominican Republic": "República Dominicana", 
    "Ecuador": "Equador", "El Salvador": "El Salvador", "Equatorial Guinea": "Guiné Equatorial", 
    "Eritrea": "Eritreia", "Estonia": "Estônia", "Eswatini": "Essuatíni", "Ethiopia": "Etiópia", 
    "Fiji": "Fiji", "Finland": "Finlândia", "Gabon": "Gabão", "Gambia": "Gâmbia", "Georgia": "Geórgia", 
    "Ghana": "Gana", "Greece": "Grécia", "Grenada": "Granada", "Guatemala": "Guatemala", "Guinea": "Guiné", 
    "Guinea-Bissau": "Guiné-Bissau", "Guyana": "Guiana", "Haiti": "Haiti", "Honduras": "Honduras", 
    "Hungary": "Hungria", "Iceland": "Islândia", "Indonesia": "Indonésia", "Iraq": "Iraque", 
    "Ireland": "Irlanda", "Jamaica": "Jamaica", "Jordan": "Jordânia", "Kazakhstan": "Cazaquistão", 
    "Kenya": "Quênia", "Kuwait": "Kuwait", "Kyrgyzstan": "Quirguistão", "Laos": "Laos", "Latvia": "Letônia", 
    "Lebanon": "Líbano", "Lesotho": "Lesoto", "Liberia": "Libéria", "Libya": "Líbia", "Liechtenstein": "Liechtenstein", 
    "Lithuania": "Lituânia", "Luxembourg": "Luxemburgo", "Madagascar": "Madagascar", "Malawi": "Malawi", 
    "Malaysia": "Malásia", "Maldives": "Maldivas", "Mali": "Mali", "Malta": "Malta", "Mauritania": "Mauritânia", 
    "Mauritius": "Maurício", "Moldova": "Moldávia", "Monaco": "Mônaco", "Mongolia": "Mongólia", 
    "Montenegro": "Montenegro", "Morocco": "Marrocos", "Mozambique": "Moçambique", "Namibia": "Namíbia", 
    "Nepal": "Nepal", "Nicaragua": "Nicarágua", "Niger": "Níger", "Nigeria": "Nigéria", 
    "North Macedonia": "Macedônia do Norte", "Norway": "Noruega", "Oman": "Omã", "Pakistan": "Paquistão", 
    "Panama": "Panamá", "Papua New Guinea": "Papua Nova Guiné", "Paraguay": "Paraguai", "Philippines": "Filipinas", 
    "Qatar": "Catar", "Rwanda": "Ruanda", "San Marino": "San Marino", "Senegal": "Senegal", "Serbia": "Sérvia", 
    "Seychelles": "Seicheles", "Sierra Leone": "Serra Leoa", "Singapore": "Singapura", "Slovakia": "Eslováquia", 
    "Slovenia": "Eslovênia", "Somalia": "Somália", "South Sudan": "Sudão do Sul", "Sri Lanka": "Sri Lanka", 
    "Sudan": "Sudão", "Syria": "Síria", "Taiwan*": "Taiwan", "Tajikistan": "Tajiquistão", "Tanzania": "Tanzânia", 
    "Thailand": "Tailândia", "Togo": "Togo", "Trinidad and Tobago": "Trinidad e Tobago", "Tunisia": "Tunísia", 
    "Uganda": "Uganda", "Uruguay": "Uruguai", "Uzbekistan": "Uzbequistão", "Venezuela": "Venezuela", 
    "Vietnam": "Vietnã", "Yemen": "Iêmen", "Zambia": "Zâmbia", "Zimbabwe": "Zimbábue"
}

# Aplicando a tradução aos países. Caso algum não esteja na lista, ele será mantido em inglês original.
if "País" in df.columns:
    df["País"] = df["País"].replace(dict_paises)

if "Confirmados" not in df.columns:
    st.error("O arquivo precisa ter uma coluna chamada 'Confirmados'.")
    st.stop()

df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
paises = sorted(df["País"].dropna().unique())
paises_selecionados = st.sidebar.multiselect("País", paises, default=paises)

data_min, data_max = df["Data"].min(), df["Data"].max()
periodo = st.sidebar.date_input(
    "Período", value=(data_min.date(), data_max.date()),
    min_value=data_min.date(), max_value=data_max.date(),
)
data_inicio, data_fim = periodo if len(periodo) == 2 else (data_min.date(), data_max.date())

df_filtrado = df[
    df["País"].isin(paises_selecionados)
    & (df["Data"].dt.date >= data_inicio)
    & (df["Data"].dt.date <= data_fim)
]

if df_filtrado.empty:
    st.warning("Nenhum registro encontrado para este filtro. Ajuste na barra lateral.")
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("Exportar Relatório")

# O módulo utils_pdf gera o relatório
# (A função generate_pdf_report foi importada de utils_pdf)

resultado_boot = st.session_state.get("bootstrap")
with st.spinner("Preparando PDF Executivo..."):
    pdf_data = generate_pdf_report(df_filtrado, resultado_boot)

if pdf_data:
    st.sidebar.download_button(
        "📄 Baixar Relatório (PDF)", 
        data=pdf_data, 
        file_name="relatorio_executivo_covid19.pdf", 
        mime="application/pdf"
    )

# =========================================================
# Estrutura de Abas
# =========================================================
aba_geral, aba_graficos, aba_bootstrap, aba_svm = st.tabs([
    "📊 Visão Geral e Estatística", 
    "📈 Análise Gráfica e Correlações", 
    "🎲 Inferência Estatística (Bootstrap)",
    "🤖 Machine Learning (SVM)"
])

# ---------------------------------------------------------
# Aba 1: Visão Geral e Estatística Descritiva
# ---------------------------------------------------------
with aba_geral:
    st.header("📊 Visão Geral e Estatística Descritiva")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Registros (após filtro)", f"{len(df_filtrado):,}")
    k2.metric("Casos Confirmados", f"{df_filtrado['Confirmados'].sum():,.0f}")
    k3.metric("Mortes", f"{df_filtrado['Mortes'].sum():,.0f}")
    k4.metric("Vacinação Média", f"{df_filtrado['Taxa de Vacinação (%)'].mean():.1f}%")
    
    st.divider()
    
    st.subheader("Estatísticas Descritivas")
    st.caption("Resumo matemático das variáveis numéricas contidas no conjunto de dados filtrado.")
    
    # Selecionar variáveis numéricas para descrever
    cols_numericas = ["Confirmados", "Mortes", "Recuperados", "Ativos", "Testes Realizados", "Taxa de Vacinação (%)"]
    cols_existentes = [c for c in cols_numericas if c in df_filtrado.columns]
    
    desc_df = df_filtrado[cols_existentes].describe()
    
    # Traduzindo os índices gerados pelo describe (que vêm em inglês por padrão)
    desc_df = desc_df.rename(index={
        "count": "Contagem",
        "mean": "Média",
        "std": "Desvio Padrão",
        "min": "Mínimo",
        "25%": "1º Quartil (25%)",
        "50%": "Mediana (50%)",
        "75%": "3º Quartil (75%)",
        "max": "Máximo"
    })
    
    st.dataframe(desc_df, use_container_width=True)
    
    csv_desc = desc_df.to_csv().encode('utf-8')
    st.download_button("📥 Baixar Estatísticas (CSV)", data=csv_desc, file_name="estatisticas_descritivas.csv", mime="text/csv")
    
    with st.expander("Visualizar Amostra dos Dados Filtrados", expanded=True):
        st.dataframe(df_filtrado.head(50), use_container_width=True)


# ---------------------------------------------------------
# Aba 2: Análise Gráfica e Correlações
# ---------------------------------------------------------
with aba_graficos:
    st.header("📈 Análise Gráfica e Correlações")
    
    st.subheader("Evolução Temporal dos Casos")
    # Agrupar por data
    evolucao = df_filtrado.groupby("Data")[["Confirmados", "Mortes", "Recuperados"]].sum().reset_index()
    
    fig_evolucao = px.line(
        evolucao, 
        x="Data", 
        y=["Confirmados", "Recuperados", "Mortes"],
        labels={"value": "Quantidade (Soma)", "variable": "Métrica", "Data": "Data"},
        color_discrete_map={"Confirmados": "#3498db", "Recuperados": "#2ecc71", "Mortes": "#e74c3c"}
    )
    fig_evolucao.update_layout(
        title="Evolução de Casos ao Longo do Tempo",
        hovermode="x unified",
        plot_bgcolor="white",
        legend_title_text="",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig_evolucao.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f0f2f6")
    fig_evolucao.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f0f2f6")
    st.plotly_chart(fig_evolucao, use_container_width=True)
    
    csv_evolucao = evolucao.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Baixar Dados da Evolução (CSV)", data=csv_evolucao, file_name="evolucao_temporal.csv", mime="text/csv")
    
    st.divider()
    
    col_graf1, col_graf2 = st.columns(2)
    
    with col_graf1:
        st.subheader("Matriz de Correlação")
        fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
        corr = df_filtrado[cols_existentes].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr, square=True)
        ax_corr.set_title("Correlação de Pearson")
        fig_corr.tight_layout()
        st.pyplot(fig_corr)
        plt.close(fig_corr)
        
        csv_corr = corr.to_csv().encode('utf-8')
        st.download_button("📥 Baixar Matriz de Correlação (CSV)", data=csv_corr, file_name="matriz_correlacao.csv", mime="text/csv")
        
    with col_graf2:
        st.subheader("Dispersão: Vacinação vs Mortes")
        fig_scatter = px.scatter(
            df_filtrado, 
            x="Taxa de Vacinação (%)", 
            y="Mortes", 
            color="País",
            size="Confirmados",
            hover_name="País",
            hover_data={"Data": True, "Confirmados": True, "País": False},
            opacity=0.7
        )
        fig_scatter.update_layout(
            title="Relação entre Taxa de Vacinação e Mortes",
            plot_bgcolor="white",
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5)
        )
        fig_scatter.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f0f2f6")
        fig_scatter.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f0f2f6")
        st.plotly_chart(fig_scatter, use_container_width=True)


# ---------------------------------------------------------
# Aba 3: Inferência Estatística (Bootstrap)
# ---------------------------------------------------------
with aba_bootstrap:
    st.header("🎲 Inferência Estatística com Bootstrap")
    st.markdown(
        """
        Nesta seção, aplicamos o **Bootstrap** (reamostragem com reposição) para estimar o 
        intervalo de confiança da **média de Casos Confirmados**, conforme especificado 
        no trabalho de Teoria do Aprendizado Estatístico.
        """
    )
    
    # Controle de filtro para não recalcular se não for necessário
    filtro_atual = (tuple(sorted(paises_selecionados)), str(data_inicio), str(data_fim))
    if st.session_state.get("bootstrap_filtro") != filtro_atual:
        st.session_state.pop("bootstrap", None)
    st.session_state["bootstrap_filtro"] = filtro_atual
    
    st.subheader("Parâmetros do Bootstrap")
    col_a, col_b = st.columns(2)
    n_iteracoes = col_a.number_input("Número de iterações", min_value=100, max_value=50000, value=10000, step=100)
    semente = col_b.number_input("Semente (reprodutibilidade)", value=42, step=1)
    
    if st.button("▶️ Executar Bootstrap para Casos Confirmados", type="primary"):
        with st.spinner("Realizando reamostragem computacional..."):
            # A lógica estatística pesada foi migrada para utils_estatistica.py
            resultado = executar_bootstrap_confirmados(df_filtrado, n_iteracoes, semente)
            st.session_state["bootstrap"] = resultado

    resultado = st.session_state.get("bootstrap")
    
    if resultado:
        st.divider()
        st.subheader("Resultados Obtidos")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Média Amostral Original", f"{resultado['media_original']:,.2f}")
        c2.metric("Média do Bootstrap", f"{resultado['media_bootstrap']:,.2f}")
        c3.metric("Intervalo de Confiança (95%)", f"[{resultado['ic_inferior']:,.0f}, {resultado['ic_superior']:,.0f}]")
        
        # Gráfico do Bootstrap
        fig_boot, ax_boot = plt.subplots(figsize=(10, 6))
        sns.histplot(resultado["medias_boot"], bins=50, kde=True, color="#3498db", ax=ax_boot)
        
        ax_boot.axvline(resultado["media_bootstrap"], color="#e74c3c", linestyle="-", linewidth=2,
                        label=f"Média Boot: {resultado['media_bootstrap']:,.0f}")
        ax_boot.axvline(resultado["ic_inferior"], color="#2ecc71", linestyle="--", linewidth=2,
                        label=f"Lim. Inf (2.5%): {resultado['ic_inferior']:,.0f}")
        ax_boot.axvline(resultado["ic_superior"], color="#2ecc71", linestyle="--", linewidth=2,
                        label=f"Lim. Sup (97.5%): {resultado['ic_superior']:,.0f}")
        
        ax_boot.set_title("Distribuição Empírica das Médias de Casos Confirmados")
        ax_boot.set_xlabel("Média de Casos Confirmados")
        ax_boot.set_ylabel("Frequência")
        ax_boot.legend(loc="upper right")
        ax_boot.grid(axis="y", alpha=0.3)
        fig_boot.tight_layout()

        st.pyplot(fig_boot)

        buf = BytesIO()
        fig_boot.savefig(buf, format="png", dpi=150)
        plt.close(fig_boot)

        st.download_button(
            "🖼️ Baixar gráfico do Bootstrap (PNG)",
            data=buf.getvalue(),
            file_name="bootstrap_covid19.png",
            mime="image/png",
        )
        
        df_boot = pd.DataFrame({"Medias_Bootstrap": resultado["medias_boot"]})
        csv_boot = df_boot.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Baixar Dados do Bootstrap (CSV)",
            data=csv_boot,
            file_name="dados_bootstrap.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------
# Aba 4: Machine Learning (SVM)
# ---------------------------------------------------------
with aba_svm:
    st.header("🤖 Support Vector Machine (SVM)")
    st.markdown(
        """
        Nesta seção, aplicamos o algoritmo **Support Vector Regression (SVR)** para modelar 
        e prever o número de **Mortes** com base em uma ou múltiplas variáveis preditoras (Regressão Múltipla).
        """
    )
    
    try:
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Preparar dados
        df_svm = df_filtrado.dropna(subset=['Confirmados', 'Mortes']).copy()
        
        if len(df_svm) < 10:
            st.warning("Dados insuficientes para treinar o modelo SVM. Por favor, amplie os filtros na barra lateral para incluir mais dados.")
        else:
            # Seleção Dinâmica de Features
            st.subheader("1. Seleção de Variáveis (Features)")
            opcoes_features = ["Confirmados", "Taxa de Vacinação (%)", "Testes Realizados", "Ativos", "Recuperados"]
            opcoes_validas = [col for col in opcoes_features if col in df_svm.columns]
            
            features_selecionadas = st.multiselect(
                "Escolha as variáveis para o modelo prever as Mortes (Regressão Múltipla):",
                opcoes_validas,
                default=["Confirmados"]
            )
            
            if not features_selecionadas:
                st.warning("Por favor, selecione pelo menos uma variável.")
                st.stop()
                
            # Limpar NaNs baseado nas colunas escolhidas
            df_ml = df_svm.dropna(subset=features_selecionadas + ['Mortes']).copy()
            if len(df_ml) < 10:
                st.warning("Filtro muito restritivo. Dados insuficientes para essas colunas específicas.")
                st.stop()

            # Features (X) e Target (y)
            X = df_ml[features_selecionadas].values
            y = df_ml['Mortes'].values
            
            # Padronização (Scaling) - crucial para SVM
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
            
            # Separar em treino e teste (70% treino, 30% teste)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)
            
            # Controles do Modelo SVM
            st.subheader("2. Parâmetros do Modelo")
            C_param = st.slider("Parâmetro de Regularização (C)", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
            
            if st.button("▶️ Treinar e Comparar Modelos SVR", type="primary"):
                with st.spinner("Treinando modelos SVM (Linear, Polinomial e RBF)..."):
                    
                    fig_svm, ax_svm = plt.subplots(figsize=(10, 6))
                    is_1d = (len(features_selecionadas) == 1)
                    
                    if is_1d:
                        # Se tiver apenas 1 feature, plotamos os pontos e a linha de regressão
                        ax_svm.scatter(X, y, color="#7f8c8d", label="Dados Reais", alpha=0.4, s=35)
                        X_grid = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                        X_grid_scaled = scaler_X.transform(X_grid)
                    else:
                        # Para Regressão Múltipla, plotamos Real x Previsto da base de Teste
                        y_test_orig_all = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
                        ax_svm.plot([y_test_orig_all.min(), y_test_orig_all.max()], 
                                    [y_test_orig_all.min(), y_test_orig_all.max()], 
                                    'k--', lw=2, label="Previsão Exata (Ideal)")
                    
                    cores_kernel = {"linear": "#3498db", "poly": "#2ecc71", "rbf": "#e74c3c"}
                    resultados_metricas = []
                    
                    for kernel_tipo, cor in cores_kernel.items():
                        model = SVR(kernel=kernel_tipo, C=C_param)
                        model.fit(X_train, y_train)
                        
                        y_pred_scaled = model.predict(X_test)
                        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                        y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
                        
                        mse = mean_squared_error(y_test_orig, y_pred)
                        r2 = r2_score(y_test_orig, y_pred)
                        resultados_metricas.append({"Kernel": kernel_tipo.capitalize(), "MSE": mse, "R² Score": r2})
                        
                        if is_1d:
                            y_grid_scaled = model.predict(X_grid_scaled)
                            y_grid = scaler_y.inverse_transform(y_grid_scaled.reshape(-1, 1)).ravel()
                            ax_svm.plot(X_grid, y_grid, color=cor, linewidth=3, label=f"SVR ({kernel_tipo.capitalize()})")
                        else:
                            ax_svm.scatter(y_test_orig, y_pred, color=cor, alpha=0.6, s=30, label=f"SVR ({kernel_tipo.capitalize()})")
                    
                    st.success("Modelos treinados com sucesso!")
                    
                    st.write("**Comparativo de Desempenho:**")
                    df_metricas = pd.DataFrame(resultados_metricas)
                    st.dataframe(df_metricas.style.format({"MSE": "{:,.0f}", "R² Score": "{:.4f}"}), use_container_width=True)
                    
                    st.divider()
                    st.subheader("3. Visualização do Ajuste")
                    
                    if is_1d:
                        ax_svm.set_xlabel(features_selecionadas[0])
                        ax_svm.set_ylabel("Mortes")
                        ax_svm.set_title(f"Comparação SVR: Previsão de Mortes baseado em {features_selecionadas[0]}")
                    else:
                        ax_svm.set_xlabel("Mortes Reais (Conjunto de Teste)")
                        ax_svm.set_ylabel("Mortes Previstas pelo Modelo")
                        ax_svm.set_title("Regressão Múltipla: Valores Reais vs Previstos")
                        
                    ax_svm.legend()
                    ax_svm.grid(True, alpha=0.3)
                    
                    fig_svm.tight_layout()
                    st.pyplot(fig_svm)
                    
                    # Botão para baixar gráfico
                    buf_svm = BytesIO()
                    fig_svm.savefig(buf_svm, format="png", dpi=150)
                    plt.close(fig_svm)
                    
                    st.download_button(
                        "🖼️ Baixar Gráfico Comparativo (PNG)",
                        data=buf_svm.getvalue(),
                        file_name="svm_comparativo.png",
                        mime="image/png",
                    )
    except ImportError:
        st.error("A biblioteca 'scikit-learn' não está instalada. Adicione 'scikit-learn' ao seu ambiente/requirements.txt e reinicie o app.")