import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import requests
import zipfile

st.set_page_config(page_title="Dashboard Acadêmico - COVID-19", page_icon="🧪", layout="wide")

@st.cache_data(show_spinner=False)
def load_data_from_kaggle(token):
    url = "https://www.kaggle.com/api/v1/datasets/download/harshadapatil31/covid-19-statistics"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                for filename in z.namelist():
                    if filename.endswith(".csv"):
                        with z.open(filename) as f:
                            return pd.read_csv(f)
        else:
            st.error(f"Erro de conexão com Kaggle: HTTP {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Falha ao conectar com Kaggle: {e}")
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

# Token oculto para autenticação no Kaggle
token_kaggle = "KGAT_992b5f0ae705f1c3822dfbf3c48e1cef"

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

@st.cache_data(show_spinner=False)
def generate_pdf_report(df_to_export, bootstrap_result=None):
    try:
        from fpdf import FPDF
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, "Relatorio Executivo: Analise Estatistica da COVID-19", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(10)
        
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, "1. Visao Geral (Filtros Aplicados)", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 11)
        
        total_casos = df_to_export['Confirmados'].sum()
        total_mortes = df_to_export['Mortes'].sum()
        vac_media = df_to_export['Taxa de Vacinação (%)'].mean()
        
        pdf.cell(0, 6, f"- Total de Registros Analisados: {len(df_to_export)}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, f"- Total de Casos Confirmados: {total_casos:,.0f}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, f"- Total de Mortes: {total_mortes:,.0f}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, f"- Taxa Media de Vacinacao: {vac_media:.1f}%", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(10)
        
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, "2. Inferencia Estatistica (Bootstrap)", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 11)
        if bootstrap_result:
            pdf.cell(0, 6, "Estimativa para Media Populacional de Casos Confirmados.", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 6, f"- Media Amostral Original: {bootstrap_result['media_original']:,.0f}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 6, f"- Media do Bootstrap: {bootstrap_result['media_bootstrap']:,.0f}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 6, f"- Intervalo de Confianca (95%): de {bootstrap_result['ic_inferior']:,.0f} ate {bootstrap_result['ic_superior']:,.0f}", new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.cell(0, 6, "A analise de Bootstrap nao foi executada na sessao atual.", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(10)
        
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, "3. Top Paises (Maior n. de Casos Confirmados)", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 10)
        
        top_paises = df_to_export.groupby("País")[["Confirmados", "Mortes"]].sum().sort_values(by="Confirmados", ascending=False).head(5)
        
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(70, 8, "Pais", border=1)
        pdf.cell(60, 8, "Casos Confirmados", border=1, align="R")
        pdf.cell(60, 8, "Mortes", border=1, align="R", new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("helvetica", "", 10)
        for pais, row in top_paises.iterrows():
            pdf.cell(70, 8, str(pais), border=1)
            pdf.cell(60, 8, f"{row['Confirmados']:,.0f}", border=1, align="R")
            pdf.cell(60, 8, f"{row['Mortes']:,.0f}", border=1, align="R", new_x="LMARGIN", new_y="NEXT")
            
        return bytes(pdf.output())
    except Exception as e:
        import streamlit as st
        st.error(f"Erro ao gerar PDF: {e}")
        return None

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
aba_geral, aba_graficos, aba_bootstrap = st.tabs([
    "📊 Visão Geral e Estatística", 
    "📈 Análise Gráfica e Correlações", 
    "🎲 Inferência Estatística (Bootstrap)"
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
    st.dataframe(desc_df, use_container_width=True)
    
    csv_desc = desc_df.to_csv().encode('utf-8')
    st.download_button("📥 Baixar Estatísticas (CSV)", data=csv_desc, file_name="estatisticas_descritivas.csv", mime="text/csv")
    
    with st.expander("Visualizar Amostra dos Dados Filtrados"):
        st.dataframe(df_filtrado.head(50), use_container_width=True)


# ---------------------------------------------------------
# Aba 2: Análise Gráfica e Correlações
# ---------------------------------------------------------
with aba_graficos:
    st.header("📈 Análise Gráfica e Correlações")
    
    st.subheader("Evolução Temporal dos Casos")
    # Agrupar por data
    evolucao = df_filtrado.groupby("Data")[["Confirmados", "Mortes", "Recuperados"]].sum().reset_index()
    
    fig_evolucao, ax_evolucao = plt.subplots(figsize=(10, 5))
    ax_evolucao.plot(evolucao["Data"], evolucao["Confirmados"], label="Confirmados", color="#3498db")
    ax_evolucao.plot(evolucao["Data"], evolucao["Recuperados"], label="Recuperados", color="#2ecc71")
    ax_evolucao.plot(evolucao["Data"], evolucao["Mortes"], label="Mortes", color="#e74c3c")
    ax_evolucao.set_title("Evolução de Casos ao Longo do Tempo")
    ax_evolucao.set_xlabel("Data")
    ax_evolucao.set_ylabel("Quantidade (Soma)")
    ax_evolucao.legend()
    ax_evolucao.grid(axis='y', alpha=0.3)
    fig_evolucao.tight_layout()
    st.pyplot(fig_evolucao)
    plt.close(fig_evolucao)
    
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
        fig_scatter, ax_scatter = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            data=df_filtrado, 
            x="Taxa de Vacinação (%)", 
            y="Mortes", 
            hue="País", 
            alpha=0.7, 
            ax=ax_scatter
        )
        ax_scatter.set_title("Relação entre Taxa de Vacinação e Mortes")
        ax_scatter.grid(True, alpha=0.3)
        # Tenta não sobrepor muitas legendas
        ax_scatter.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
        fig_scatter.tight_layout()
        st.pyplot(fig_scatter)
        plt.close(fig_scatter)


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
            rng = np.random.default_rng(int(semente))
            casos_confirmados = df_filtrado["Confirmados"].values
            n = len(casos_confirmados)

            # Reamostragem vetorizada
            amostras = rng.choice(casos_confirmados, size=(int(n_iteracoes), n), replace=True)
            medias_boot = amostras.mean(axis=1)

            media_original = np.mean(casos_confirmados)
            media_bootstrap = np.mean(medias_boot)
            ic_inferior = np.percentile(medias_boot, 2.5)
            ic_superior = np.percentile(medias_boot, 97.5)

            st.session_state["bootstrap"] = {
                "n_iteracoes": int(n_iteracoes),
                "media_original": media_original,
                "media_bootstrap": media_bootstrap,
                "ic_inferior": ic_inferior,
                "ic_superior": ic_superior,
                "medias_boot": medias_boot,
            }

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
