import streamlit as st

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
        st.error(f"Erro ao gerar PDF: {e}")
        return None
